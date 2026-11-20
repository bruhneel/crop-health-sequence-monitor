"""Regression tests for synthetic field sequence monitoring."""
def test_field_generation_loading_views_and_split(tmp_path):
    import pytest

    from app.core import FieldHistory, generate_fields, split_fields
    from app.storage import load_fields, write_fields

    fields = generate_fields(20)
    assert fields == generate_fields(20) and fields != generate_fields(20, 7)
    path = tmp_path / "fields.jsonl"
    write_fields(path, fields)
    assert load_fields(path) == fields
    field = FieldHistory("demo", (("wheat", "humid_season"), ("chickpea", "dry_season")))
    assert field.baskets()[0] == frozenset(("humid_season", "wheat"))
    assert field.sequence() == ("wheat", "humid_season", "chickpea", "dry_season")
    parts = split_fields(fields)
    assert parts == split_fields(list(reversed(fields)))
    ids = [{field.field_id for field in part} for part in parts.values()]
    assert len(set.union(*ids)) == sum(map(len, ids)) == len(fields)
    write_fields(path, [FieldHistory("bad", (("unknown",),))])
    with pytest.raises(ValueError, match="Unknown item"):
        load_fields(path)


def test_exact_support_counts_and_association_thresholds():
    import pytest

    from app.mining import mine_baskets

    baskets = [frozenset(items) for items in ("ab", "ab", "ac", "c")]
    report = mine_baskets(baskets, 0.5, 0.6, 3)
    counts = {tuple(row["items"]): row["count"] for row in report["itemsets"]}
    assert counts == {("a",): 3, ("b",): 2, ("c",): 2, ("a", "b"): 2}
    rule = next(row for row in report["rules"] if row["antecedent"] == ["a"])
    assert rule["consequent"] == ["b"] and rule["support"] == 0.5
    assert rule["confidence"] == pytest.approx(2 / 3)
    assert rule["lift"] == pytest.approx(4 / 3)
    stricter = mine_baskets(baskets, 0.5, 0.8)
    assert all(row["antecedent"] != ["a"] for row in stricter["rules"])
    assert mine_baskets(baskets, 0.51)["rules"] == []


def test_causal_training_and_local_checkpoint(tiny_pair, tmp_path):
    import math

    import torch

    from app.core import Bigram, FieldHistory
    from app.llm import continue_pretraining, encode_sequence, event_ids, event_token
    from app.storage import load_model, save_model

    model, tokenizer = tiny_pair
    mapping = event_ids(tokenizer)
    assert tokenizer.encode(event_token("wheat"), add_special_tokens=False) == [mapping["wheat"]]
    field = FieldHistory("demo", (("wheat", "humid_season", "scouting_visit"),))
    ids = encode_sequence(field.sequence(), tokenizer, mapping)
    history = continue_pretraining(
        model, [ids] * 8, [ids], pad=tokenizer.pad_token_id, epochs=8, batch_size=2,
        learning_rate=0.02, seed=42, log=False,
    )
    assert all(math.isfinite(row["train_loss"]) for row in history)
    assert history[-1]["val_loss"] < 0.6 * history[0]["val_loss"]
    save_model(tmp_path, model, tokenizer, {"seed": 42})
    restored, local_tokenizer, record = load_model(tmp_path)
    assert event_ids(local_tokenizer) == mapping and record == {"seed": 42}
    inputs = torch.tensor([ids])
    torch.testing.assert_close(restored(inputs).logits, model(inputs).logits, atol=0, rtol=0)
    baseline = Bigram.fit([field])
    assert max(baseline.predict(()), key=baseline.predict(()).get) == "wheat"
    assert max(baseline.predict(["wheat"]), key=baseline.predict(["wheat"]).get) == "humid_season"


def test_rank_and_perplexity_by_hand():
    import pytest

    from app.core import Metrics

    metrics = Metrics()
    metrics.add("wheat", {"wheat": 0.5, "rice": 0.3, "maize": 0.2})
    metrics.add("wheat", {"wheat": 0.25, "rice": 0.55, "maize": 0.2})
    result = metrics.report()
    assert (result["top_1"], result["top_3"], result["top_5"]) == (0.5, 1.0, 1.0)
    assert result["perplexity"] == pytest.approx((1 / (0.5 * 0.25))**0.5)
    assert result["common_mistakes"] == [{"actual": "wheat", "predicted": "rice", "count": 1}]


def test_test_fields_are_excluded_and_prefix_scores_are_causal(tiny_pair):
    import pytest

    from app.core import ITEMS, generate_fields, split_fields
    from app.llm import ItemPredictor

    parts = split_fields(generate_fields(30))
    assert not {row.field_id for row in parts["test"]} & {row.field_id for row in parts["train"]}
    model, tokenizer = tiny_pair
    predict = ItemPredictor(model, tokenizer)
    shorter, longer = predict.score_prefixes([["wheat"], ["wheat", "humid_season", "leaf_rust"]])
    assert len(shorter) == 2 and len(longer) == 4
    assert set(shorter[-1]) == set(ITEMS) and sum(shorter[-1].values()) == pytest.approx(1)
    assert shorter[-1] == pytest.approx(longer[1], abs=1e-7)
    assert predict.predict(["wheat"]) == pytest.approx(shorter[-1], abs=1e-7)


def test_cli_end_to_end_on_tiny_local_model(tiny_pair, tmp_path, monkeypatch, capsys):
    from pathlib import Path

    import pytest

    from app.cli import main
    from app.storage import save_model

    model, tokenizer = tiny_pair
    save_model(tmp_path / "base", model, tokenizer, {})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(["suggest", "--observed", "wheatt"])
    assert "Did you mean wheat?" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["suggest", "--observed", "   "])
    assert "Enter at least one item" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["suggest", "--observed", "wheat"])
    assert "python -m app.cli train" in capsys.readouterr().err

    main(["generate-data", "--count", "15"])
    main(["mine", "--min-support", "0.05"])
    main(["train", "--base", str(tmp_path / "base"), "--epochs", "1",
          "--batch-size", "4", "--threads", "1"])
    main(["evaluate"])
    main(["suggest", "--observed", "wheat, humid_season", "--top", "3"])
    output = capsys.readouterr().out
    assert "LM top-1" in output and "Likely next items" in output
    assert "Matching basket rules" in output and "not agronomic advice" in output
    assert Path("results/evaluation.json").is_file() and Path("results/patterns.json").is_file()
    inputs = iter(["wheat", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    main(["interactive", "--top", "1"])
    assert "Likely next items" in capsys.readouterr().out
