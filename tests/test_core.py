"""Regression tests for synthetic field sequence monitoring."""
import pytest

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


# Milestone 2: Mine frequent co-occurring patterns using `thilimarketbasket`.
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Train a small LM on these sequences for next-item prediction.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Evaluate model's predictions on new, unseen agricultural sequences.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement a CLI to query likely next items based on current observations.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

