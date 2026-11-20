"""Score the continued-pretraining LM and bigram on every held-out field prefix."""
import json
from pathlib import Path

from app.core import Bigram, Metrics, split_fields, top_items
from app.llm import ItemPredictor
from app.storage import dataset_digest, load_fields, load_model, write_report


def render_evaluation(report: dict) -> str:
    """Write the aggregate metrics, common mistakes, and raw sample rankings."""
    lines = ["# Held-out next-item evaluation", "",
             f'{report["fields"]} fields and {report["model"]["count"]:,} proper prefixes (including empty prefixes).',
             "",
             "| Model | Top-1 | Top-3 | Top-5 | Perplexity |", "| --- | ---: | ---: | ---: | ---: |"]
    for name, key in (("Continued DistilGPT2", "model"), ("Smoothed bigram", "baseline")):
        row = report[key]
        lines.append(f'| {name} | {row["top_1"]:.2%} | {row["top_3"]:.2%} | {row["top_5"]:.2%} | {row["perplexity"]:.3f} |')
    lines += ["", "Perplexity is over the fixed item vocabulary for both models. The evaluation "
              "counts only real next events and omits end-of-field tokens.", "",
              "## Example predictions", "",
              "| Prefix (last 8 displayed) | True event | LM top 3 | Bigram top 3 |",
              "| --- | --- | --- | --- |"]
    for row in report["examples"]:
        prefix = " → ".join(row["prefix"][-8:]) or "(empty)"
        rankings = ["; ".join(f"{item} {prob:.1%}" for item, prob in row[name])
                    for name in ("model", "baseline")]
        lines.append(f'| {prefix} | {row["target"]} | {rankings[0]} | {rankings[1]} |')
    lines += ["", "## Most common LM errors", "", "| True event | Predicted event | Count |",
              "| --- | --- | ---: |"]
    for row in report["model"]["common_mistakes"]:
        lines.append(f'| {row["actual"]} | {row["predicted"]} | {row["count"]} |')
    return "\n".join(lines) + "\n"


def evaluate_run(data: Path, model_path: Path, results: Path) -> dict:
    """Evaluate held-out field histories against training-only bigram counts."""
    import torch

    torch.set_num_threads(4)
    model, tokenizer, metadata = load_model(model_path)
    if metadata["dataset_sha256"] != dataset_digest(data):
        raise ValueError("The dataset differs from the training input. Run train again or use the original data.")
    test = split_fields(load_fields(data), metadata["seed"])["test"]
    baseline = Bigram(**json.loads((model_path / "bigram.json").read_text(encoding="utf-8")))
    predictor = ItemPredictor(model, tokenizer)
    neural_metrics, count_metrics = Metrics(), Metrics()
    examples: dict[str, list[dict]] = {"hit": [], "miss": []}
    batch_size = 12
    for start in range(0, len(test), batch_size):
        fields = test[start:start + batch_size]
        sequences = [field.sequence() for field in fields]
        distributions = predictor.score_prefixes([sequence[:-1] for sequence in sequences])
        for field, events, probabilities in zip(fields, sequences, distributions, strict=True):
            for i, (target, neural) in enumerate(zip(events, probabilities, strict=True)):
                prefix = events[:i]
                counted = baseline.predict(prefix)
                neural_metrics.add(target, neural)
                count_metrics.add(target, counted)
                ranked = top_items(neural, 3)
                bucket = examples["hit" if ranked[0][0] == target else "miss"]
                if len(bucket) < 3 and len(prefix) >= 2 and all(sample["field"] != field.field_id for sample in bucket):
                    bucket.append({"field": field.field_id, "prefix": prefix, "target": target,
                                   "model": ranked, "baseline": top_items(counted, 3)})
    report = {"fields": len(test), "split_seed": metadata["seed"],
              "model": neural_metrics.report(), "baseline": count_metrics.report(),
              "examples": examples["hit"] + examples["miss"]}
    write_report(results, "evaluation", report, render_evaluation(report))
    return report
