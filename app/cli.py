"""Command-line entry point: `python -m app.cli --help`."""
import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crop-health-sequence-monitor", description="Mine co-occurrences and rank next events in synthetic field histories.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    generate = sub.add_parser("generate-data", help="write a synthetic field history dataset")
    generate.add_argument("--count", type=int, default=1200)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", type=Path, default=Path("data/sequences.jsonl"))
    mine = sub.add_parser("mine", help="mine associations from training-season baskets")
    mine.add_argument("--data", type=Path, default=Path("data/sequences.jsonl"))
    mine.add_argument("--min-support", type=float, default=0.04)
    mine.add_argument("--min-confidence", type=float, default=0.55)
    mine.add_argument("--max-size", type=int, default=4)
    mine.add_argument("--seed", type=int, default=42)
    mine.add_argument("--results", type=Path, default=Path("results"))
    train = sub.add_parser("train", help="continue pretraining the local CPU language model")
    train.add_argument("--data", type=Path, default=Path("data/sequences.jsonl"))
    train.add_argument("--output", type=Path, default=Path("artifacts/model"))
    train.add_argument("--results", type=Path, default=Path("results"))
    train.add_argument("--base", default="distilbert/distilgpt2")
    train.add_argument("--epochs", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=16)
    train.add_argument("--learning-rate", type=float, default=0.00015)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--seed", type=int, default=42)
    evaluate = sub.add_parser("evaluate", help="score every prefix of held-out fields")
    evaluate.add_argument("--data", type=Path, default=Path("data/sequences.jsonl"))
    evaluate.add_argument("--model", type=Path, default=Path("artifacts/model"))
    evaluate.add_argument("--results", type=Path, default=Path("results"))
    compare = sub.add_parser("compare-support", help="compare itemsets and rules at different support floors")
    compare.add_argument("--data", type=Path, default=Path("data/sequences.jsonl"))
    compare.add_argument("--results", type=Path, default=Path("results"))
    compare.add_argument("--seed", type=int, default=42)
    compare.add_argument("--supports", nargs="+", type=float, default=[0.02, 0.04, 0.08])
    for name, help_text in (("suggest", "score a comma-separated event prefix"),
                            ("interactive", "score several prefixes in a prompt loop")):
        query = sub.add_parser(name, help=help_text)
        query.add_argument("--model", type=Path, default=Path("artifacts/model"))
        query.add_argument("--patterns", type=Path, default=Path("results/patterns.json"))
        query.add_argument("--top", type=int, default=5)
        query.add_argument("--explain", action="store_true",
                           help="show matching rule evidence for each ranked event")
        if name == "suggest":
            query.add_argument("--observed", required=True)
    args = parser.parse_args(argv)
    if args.command in {"suggest", "interactive"}:
        try:
            run_query(args)
        except (ValueError, OSError, RuntimeError) as exc:
            parser.error(str(exc))
    if args.command == "compare-support":
        from app.core import split_fields
        from app.mining import compare_support, render_support
        from app.storage import load_fields, write_report

        try:
            train = split_fields(load_fields(args.data), args.seed)["train"]
            report = compare_support(train, args.supports)
            markdown = render_support(report)
            write_report(args.results, "support_comparison", report, markdown)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(markdown, end="")
    if args.command == "evaluate":
        from app.evaluate import evaluate_run

        try:
            report = evaluate_run(args.data, args.model, args.results)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f'LM top-1 {report["model"]["top_1"]:.2%}; bigram top-1 {report["baseline"]["top_1"]:.2%}.')
    if args.command == "train":
        from app.train import train_run

        try:
            train_run(args.data, args.output, args.results, args.base, args.seed, args.epochs,
                      args.batch_size, args.learning_rate, args.threads)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
    if args.command == "mine":
        from app.core import split_fields
        from app.mining import mine_training, render_patterns
        from app.storage import load_fields, write_report

        try:
            fields = split_fields(load_fields(args.data), args.seed)["train"]
            report = mine_training(fields, args.min_support, args.min_confidence, args.max_size)
            write_report(args.results, "patterns", report, render_patterns(report))
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print(f'Mined {len(report["itemsets"])} itemsets and {len(report["rules"])} rules.')
    if args.command == "generate-data":
        from app.core import generate_fields
        from app.storage import write_fields

        try:
            write_fields(args.output, generate_fields(args.count, args.seed))
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Wrote {args.count} synthetic field histories to {args.output}.")
    if args.command == "hello":
        print("Crop Health Sequence Monitor is ready. Open README.md for your milestones.")


def parse_observed(value: str) -> list[str]:
    """Parse chronological comma-separated identifiers and validate the vocabulary."""
    from app.core import validate_items

    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    validate_items(items)
    return items


def run_query(args: argparse.Namespace) -> None:
    """Load models once and show rankings, associations, and log-review actions."""
    import json

    import torch

    from app.core import ITEMS, top_items
    from app.llm import ItemPredictor
    from app.storage import load_model

    if not 1 <= args.top <= len(ITEMS):
        raise ValueError(f"--top must be between 1 and {len(ITEMS)}.")
    pending = parse_observed(args.observed) if args.command == "suggest" else None
    torch.set_num_threads(4)
    model, tokenizer, _ = load_model(args.model)
    predictor = ItemPredictor(model, tokenizer)
    if not args.patterns.is_file():
        raise FileNotFoundError("No mined patterns found. Run python -m app.cli mine first.")
    report = json.loads(args.patterns.read_text(encoding="utf-8"))

    def show(observed: list[str]) -> None:
        ranked = top_items(predictor.predict(observed), args.top)
        present = set(observed)
        matched = [
            row for row in report["rules"]
            if set(row["antecedent"]) <= present and not set(row["consequent"]) <= present
        ]
        print("Likely next items (synthetic model)")
        for item, probability in ranked:
            print(f"  {item:24s} {probability:6.2%}")
        print("Matching basket rules (supplied items treated as one window)")
        for row in matched[:5]:
            print(f'  {", ".join(row["antecedent"])} -> {", ".join(row["consequent"])}'
                  f' | support {row["support"]:.2%} | confidence {row["confidence"]:.2%}'
                  f' | lift {row["lift"]:.2f}')
        if not matched:
            print("  No rule matched these observed items.")
        if args.explain:
            print("Basket rule evidence by predicted item")
            for item, _ in ranked:
                evidence = [row for row in matched if item in row["consequent"]]
                if not evidence:
                    print(f"  {item}: No matching rule.")
                for row in evidence[:2]:
                    print(f'  {item}: {", ".join(row["antecedent"])} -> {", ".join(row["consequent"])}'
                          f' | support {row["support"]:.2%} | confidence {row["confidence"]:.2%}'
                          f' | lift {row["lift"]:.2f}')
        print("Suggested next actions (rule-based log review; not agronomic advice)")
        for item, _ in ranked:
            print(f"  {item}: {ITEMS[item].next_step}")

    if pending is not None:
        show(pending)
        return
    print("Enter comma-separated item IDs in chronological order, or /quit to exit.")
    while True:
        try:
            line = input("Field> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() == "/quit":
            break
        try:
            show(parse_observed(line))
        except ValueError as exc:
            print(exc)


if __name__ == "__main__":
    main()
