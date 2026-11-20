"""Command-line entry point: `python -m app.cli --help`."""
import argparse
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crop-health-sequence-monitor", description="Farmers track crop health to predict co-occurring pests or optimal crop sequences. The tool identifies common patterns and suggests next actions.")
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
    args = parser.parse_args(argv)
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


if __name__ == "__main__":
    main()
