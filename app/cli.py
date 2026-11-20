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
    args = parser.parse_args(argv)
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
