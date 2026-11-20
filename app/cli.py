"""Command-line entry point: `python -m app.cli --help`."""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="crop-health-sequence-monitor", description="Farmers track crop health to predict co-occurring pests or optimal crop sequences. The tool identifies common patterns and suggests next actions.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")
    # TODO: add your own commands here as you work through the milestones.
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Crop Health Sequence Monitor is ready. Open README.md for your milestones.")


if __name__ == "__main__":
    main()
