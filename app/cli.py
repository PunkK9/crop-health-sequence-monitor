"""Command-line entry point: `python -m app.cli --help`."""

from __future__ import annotations

import argparse

from app.core import milestone_4, milestone_5, pattern_report


def cmd_next(args: argparse.Namespace) -> None:
    result = milestone_5(context=args.items, k=args.k)
    print(result["report"])


def cmd_patterns(args: argparse.Namespace) -> None:
    print(pattern_report(min_support=args.min_support, limit=args.limit))


def cmd_evaluate(_: argparse.Namespace) -> None:
    metrics = milestone_4()
    print("Held-out next-item evaluation")
    print(f"  cases: {metrics['n_cases']}")
    print(f"  top-1 accuracy: {metrics['top1_accuracy']:.1%}")
    print(f"  top-3 accuracy: {metrics['top3_accuracy']:.1%}")
    print(f"  MRR: {metrics['mrr']:.3f}")
    print(f"  popularity baseline MRR: {metrics['baseline_mrr']:.3f}")
    print(f"  final training loss: {metrics['final_loss']:.3f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="crop-health-sequence-monitor",
        description="Find co-occurring crop-health patterns and predict the next observation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("hello", help="check that everything is wired up")

    nxt = sub.add_parser("next", help="predict the next items in a crop or pest sequence")
    nxt.add_argument("items", nargs="+", help="observations so far, for example: rice 'brown planthopper'")
    nxt.add_argument("--k", type=int, default=5, help="how many next items to show")
    nxt.set_defaults(func=cmd_next)

    patterns = sub.add_parser("patterns", help="mine frequent co-occurring observations")
    patterns.add_argument("--min-support", type=float, default=0.03)
    patterns.add_argument("--limit", type=int, default=12)
    patterns.set_defaults(func=cmd_patterns)

    evaluate = sub.add_parser("evaluate", help="score next-item predictions on held-out sequences")
    evaluate.set_defaults(func=cmd_evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "hello":
        print("Crop Health Sequence Monitor is ready. Open README.md for your milestones.")
        return
    args.func(args)


if __name__ == "__main__":
    main()
