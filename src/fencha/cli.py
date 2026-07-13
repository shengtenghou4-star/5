from __future__ import annotations

import argparse

from .demo import run_demo


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha",
        description="Auditable, time-aware historical forecasting engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the v0.1 walk-forward demonstration")
    demo.add_argument("--database", default="fencha.db", help="SQLite ledger path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        print(run_demo(args.database))


if __name__ == "__main__":
    main()
