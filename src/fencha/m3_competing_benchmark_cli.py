from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .competing_benchmark import temporal_competing_risk_benchmark
from .datasets.parlgov_competing import DEFAULT_MECHANISMS
from .engine import AnalogForecaster
from .m2 import STRUCTURE_WEIGHTS
from .m3_cli import _date, _horizons


def _mechanisms(value: str) -> tuple[str, ...]:
    names = tuple(sorted(set(item.strip() for item in value.split(",") if item.strip())))
    if not names:
        raise argparse.ArgumentTypeError("mechanisms must be comma-separated names")
    return names


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m3-competing-benchmark",
        description="Run a time-safe competing-risk leader-exit benchmark",
    )
    parser.add_argument(
        "--cases",
        default="data/processed/parlgov_leader_competing_risks.jsonl",
    )
    parser.add_argument("--holdout-start", type=_date, default=_date("2015-01-01"))
    parser.add_argument("--horizons", type=_horizons, default=(30, 90, 180, 365))
    parser.add_argument(
        "--mechanisms",
        type=_mechanisms,
        default=DEFAULT_MECHANISMS,
    )
    parser.add_argument("--minimum-training-cases", type=int, default=500)
    parser.add_argument("--target-stride", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=3000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prior-strength", type=float, default=8.0)
    parser.add_argument("--minimum-similarity", type=float, default=0.05)
    parser.add_argument(
        "--numeric-scale",
        choices=("range", "iqr"),
        default="iqr",
    )
    parser.add_argument(
        "--output",
        default="data/processed/m3_competing_benchmark.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = AnalogForecaster(
        feature_weights=STRUCTURE_WEIGHTS,
        prior_strength=args.prior_strength,
        top_k=args.top_k,
        minimum_similarity=args.minimum_similarity,
        numeric_scale=args.numeric_scale,
    )
    report = temporal_competing_risk_benchmark(
        read_jsonl(args.cases),
        model,
        holdout_start=datetime.combine(
            args.holdout_start,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ),
        mechanisms=args.mechanisms,
        horizons=args.horizons,
        minimum_training_cases=args.minimum_training_cases,
        target_stride=args.target_stride,
        max_history=args.max_history,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"snapshots={report.snapshots} exits={report.exit_observations} "
        f"total_skill={report.total_brier_skill_vs_baseline:.6f} "
        f"mechanism_skill={report.mechanism_brier_skill_vs_baseline:.6f} "
        f"conditional_log_skill={report.conditional_log_loss_skill_vs_baseline:.6f} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
