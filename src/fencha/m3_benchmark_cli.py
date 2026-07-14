from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .engine import AnalogForecaster
from .m2 import STRUCTURE_WEIGHTS
from .survival_benchmark import temporal_survival_benchmark


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _horizons(value: str) -> tuple[int, ...]:
    try:
        horizons = tuple(
            sorted({int(item.strip()) for item in value.split(",") if item.strip()})
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "horizons must be comma-separated integers"
        ) from exc
    if not horizons or any(value <= 0 for value in horizons):
        raise argparse.ArgumentTypeError("horizons must be positive")
    return horizons


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m3-benchmark",
        description="Run a time-safe multi-horizon leader-survival benchmark",
    )
    parser.add_argument(
        "--cases",
        default="data/processed/parlgov_leader_survival.jsonl",
    )
    parser.add_argument("--holdout-start", type=_date, default=date(2015, 1, 1))
    parser.add_argument("--horizons", type=_horizons, default=(30, 90, 180, 365))
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
        default="data/processed/m3_survival_benchmark.json",
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
    report = temporal_survival_benchmark(
        read_jsonl(args.cases),
        model,
        holdout_start=datetime.combine(
            args.holdout_start,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ),
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
        f"snapshots={report.snapshots} crossings={report.raw_crossing_curves} "
        f"crossing_rate={report.raw_crossing_rate:.2%} "
        f"raw_integrated_brier={report.raw_integrated_brier:.6f} "
        f"adjusted_integrated_brier={report.adjusted_integrated_brier:.6f} "
        f"adjustment_skill={report.adjusted_integrated_brier_skill_vs_raw:.6f} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
