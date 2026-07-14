from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .conditioned_paths import ExitConditionedPathForecaster
from .datasets.parlgov_competing import DEFAULT_MECHANISMS
from .engine import AnalogForecaster
from .event_balanced_paths import EventBalancedPathForecaster
from .event_path_audit import temporal_event_path_audit
from .m2 import STRUCTURE_WEIGHTS
from .m3_cli import _date, _horizons
from .m3_competing_benchmark_cli import _mechanisms


def _analog(
    *,
    top_k: int,
    prior_strength: float,
    minimum_similarity: float,
    numeric_scale: str,
) -> AnalogForecaster:
    return AnalogForecaster(
        feature_weights=STRUCTURE_WEIGHTS,
        prior_strength=prior_strength,
        top_k=top_k,
        minimum_similarity=minimum_similarity,
        numeric_scale=numeric_scale,  # type: ignore[arg-type]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m3-4-event-audit",
        description=(
            "Audit transition-path models once per unique leader exit event, "
            "with event-clustered uncertainty"
        ),
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
    parser.add_argument("--minimum-exit-events", type=int, default=20)
    parser.add_argument(
        "--max-history",
        type=int,
        default=0,
        help=(
            "maximum resolved snapshots per mechanism domain; 0 keeps the full "
            "history, which is the default for rare exit-event auditing"
        ),
    )
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prior-strength", type=float, default=8.0)
    parser.add_argument("--minimum-similarity", type=float, default=0.05)
    parser.add_argument(
        "--numeric-scale",
        choices=("range", "iqr"),
        default="iqr",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260715)
    parser.add_argument(
        "--output",
        default="data/processed/m3_4_event_path_audit.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_history < 0:
        raise SystemExit("max-history must be zero or positive")
    max_history = None if args.max_history == 0 else args.max_history
    conditioned = ExitConditionedPathForecaster(
        _analog(
            top_k=args.top_k,
            prior_strength=args.prior_strength,
            minimum_similarity=args.minimum_similarity,
            numeric_scale=args.numeric_scale,
        )
    )
    event_balanced = EventBalancedPathForecaster(
        _analog(
            top_k=args.top_k,
            prior_strength=args.prior_strength,
            minimum_similarity=args.minimum_similarity,
            numeric_scale=args.numeric_scale,
        )
    )
    report = temporal_event_path_audit(
        read_jsonl(args.cases),
        {
            "conditioned_snapshots": conditioned,
            "event_balanced": event_balanced,
        },
        holdout_start=datetime.combine(
            args.holdout_start,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ),
        mechanisms=args.mechanisms,
        horizons=args.horizons,
        minimum_training_cases=args.minimum_training_cases,
        minimum_exit_events=args.minimum_exit_events,
        max_history=max_history,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    summary = " ".join(
        f"{item.model}_accuracy={item.metrics.accuracy:.6f} "
        f"{item.model}_brier_delta={item.brier_improvement:.6f}"
        for item in report.models
    )
    print(
        f"events={report.unique_exit_events} "
        f"records={report.selected_event_horizon_predictions} "
        f"baseline_accuracy={report.baseline.accuracy:.6f} "
        f"{summary} output={output}"
    )


if __name__ == "__main__":
    main()
