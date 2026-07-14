from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .competing_benchmark import temporal_competing_risk_benchmark
from .datasets.parlgov_competing import DEFAULT_MECHANISMS
from .engine import AnalogForecaster
from .hierarchical import HierarchicalRiskConfig, HierarchicalRiskForecaster
from .hybrid_competing import HybridCompetingForecaster
from .m2 import STRUCTURE_WEIGHTS
from .m3_cli import _date, _horizons
from .m3_competing_benchmark_cli import _mechanisms


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m3-2-hybrid",
        description=(
            "Explore a hybrid competing-risk model: hierarchical total exit "
            "risk plus analog conditional transition paths"
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
    parser.add_argument("--target-stride", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=3000)

    parser.add_argument("--tenure-bucket-days", type=int, default=730)
    parser.add_argument("--country-strength", type=float, default=400.0)
    parser.add_argument("--tenure-strength", type=float, default=400.0)
    parser.add_argument("--context-strength", type=float, default=400.0)

    parser.add_argument("--path-top-k", type=int, default=50)
    parser.add_argument("--path-prior-strength", type=float, default=8.0)
    parser.add_argument("--path-minimum-similarity", type=float, default=0.05)
    parser.add_argument(
        "--path-numeric-scale",
        choices=("range", "iqr"),
        default="iqr",
    )
    parser.add_argument(
        "--output",
        default="data/processed/m3_2_hybrid_competing_benchmark.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    total_model = HierarchicalRiskForecaster(
        HierarchicalRiskConfig(
            use_country=True,
            use_tenure=True,
            use_context=True,
            tenure_bucket_days=args.tenure_bucket_days,
            country_strength=args.country_strength,
            tenure_strength=args.tenure_strength,
            context_strength=args.context_strength,
        )
    )
    path_model = AnalogForecaster(
        feature_weights=STRUCTURE_WEIGHTS,
        prior_strength=args.path_prior_strength,
        top_k=args.path_top_k,
        minimum_similarity=args.path_minimum_similarity,
        numeric_scale=args.path_numeric_scale,
    )
    hybrid = HybridCompetingForecaster(
        total_model=total_model,
        mechanism_model=path_model,
    )
    report = temporal_competing_risk_benchmark(
        read_jsonl(args.cases),
        hybrid,  # type: ignore[arg-type]
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
