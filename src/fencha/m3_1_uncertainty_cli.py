from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .hierarchical import HierarchicalRiskConfig, HierarchicalRiskForecaster
from .m3_cli import _date, _horizons
from .survival_uncertainty import (
    country_cluster_bootstrap,
    paired_survival_snapshot_scores,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m3-1-uncertainty",
        description="Estimate country-cluster uncertainty for the locked M3.1 model",
    )
    parser.add_argument(
        "--cases",
        default="data/processed/parlgov_leader_survival.jsonl",
    )
    parser.add_argument("--holdout-start", type=_date, default=_date("2015-01-01"))
    parser.add_argument("--horizons", type=_horizons, default=(30, 90, 180, 365))
    parser.add_argument("--minimum-training-cases", type=int, default=500)
    parser.add_argument("--target-stride", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=3000)
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument(
        "--scores-output",
        default="data/processed/m3_1_snapshot_scores.jsonl",
    )
    parser.add_argument(
        "--report-output",
        default="data/processed/m3_1_cluster_bootstrap.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    model = HierarchicalRiskForecaster(
        HierarchicalRiskConfig(
            use_country=True,
            use_tenure=True,
            use_context=True,
            tenure_bucket_days=730,
            country_strength=400.0,
            tenure_strength=400.0,
            context_strength=400.0,
        )
    )
    scores = paired_survival_snapshot_scores(
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
    report = country_cluster_bootstrap(
        scores,
        replicates=args.replicates,
        seed=args.seed,
        confidence=args.confidence,
    )

    scores_path = Path(args.scores_output)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(
        "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in scores
        ),
        encoding="utf-8",
    )
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    print(
        f"clusters={report.clusters} snapshots={report.snapshots} "
        f"skill={report.observed_brier_skill:.6f} "
        f"skill_ci=[{report.skill_ci_low:.6f},{report.skill_ci_high:.6f}] "
        f"probability_better={report.probability_model_better:.3f} "
        f"report={report_path}"
    )


if __name__ == "__main__":
    main()
