from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .m2_1 import compare_matched_architecture, summarize_diagnostics


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m2-1",
        description="Run matched-architecture M2.1 GDELT diagnostics",
    )
    parser.add_argument(
        "--cases",
        default="data/processed/parlgov_gdelt_weekly.jsonl",
    )
    parser.add_argument(
        "--holdout-start",
        type=_date,
        default=date(2022, 1, 1),
    )
    parser.add_argument("--minimum-training-cases", type=int, default=300)
    parser.add_argument("--target-stride", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=3000)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--prior-strength", type=float, default=8.0)
    parser.add_argument("--minimum-similarity", type=float, default=0.05)
    parser.add_argument("--gdelt-multiplier", type=float, default=1.0)
    parser.add_argument(
        "--signal-family",
        choices=("none", "volume", "conflict", "tone", "all"),
        default="all",
    )
    parser.add_argument(
        "--numeric-scale",
        choices=("range", "iqr"),
        default="range",
    )
    parser.add_argument(
        "--minimum-group-predictions",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--report-output",
        default="data/processed/m2_1_matched_report.json",
    )
    parser.add_argument(
        "--diagnostics-output",
        default="data/processed/m2_1_prediction_diagnostics.jsonl",
    )
    parser.add_argument(
        "--subgroups-output",
        default="data/processed/m2_1_subgroups.json",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = read_jsonl(args.cases)
    report, diagnostics = compare_matched_architecture(
        cases,
        holdout_start=datetime.combine(
            args.holdout_start,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ),
        minimum_training_cases=args.minimum_training_cases,
        target_stride=args.target_stride,
        max_history=args.max_history,
        top_k=args.top_k,
        prior_strength=args.prior_strength,
        minimum_similarity=args.minimum_similarity,
        gdelt_multiplier=args.gdelt_multiplier,
        signal_family=args.signal_family,
        numeric_scale=args.numeric_scale,
    )

    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    diagnostics_path = Path(args.diagnostics_output)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.write_text(
        "".join(
            json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) + "\n"
            for item in diagnostics
        ),
        encoding="utf-8",
    )

    subgroup_report = summarize_diagnostics(
        diagnostics,
        minimum_group_predictions=args.minimum_group_predictions,
    )
    subgroups_path = Path(args.subgroups_output)
    subgroups_path.parent.mkdir(parents=True, exist_ok=True)
    subgroups_path.write_text(
        json.dumps(subgroup_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    print(
        f"predictions={report.predictions} positives={report.positives} "
        f"family={report.signal_family} scale={report.numeric_scale} "
        f"structure_brier={report.structure_analog.brier_score:.6f} "
        f"gdelt_brier={report.gdelt_analog.brier_score:.6f} "
        f"gdelt_skill_vs_structure={report.gdelt_brier_skill_vs_structure:.6f} "
        f"mean_abs_delta={report.mean_absolute_probability_delta:.6f} "
        f"neighbor_overlap={report.mean_neighbor_overlap:.6f} "
        f"report={report_path} diagnostics={diagnostics_path} "
        f"subgroups={subgroups_path}"
    )


if __name__ == "__main__":
    main()
