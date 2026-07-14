from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .m2_1 import summarize_diagnostics
from .m2_1_select import select_and_evaluate


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _csv_ints(value: str) -> tuple[int, ...]:
    try:
        result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one integer is required")
    return result


def _csv_floats(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated numbers") from exc
    if not result:
        raise argparse.ArgumentTypeError("at least one number is required")
    return result


def _csv_strings(value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in value.split(",") if item.strip())
    if not result:
        raise argparse.ArgumentTypeError("at least one value is required")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m2-1-select",
        description=(
            "Select M2.1 settings on pre-holdout data and evaluate the locked "
            "specification once on the final holdout"
        ),
    )
    parser.add_argument(
        "--cases",
        default="data/processed/parlgov_gdelt_weekly.jsonl",
    )
    parser.add_argument("--validation-start", type=_date, default=date(2021, 1, 1))
    parser.add_argument("--validation-end", type=_date, default=date(2022, 1, 1))
    parser.add_argument("--holdout-start", type=_date, default=date(2022, 1, 1))
    parser.add_argument("--minimum-training-cases", type=int, default=150)
    parser.add_argument("--validation-target-stride", type=int, default=2)
    parser.add_argument("--holdout-target-stride", type=int, default=1)
    parser.add_argument("--max-history", type=int, default=3000)
    parser.add_argument("--top-k-candidates", type=_csv_ints, default=(25, 50, 75))
    parser.add_argument(
        "--numeric-scale-candidates",
        type=_csv_strings,
        default=("range", "iqr"),
    )
    parser.add_argument(
        "--signal-family-candidates",
        type=_csv_strings,
        default=("volume", "conflict", "tone", "all"),
    )
    parser.add_argument(
        "--gdelt-multiplier-candidates",
        type=_csv_floats,
        default=(0.1, 0.25, 0.5, 1.0),
    )
    parser.add_argument("--prior-strength", type=float, default=8.0)
    parser.add_argument("--minimum-similarity", type=float, default=0.05)
    parser.add_argument("--minimum-group-predictions", type=int, default=10)
    parser.add_argument(
        "--selection-output",
        default="data/processed/m2_1_selection_report.json",
    )
    parser.add_argument(
        "--final-report-output",
        default="data/processed/m2_1_selected_holdout_report.json",
    )
    parser.add_argument(
        "--diagnostics-output",
        default="data/processed/m2_1_selected_diagnostics.jsonl",
    )
    parser.add_argument(
        "--subgroups-output",
        default="data/processed/m2_1_selected_subgroups.json",
    )
    return parser


def _midnight(value: date) -> datetime:
    return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)


def main() -> None:
    args = build_parser().parse_args()
    cases = read_jsonl(args.cases)
    selection, diagnostics = select_and_evaluate(
        cases,
        validation_start=_midnight(args.validation_start),
        validation_end=_midnight(args.validation_end),
        holdout_start=_midnight(args.holdout_start),
        minimum_training_cases=args.minimum_training_cases,
        validation_target_stride=args.validation_target_stride,
        holdout_target_stride=args.holdout_target_stride,
        max_history=args.max_history,
        top_k_candidates=args.top_k_candidates,
        numeric_scale_candidates=args.numeric_scale_candidates,
        signal_family_candidates=args.signal_family_candidates,
        gdelt_multiplier_candidates=args.gdelt_multiplier_candidates,
        prior_strength=args.prior_strength,
        minimum_similarity=args.minimum_similarity,
    )

    selection_path = Path(args.selection_output)
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    final_report_path = Path(args.final_report_output)
    final_report_path.parent.mkdir(parents=True, exist_ok=True)
    final_report_path.write_text(
        json.dumps(
            selection.final_holdout.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
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

    subgroups = summarize_diagnostics(
        diagnostics,
        minimum_group_predictions=args.minimum_group_predictions,
    )
    subgroups_path = Path(args.subgroups_output)
    subgroups_path.parent.mkdir(parents=True, exist_ok=True)
    subgroups_path.write_text(
        json.dumps(subgroups, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    final = selection.final_holdout
    print(
        f"selected_top_k={selection.selected_top_k} "
        f"selected_scale={selection.selected_numeric_scale} "
        f"selected_family={selection.selected_signal_family} "
        f"selected_multiplier={selection.selected_gdelt_multiplier} "
        f"validation_skill={selection.selected_validation_candidate.gdelt_brier_skill_vs_structure:.6f} "
        f"holdout_predictions={final.predictions} "
        f"holdout_skill={final.gdelt_brier_skill_vs_structure:.6f} "
        f"selection={selection_path} final={final_report_path}"
    )


if __name__ == "__main__":
    main()
