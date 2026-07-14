from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .m3_1_select import select_hierarchical_survival_model


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _horizons(value: str) -> tuple[int, ...]:
    try:
        result = tuple(
            sorted({int(item.strip()) for item in value.split(",") if item.strip()})
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "horizons must be comma-separated integers"
        ) from exc
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("horizons must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m3-1-select",
        description=(
            "Select a hierarchical rare-event survival model before the final "
            "holdout and evaluate it once"
        ),
    )
    parser.add_argument(
        "--cases",
        default="data/processed/parlgov_leader_survival.jsonl",
    )
    parser.add_argument("--validation-start", type=_date, default=date(2010, 1, 1))
    parser.add_argument("--validation-end", type=_date, default=date(2015, 1, 1))
    parser.add_argument("--holdout-start", type=_date, default=date(2015, 1, 1))
    parser.add_argument("--horizons", type=_horizons, default=(30, 90, 180, 365))
    parser.add_argument("--minimum-training-cases", type=int, default=500)
    parser.add_argument("--validation-target-stride", type=int, default=6)
    parser.add_argument("--holdout-target-stride", type=int, default=3)
    parser.add_argument("--max-history", type=int, default=3000)
    parser.add_argument(
        "--output",
        default="data/processed/m3_1_hierarchical_selection.json",
    )
    return parser


def _midnight(value: date) -> datetime:
    return datetime.combine(value, datetime.min.time(), tzinfo=timezone.utc)


def main() -> None:
    args = build_parser().parse_args()
    report = select_hierarchical_survival_model(
        read_jsonl(args.cases),
        validation_start=_midnight(args.validation_start),
        validation_end=_midnight(args.validation_end),
        holdout_start=_midnight(args.holdout_start),
        horizons=args.horizons,
        minimum_training_cases=args.minimum_training_cases,
        validation_target_stride=args.validation_target_stride,
        holdout_target_stride=args.holdout_target_stride,
        max_history=args.max_history,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    final = report.final_holdout
    print(
        f"candidates={len(report.candidates)} "
        f"selected_template={report.selected_template} "
        f"validation_brier={report.selected_validation.adjusted_integrated_brier:.6f} "
        f"holdout_snapshots={final.snapshots} "
        f"holdout_brier={final.adjusted_integrated_brier:.6f} "
        f"holdout_skill_vs_baseline={final.adjusted_integrated_brier_skill_vs_baseline:.6f} "
        f"output={output}"
    )


if __name__ == "__main__":
    main()
