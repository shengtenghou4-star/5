from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .datasets.gdelt import (
    BUILDER_VERSION,
    GDELT2_BASE_URL,
    collect_weekly_samples,
    enrich_cases,
    write_samples,
)
from .datasets.parlgov import write_jsonl
from .m2 import compare_structure_and_gdelt


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m2",
        description="Build and benchmark sampled GDELT historical signals",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build",
        help="download deterministic GDELT slices and enrich ParlGov cases",
    )
    build.add_argument(
        "--cases",
        default="data/processed/parlgov_leader_exit_180d.jsonl",
    )
    build.add_argument("--start", type=_date, default=date(2019, 10, 6))
    build.add_argument("--end", type=_date, default=date(2023, 1, 1))
    build.add_argument("--every-days", type=int, default=7)
    build.add_argument("--hour", type=int, default=12)
    build.add_argument("--workers", type=int, default=6)
    build.add_argument("--base-url", default=GDELT2_BASE_URL)
    build.add_argument("--max-missing-rate", type=float, default=0.15)
    build.add_argument(
        "--output",
        default="data/processed/parlgov_gdelt_weekly.jsonl",
    )
    build.add_argument(
        "--samples-output",
        default="data/processed/gdelt_weekly_country_slices.jsonl",
    )
    build.add_argument(
        "--manifest",
        default="data/processed/gdelt_weekly_manifest.json",
    )

    compare = subparsers.add_parser(
        "compare",
        help="compare structure-only and GDELT-enriched models on one holdout",
    )
    compare.add_argument(
        "--cases",
        default="data/processed/parlgov_gdelt_weekly.jsonl",
    )
    compare.add_argument(
        "--holdout-start", type=_date, default=date(2022, 1, 1)
    )
    compare.add_argument("--minimum-training-cases", type=int, default=500)
    compare.add_argument("--target-stride", type=int, default=1)
    compare.add_argument("--max-history", type=int, default=3000)
    compare.add_argument(
        "--output",
        default="data/processed/m2_gdelt_comparison.json",
    )
    return parser


def _build(args: argparse.Namespace) -> str:
    cases = read_jsonl(args.cases)
    countries = sorted({case.tags[0] for case in cases if case.tags})
    slices, downloads = collect_weekly_samples(
        start=args.start,
        end=args.end,
        countries=countries,
        every_days=args.every_days,
        hour=args.hour,
        workers=args.workers,
        base_url=args.base_url,
    )
    downloaded = [record for record in downloads if record.status == "ok"]
    missing = [record for record in downloads if record.status != "ok"]
    missing_rate = len(missing) / len(downloads) if downloads else 1.0
    if missing_rate > args.max_missing_rate:
        raise RuntimeError(
            f"GDELT missing rate {missing_rate:.1%} exceeds "
            f"limit {args.max_missing_rate:.1%}"
        )

    sample_rows = write_samples(slices, args.samples_output)
    enriched = enrich_cases(cases, slices, every_days=args.every_days)
    case_rows = write_jsonl(enriched, args.output)
    positives = sum(case.outcome for case in enriched)
    enriched_countries = sorted({case.tags[0] for case in enriched if case.tags})
    feature_names = sorted(
        {
            name
            for case in enriched
            for name in case.features
            if name.startswith("gdelt_")
        }
    )

    manifest = {
        "builder_version": BUILDER_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "start": args.start.isoformat(),
        "end": args.end.isoformat(),
        "every_days": args.every_days,
        "hour_utc": args.hour,
        "requested_files": len(downloads),
        "downloaded_files": len(downloaded),
        "missing_files": len(missing),
        "missing_rate": missing_rate,
        "downloaded_bytes": sum(record.bytes for record in downloaded),
        "country_slice_rows": sample_rows,
        "enriched_cases": case_rows,
        "positive_cases": positives,
        "base_rate": positives / case_rows if case_rows else None,
        "countries": enriched_countries,
        "features": feature_names,
        "downloads": [asdict(record) for record in downloads],
        "output": str(Path(args.output)),
        "samples_output": str(Path(args.samples_output)),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        f"requested={len(downloads)} downloaded={len(downloaded)} "
        f"missing={len(missing)} slices={sample_rows} cases={case_rows} "
        f"countries={len(enriched_countries)} output={args.output}"
    )


def _compare(args: argparse.Namespace) -> str:
    cases = read_jsonl(args.cases)
    report = compare_structure_and_gdelt(
        cases,
        holdout_start=datetime.combine(
            args.holdout_start,
            datetime.min.time(),
            tzinfo=timezone.utc,
        ),
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
    return (
        f"predictions={report.predictions} "
        f"structure_brier={report.structure_analog.brier_score:.6f} "
        f"gdelt_brier={report.gdelt_analog.brier_score:.6f} "
        f"gdelt_skill_vs_structure={report.gdelt_brier_skill_vs_structure:.6f} "
        f"output={output}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        print(_build(args))
    elif args.command == "compare":
        print(_compare(args))


if __name__ == "__main__":
    main()
