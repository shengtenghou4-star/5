from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .benchmark import temporal_holdout_benchmark
from .caseio import read_jsonl
from .datasets.parlgov import (
    BUILDER_VERSION,
    PARLGOV_CSV_URL,
    SnapshotManifest,
    build_leader_exit_cases,
    extract_view_cabinet,
    read_cabinets,
    write_jsonl,
)
from .datasets.parlgov_download import download_snapshot_cached
from .demo import run_demo
from .engine import AnalogForecaster


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha",
        description="Auditable, time-aware historical forecasting engine",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run the v0.1 walk-forward demonstration")
    demo.add_argument("--database", default="fencha.db", help="SQLite ledger path")

    parlgov = subparsers.add_parser(
        "build-parlgov",
        help="build monthly 180-day government-leader-exit cases from ParlGov",
    )
    parlgov.add_argument(
        "--source",
        default=PARLGOV_CSV_URL,
        help="official ParlGov view_cabinet.csv URL or a local ZIP/CSV path",
    )
    parlgov.add_argument(
        "--work-dir",
        default="data/raw/parlgov",
        help="directory for immutable source snapshots",
    )
    parlgov.add_argument(
        "--download-timeout",
        type=int,
        default=120,
        help="remote ParlGov timeout in seconds",
    )
    parlgov.add_argument(
        "--download-retries",
        type=int,
        default=3,
        help="number of retries after the first remote ParlGov attempt",
    )
    parlgov.add_argument(
        "--refresh-source",
        action="store_true",
        help="ignore a validated cached ParlGov snapshot and download again",
    )
    parlgov.add_argument(
        "--output",
        default="data/processed/parlgov_leader_exit_180d.jsonl",
        help="output JSONL case table",
    )
    parlgov.add_argument(
        "--manifest",
        default="data/processed/parlgov_leader_exit_180d.manifest.json",
        help="output source and build manifest",
    )
    parlgov.add_argument(
        "--as-of",
        type=_date,
        default=date(2023, 6, 30),
        help="last date considered observable (default: 2023-06-30)",
    )
    parlgov.add_argument(
        "--earliest-cutoff",
        type=_date,
        default=date(1945, 1, 1),
        help="earliest monthly forecast cutoff",
    )

    benchmark = subparsers.add_parser(
        "benchmark",
        help="compare the analog model with a smoothed historical base rate",
    )
    benchmark.add_argument(
        "--cases",
        default="data/processed/parlgov_leader_exit_180d.jsonl",
        help="historical case JSONL file",
    )
    benchmark.add_argument(
        "--holdout-start",
        type=_date,
        default=date(2015, 1, 1),
        help="first date of the chronological holdout block",
    )
    benchmark.add_argument(
        "--minimum-training-cases",
        type=int,
        default=500,
        help="minimum resolved historical cases required before forecasting",
    )
    benchmark.add_argument(
        "--target-stride",
        type=int,
        default=3,
        help="evaluate every Nth monthly target per country",
    )
    benchmark.add_argument(
        "--max-history",
        type=int,
        default=5000,
        help="maximum most-recent eligible cases used for each prediction",
    )
    benchmark.add_argument(
        "--output",
        default="data/processed/parlgov_benchmark.json",
        help="benchmark report path",
    )
    return parser


def _local_manifest(path: Path) -> SnapshotManifest:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return SnapshotManifest(
        source_url=path.resolve().as_uri(),
        retrieved_at=datetime.now(timezone.utc).isoformat(),
        sha256=digest.hexdigest(),
        bytes=total,
        builder_version=BUILDER_VERSION,
    )


def _build_parlgov(args: argparse.Namespace) -> str:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(args.source)

    if source_path.exists():
        snapshot = _local_manifest(source_path)
        if source_path.suffix.lower() == ".zip":
            csv_path = extract_view_cabinet(source_path, work_dir / "view_cabinet.csv")
        else:
            csv_path = source_path
    else:
        remote_suffix = Path(urlparse(args.source).path).suffix.lower()
        if remote_suffix == ".zip":
            archive_path = work_dir / "parlgov-source.zip"
            snapshot = download_snapshot_cached(
                args.source,
                archive_path,
                timeout=args.download_timeout,
                retries=args.download_retries,
                reuse_existing=not args.refresh_source,
            )
            csv_path = extract_view_cabinet(
                archive_path, work_dir / "view_cabinet.csv"
            )
        else:
            csv_path = work_dir / "view_cabinet.csv"
            snapshot = download_snapshot_cached(
                args.source,
                csv_path,
                timeout=args.download_timeout,
                retries=args.download_retries,
                reuse_existing=not args.refresh_source,
            )

    cabinets = read_cabinets(csv_path)
    cases = build_leader_exit_cases(
        cabinets,
        as_of=args.as_of,
        earliest_cutoff=args.earliest_cutoff,
    )
    count = write_jsonl(cases, args.output)
    yes_count = sum(case.outcome for case in cases)
    countries = sorted({case.tags[0] for case in cases})
    feature_names = sorted({name for case in cases for name in case.features})

    manifest = {
        "snapshot": asdict(snapshot),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "builder_version": BUILDER_VERSION,
        "as_of": args.as_of.isoformat(),
        "earliest_cutoff": args.earliest_cutoff.isoformat(),
        "cabinet_records": len(cabinets),
        "forecast_cases": count,
        "positive_cases": yes_count,
        "base_rate": yes_count / count if count else None,
        "countries": countries,
        "features": feature_names,
        "output": str(Path(args.output)),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        f"cabinets={len(cabinets)} cases={count} positives={yes_count} "
        f"countries={len(countries)} output={args.output} manifest={args.manifest}"
    )


def _benchmark(args: argparse.Namespace) -> str:
    cases = read_jsonl(args.cases)
    model = AnalogForecaster(
        feature_weights={
            "country_code": 0.25,
            "tenure_days": 2.0,
            "cabinet_age_days": 0.75,
            "coalition_size": 0.5,
            "caretaker": 1.5,
            "cabinet_type": 0.75,
            "election_age_days": 1.25,
            "government_seat_share": 1.25,
            "minority_government": 1.0,
        },
        prior_strength=8.0,
        top_k=50,
    )
    report = temporal_holdout_benchmark(
        cases,
        model,
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
        f"predictions={report.analog.predictions} "
        f"baseline_brier={report.baseline.brier_score:.6f} "
        f"analog_brier={report.analog.brier_score:.6f} "
        f"brier_skill={report.analog_brier_skill:.6f} output={output}"
    )


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "demo":
        print(run_demo(args.database))
    elif args.command == "build-parlgov":
        print(_build_parlgov(args))
    elif args.command == "benchmark":
        print(_benchmark(args))


if __name__ == "__main__":
    main()
