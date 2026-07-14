from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from .caseio import read_jsonl
from .datasets.parlgov import PARLGOV_CSV_URL, extract_view_cabinet, read_cabinets, write_jsonl
from .datasets.parlgov_download import download_snapshot_cached
from .datasets.parlgov_survival import (
    DEFAULT_HORIZONS,
    M3_BUILDER_VERSION,
    build_leader_survival_cases,
    normalize_horizons,
)
from .engine import AnalogForecaster
from .survival import forecast_survival_curve


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must use YYYY-MM-DD") from exc


def _horizons(value: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("horizons must be comma-separated integers") from exc
    try:
        return normalize_horizons(values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m3",
        description="Build and replay coherent government-leader survival forecasts",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build", help="build time-safe 30/90/180/365-day ParlGov cases"
    )
    build.add_argument("--source", default=PARLGOV_CSV_URL)
    build.add_argument("--work-dir", default="data/raw/parlgov")
    build.add_argument("--download-timeout", type=int, default=120)
    build.add_argument("--download-retries", type=int, default=3)
    build.add_argument("--refresh-source", action="store_true")
    build.add_argument("--as-of", type=_date, default=date(2023, 6, 30))
    build.add_argument("--earliest-cutoff", type=_date, default=date(1945, 1, 1))
    build.add_argument(
        "--horizons",
        type=_horizons,
        default=DEFAULT_HORIZONS,
        help="comma-separated positive day horizons",
    )
    build.add_argument(
        "--output",
        default="data/processed/parlgov_leader_survival.jsonl",
    )
    build.add_argument(
        "--manifest",
        default="data/processed/parlgov_leader_survival.manifest.json",
    )

    forecast = commands.add_parser(
        "forecast", help="replay one historical cutoff as a survival curve"
    )
    forecast.add_argument(
        "--cases",
        default="data/processed/parlgov_leader_survival.jsonl",
    )
    forecast.add_argument("--target-case-id", required=True)
    forecast.add_argument(
        "--horizons",
        type=_horizons,
        default=DEFAULT_HORIZONS,
    )
    forecast.add_argument("--top-k", type=int, default=50)
    forecast.add_argument("--prior-strength", type=float, default=8.0)
    forecast.add_argument(
        "--numeric-scale",
        choices=("range", "iqr"),
        default="iqr",
    )
    forecast.add_argument("--output")
    return parser


def _prepare_source(args: argparse.Namespace) -> Path:
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    local = Path(args.source)
    if local.exists():
        if local.suffix.lower() == ".zip":
            return extract_view_cabinet(local, work_dir / "view_cabinet.csv")
        return local

    suffix = Path(urlparse(args.source).path).suffix.lower()
    if suffix == ".zip":
        archive = work_dir / "parlgov-source.zip"
        download_snapshot_cached(
            args.source,
            archive,
            timeout=args.download_timeout,
            retries=args.download_retries,
            reuse_existing=not args.refresh_source,
        )
        return extract_view_cabinet(archive, work_dir / "view_cabinet.csv")

    csv_path = work_dir / "view_cabinet.csv"
    download_snapshot_cached(
        args.source,
        csv_path,
        timeout=args.download_timeout,
        retries=args.download_retries,
        reuse_existing=not args.refresh_source,
    )
    return csv_path


def _build(args: argparse.Namespace) -> str:
    source_path = _prepare_source(args)
    cabinets = read_cabinets(source_path)
    cases = build_leader_survival_cases(
        cabinets,
        as_of=args.as_of,
        horizons=args.horizons,
        earliest_cutoff=args.earliest_cutoff,
    )
    count = write_jsonl(cases, args.output)
    countries = sorted({case.tags[0] for case in cases})
    counts_by_horizon = {
        str(horizon): sum(
            case.domain == f"government_leader_exit_{horizon}d" for case in cases
        )
        for horizon in args.horizons
    }
    positives_by_horizon = {
        str(horizon): sum(
            case.outcome
            for case in cases
            if case.domain == f"government_leader_exit_{horizon}d"
        )
        for horizon in args.horizons
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "builder_version": M3_BUILDER_VERSION,
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "as_of": args.as_of.isoformat(),
        "earliest_cutoff": args.earliest_cutoff.isoformat(),
        "horizons": list(args.horizons),
        "cabinet_records": len(cabinets),
        "forecast_cases": count,
        "countries": countries,
        "cases_by_horizon": counts_by_horizon,
        "positives_by_horizon": positives_by_horizon,
        "output": str(Path(args.output)),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        f"cases={count} countries={len(countries)} horizons={args.horizons} "
        f"output={args.output} manifest={args.manifest}"
    )


def _forecast(args: argparse.Namespace) -> str:
    cases = read_jsonl(args.cases)
    target = next(
        (case for case in cases if case.case_id == args.target_case_id),
        None,
    )
    if target is None:
        raise SystemExit(f"target case not found: {args.target_case_id}")

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
        prior_strength=args.prior_strength,
        top_k=args.top_k,
        numeric_scale=args.numeric_scale,
    )
    curve = forecast_survival_curve(
        cases,
        target_features=target.features,
        target_cutoff=target.cutoff_at,
        horizons=args.horizons,
        model=model,
    )
    payload = curve.to_dict()
    payload["target_case_id"] = target.case_id
    payload["target_question"] = target.question
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        return f"target={target.case_id} output={output}"
    return text.rstrip()


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        print(_build(args))
    elif args.command == "forecast":
        print(_forecast(args))


if __name__ == "__main__":
    main()
