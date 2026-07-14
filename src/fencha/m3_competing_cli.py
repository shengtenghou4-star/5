from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timezone
from pathlib import Path

from .caseio import read_jsonl
from .competing_risks import forecast_competing_risks
from .datasets.parlgov import PARLGOV_CSV_URL, read_cabinets, write_jsonl
from .datasets.parlgov_competing import (
    DEFAULT_MECHANISMS,
    M3_COMPETING_BUILDER_VERSION,
    build_leader_spells,
    build_m3_competing_dataset,
)
from .datasets.parlgov_survival import DEFAULT_HORIZONS
from .engine import AnalogForecaster
from .m3_cli import _date, _horizons, _prepare_source, _sha256


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fencha-m3-risks",
        description="Build and replay coherent cause-specific leader-exit forecasts",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    build = commands.add_parser(
        "build",
        help="build total-exit and observable transition-channel labels",
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
    build.add_argument("--election-window-days", type=int, default=120)
    build.add_argument(
        "--output",
        default="data/processed/parlgov_leader_competing_risks.jsonl",
    )
    build.add_argument(
        "--manifest",
        default="data/processed/parlgov_leader_competing_risks.manifest.json",
    )

    forecast = commands.add_parser(
        "forecast",
        help="replay one historical cutoff as a total and cause-specific curve",
    )
    forecast.add_argument(
        "--cases",
        default="data/processed/parlgov_leader_competing_risks.jsonl",
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


def _build(args: argparse.Namespace) -> str:
    if args.election_window_days < 0:
        raise SystemExit("election-window-days must be non-negative")
    source_path = _prepare_source(args)
    cabinets = read_cabinets(source_path)
    cases = build_m3_competing_dataset(
        cabinets,
        as_of=args.as_of,
        horizons=args.horizons,
        earliest_cutoff=args.earliest_cutoff,
        election_window_days=args.election_window_days,
    )
    count = write_jsonl(cases, args.output)
    spells = build_leader_spells(
        cabinets,
        as_of=args.as_of,
        election_window_days=args.election_window_days,
    )
    transition_counts = {
        mechanism: sum(
            spell.transition_mechanism == mechanism for spell in spells
        )
        for mechanism in DEFAULT_MECHANISMS
    }
    cases_by_domain: dict[str, int] = {}
    positives_by_domain: dict[str, int] = {}
    for case in cases:
        cases_by_domain[case.domain] = cases_by_domain.get(case.domain, 0) + 1
        positives_by_domain[case.domain] = positives_by_domain.get(case.domain, 0) + int(
            case.outcome
        )
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "builder_version": M3_COMPETING_BUILDER_VERSION,
        "source": str(source_path),
        "source_sha256": _sha256(source_path),
        "as_of": args.as_of.isoformat(),
        "earliest_cutoff": args.earliest_cutoff.isoformat(),
        "horizons": list(args.horizons),
        "mechanisms": list(DEFAULT_MECHANISMS),
        "election_window_days": args.election_window_days,
        "cabinet_records": len(cabinets),
        "leader_spells": len(spells),
        "recorded_transitions_by_mechanism": transition_counts,
        "forecast_cases": count,
        "cases_by_domain": dict(sorted(cases_by_domain.items())),
        "positives_by_domain": dict(sorted(positives_by_domain.items())),
        "output": str(Path(args.output)),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return (
        f"cases={count} spells={len(spells)} mechanisms={DEFAULT_MECHANISMS} "
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
    forecast = forecast_competing_risks(
        cases,
        target_features=target.features,
        target_cutoff=target.cutoff_at,
        mechanisms=DEFAULT_MECHANISMS,
        horizons=args.horizons,
        model=model,
    )
    payload = forecast.to_dict()
    payload["target_case_id"] = target.case_id
    payload["target_question"] = target.question
    payload["label_scope"] = (
        "observable transition channels; not asserted hidden political causes"
    )
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
