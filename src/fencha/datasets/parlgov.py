from __future__ import annotations

import csv
import hashlib
import json
import re
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence, TextIO

from fencha.models import FeatureValue, HistoricalCase

PARLGOV_CSV_URL = "https://www.parlgov.org/data/parlgov-development_csv-utf-8.zip"
BUILDER_VERSION = "parlgov-leader-exit-v1"
UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class CabinetRecord:
    cabinet_id: str
    country_code: str
    country_name: str
    cabinet_name: str
    leader_name: str
    start_date: date
    end_date: date | None = None
    election_date: date | None = None
    government_seats: float | None = None
    parliament_seats: float | None = None
    coalition_size: int = 1
    caretaker: bool = False
    cabinet_type: str = "unknown"


@dataclass(frozen=True, slots=True)
class SnapshotManifest:
    source_url: str
    retrieved_at: str
    sha256: str
    bytes: int
    builder_version: str


ALIASES: dict[str, tuple[str, ...]] = {
    "cabinet_id": ("cabinet_id", "id"),
    "country_code": ("country_name_short", "country_code", "iso3", "country"),
    "country_name": ("country_name", "country_name_english", "country"),
    "cabinet_name": ("cabinet_name", "name"),
    "leader_name": ("prime_minister", "pm_name", "head_of_government"),
    "start_date": ("start_date", "cabinet_start", "appointment_date", "date"),
    "end_date": ("end_date", "cabinet_end", "termination_date", "resignation_date"),
    "election_date": ("election_date", "previous_election_date"),
    "government_seats": ("cabinet_seats", "government_seats", "seats_government"),
    "parliament_seats": ("seats_total", "parliament_seats", "total_seats"),
    "party_seats": ("seats", "party_seats"),
    "party_id": ("party_id", "cabinet_party_id"),
    "cabinet_member": ("cabinet_party", "cabinet_member", "government_party"),
    "caretaker": ("caretaker", "caretaker_status"),
    "cabinet_type": ("cabinet_type", "type"),
}

_ROMAN_SUFFIX = re.compile(r"\s+(?:[IVXLCDM]+|\d+)$", re.IGNORECASE)
_STATUS_SUFFIX = re.compile(
    r"\s+(?:caretaker|interim|provisional|continuation|technical)$", re.IGNORECASE
)


def _canonical(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


def _value(row: Mapping[str, str], field: str) -> str:
    for alias in ALIASES[field]:
        value = row.get(alias, "").strip()
        if value:
            return value
    return ""


def _parse_date(value: str) -> date | None:
    value = value.strip()
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"unsupported date format: {value!r}")


def _parse_float(value: str) -> float | None:
    value = value.strip().replace(",", ".")
    if not value:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"invalid numeric value: {value!r}") from exc


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "t", "yes", "y", "caretaker"}


def infer_leader_name(cabinet_name: str) -> str:
    """Infer the leader key used by ParlGov's surname-based cabinet names."""
    value = cabinet_name.strip()
    while True:
        changed = _ROMAN_SUFFIX.sub("", value)
        changed = _STATUS_SUFFIX.sub("", changed)
        if changed == value:
            break
        value = changed.strip()
    return value or cabinet_name.strip()


def read_cabinets(source: str | Path | TextIO) -> list[CabinetRecord]:
    if hasattr(source, "read"):
        handle = source
        close = False
    else:
        handle = Path(source).open("r", encoding="utf-8-sig", newline="")
        close = True
    try:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError("ParlGov CSV has no header")
        reader.fieldnames = [_canonical(name) for name in reader.fieldnames]
        rows = [dict(row) for row in reader]
    finally:
        if close:
            handle.close()
    return _aggregate_rows(rows)


def _aggregate_rows(rows: Sequence[Mapping[str, str]]) -> list[CabinetRecord]:
    grouped: dict[str, dict[str, object]] = {}
    for raw in rows:
        row = {_canonical(k): (v or "") for k, v in raw.items()}
        start = _parse_date(_value(row, "start_date"))
        cabinet_name = _value(row, "cabinet_name")
        country_code = _value(row, "country_code").upper()
        if not start or not cabinet_name or not country_code:
            continue
        cabinet_id = _value(row, "cabinet_id") or f"{country_code}:{start}:{cabinet_name}"
        state = grouped.setdefault(
            cabinet_id,
            {
                "cabinet_id": cabinet_id,
                "country_code": country_code,
                "country_name": _value(row, "country_name") or country_code,
                "cabinet_name": cabinet_name,
                "leader_name": _value(row, "leader_name") or infer_leader_name(cabinet_name),
                "start_date": start,
                "end_date": _parse_date(_value(row, "end_date")),
                "election_date": _parse_date(_value(row, "election_date")),
                "government_seats": _parse_float(_value(row, "government_seats")),
                "parliament_seats": _parse_float(_value(row, "parliament_seats")),
                "party_seats": 0.0,
                "party_ids": set(),
                "caretaker": _parse_bool(_value(row, "caretaker")),
                "cabinet_type": _value(row, "cabinet_type") or "unknown",
            },
        )
        party_id = _value(row, "party_id")
        member_value = _value(row, "cabinet_member")
        is_member = _parse_bool(member_value) if member_value else True
        party_seats = _parse_float(_value(row, "party_seats"))
        if party_id and is_member:
            party_ids = state["party_ids"]
            assert isinstance(party_ids, set)
            party_ids.add(party_id)
        if party_seats is not None and is_member and state["government_seats"] is None:
            state["party_seats"] = float(state["party_seats"]) + party_seats

    result: list[CabinetRecord] = []
    for state in grouped.values():
        party_ids = state.pop("party_ids")
        party_seats = float(state.pop("party_seats"))
        if state["government_seats"] is None and party_seats > 0:
            state["government_seats"] = party_seats
        state["coalition_size"] = max(1, len(party_ids))
        result.append(CabinetRecord(**state))
    result.sort(key=lambda item: (item.country_code, item.start_date, item.cabinet_id))
    return result


def _month_starts(start: date, end_exclusive: date) -> Iterator[date]:
    cursor = date(start.year, start.month, 1)
    if cursor < start:
        cursor = date(
            cursor.year + (cursor.month == 12),
            1 if cursor.month == 12 else cursor.month + 1,
            1,
        )
    while cursor < end_exclusive:
        yield cursor
        cursor = date(
            cursor.year + (cursor.month == 12),
            1 if cursor.month == 12 else cursor.month + 1,
            1,
        )


def _at_utc(value: date) -> datetime:
    return datetime.combine(value, time.min, tzinfo=UTC)


def build_leader_exit_cases(
    cabinets: Iterable[CabinetRecord],
    *,
    as_of: date,
    horizon_days: int = 180,
    earliest_cutoff: date | None = None,
) -> list[HistoricalCase]:
    """Create monthly, fully resolved leader-exit cases without future leakage."""
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    by_country: dict[str, list[CabinetRecord]] = {}
    for cabinet in cabinets:
        if cabinet.start_date <= as_of:
            by_country.setdefault(cabinet.country_code, []).append(cabinet)

    cases: list[HistoricalCase] = []
    for country_code, country_cabinets in sorted(by_country.items()):
        country_cabinets.sort(key=lambda item: (item.start_date, item.cabinet_id))
        spells: list[tuple[str, date, date | None, list[CabinetRecord]]] = []
        current: list[CabinetRecord] = []
        current_leader = ""
        for cabinet in country_cabinets:
            leader_key = cabinet.leader_name.casefold().strip()
            if current and leader_key != current_leader:
                spells.append(
                    (current[0].leader_name, current[0].start_date, cabinet.start_date, current)
                )
                current = []
            if not current:
                current_leader = leader_key
            current.append(cabinet)
        if current:
            spells.append((current[0].leader_name, current[0].start_date, None, current))

        for leader_name, spell_start, exit_date, spell_cabinets in spells:
            active_end = exit_date or (as_of + timedelta(days=1))
            start = max(spell_start, earliest_cutoff) if earliest_cutoff else spell_start
            for cutoff in _month_starts(start, active_end):
                horizon_end = cutoff + timedelta(days=horizon_days)
                outcome = exit_date is not None and exit_date <= horizon_end
                if not outcome and horizon_end > as_of:
                    continue
                current_cabinet = max(
                    (item for item in spell_cabinets if item.start_date <= cutoff),
                    key=lambda item: item.start_date,
                    default=None,
                )
                if current_cabinet is None:
                    continue
                resolved = exit_date if outcome else horizon_end
                assert resolved is not None
                observed_at = _at_utc(cutoff)
                features: dict[str, FeatureValue] = {
                    "country_code": FeatureValue(country_code, observed_at, "categorical"),
                    "tenure_days": FeatureValue(
                        float((cutoff - spell_start).days), observed_at, "numeric"
                    ),
                    "cabinet_age_days": FeatureValue(
                        float((cutoff - current_cabinet.start_date).days), observed_at, "numeric"
                    ),
                    "coalition_size": FeatureValue(
                        float(current_cabinet.coalition_size), observed_at, "numeric"
                    ),
                    "caretaker": FeatureValue(
                        current_cabinet.caretaker, observed_at, "boolean"
                    ),
                    "cabinet_type": FeatureValue(
                        current_cabinet.cabinet_type, observed_at, "categorical"
                    ),
                }
                if current_cabinet.election_date and current_cabinet.election_date <= cutoff:
                    features["election_age_days"] = FeatureValue(
                        float((cutoff - current_cabinet.election_date).days),
                        observed_at,
                        "numeric",
                    )
                if (
                    current_cabinet.government_seats is not None
                    and current_cabinet.parliament_seats
                    and current_cabinet.parliament_seats > 0
                ):
                    share = (
                        current_cabinet.government_seats
                        / current_cabinet.parliament_seats
                    )
                    features["government_seat_share"] = FeatureValue(
                        share, observed_at, "numeric"
                    )
                    features["minority_government"] = FeatureValue(
                        share < 0.5, observed_at, "boolean"
                    )
                cases.append(
                    HistoricalCase(
                        case_id=(
                            f"parlgov:{country_code}:{leader_name}:{cutoff.isoformat()}"
                        ),
                        domain="government_leader_exit_180d",
                        question=(
                            f"Will {leader_name}, head of government of "
                            f"{current_cabinet.country_name}, leave office within 180 days?"
                        ),
                        cutoff_at=observed_at,
                        resolved_at=_at_utc(resolved),
                        outcome=outcome,
                        features=features,
                        tags=(country_code, "parlgov", BUILDER_VERSION),
                    )
                )
    cases.sort(key=lambda item: (item.cutoff_at, item.case_id))
    return cases


def case_to_dict(case: HistoricalCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "domain": case.domain,
        "question": case.question,
        "cutoff_at": case.cutoff_at.isoformat(),
        "resolved_at": case.resolved_at.isoformat(),
        "outcome": case.outcome,
        "tags": list(case.tags),
        "features": {
            name: {
                "value": feature.value,
                "kind": feature.kind,
                "observed_at": feature.observed_at.isoformat(),
            }
            for name, feature in sorted(case.features.items())
        },
    }


def write_jsonl(cases: Iterable[HistoricalCase], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(
                json.dumps(case_to_dict(case), ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
    return count


def download_snapshot(url: str, destination: str | Path) -> SnapshotManifest:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FENCHA/0.1 historical-forecasting-research"},
    )
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response, destination.open(
        "wb"
    ) as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
            digest.update(chunk)
            total += len(chunk)
    return SnapshotManifest(
        source_url=url,
        retrieved_at=datetime.now(UTC).isoformat(),
        sha256=digest.hexdigest(),
        bytes=total,
        builder_version=BUILDER_VERSION,
    )


def extract_view_cabinet(archive: str | Path, destination: str | Path) -> Path:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        candidates = [
            name
            for name in bundle.namelist()
            if name.lower().endswith("view_cabinet.csv")
        ]
        if not candidates:
            raise FileNotFoundError("view_cabinet.csv not found in ParlGov archive")
        with bundle.open(candidates[0]) as source, destination.open("wb") as output:
            output.write(source.read())
    return destination
