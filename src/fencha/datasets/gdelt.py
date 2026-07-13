from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Iterable

from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc
GDELT2_BASE_URL = "https://data.gdeltproject.org/gdeltv2"
BUILDER_VERSION = "gdelt2-weekly-v1"

# ParlGov uses ISO3; GDELT ActionGeo_CountryCode uses FIPS 10-4.
PARLGOV_TO_FIPS = {
    "AUS": "AS",
    "AUT": "AU",
    "BEL": "BE",
    "BGR": "BU",
    "CAN": "CA",
    "CHE": "SZ",
    "CYP": "CY",
    "CZE": "EZ",
    "DEU": "GM",
    "DNK": "DA",
    "ESP": "SP",
    "EST": "EN",
    "FIN": "FI",
    "FRA": "FR",
    "GBR": "UK",
    "GRC": "GR",
    "HRV": "HR",
    "HUN": "HU",
    "IRL": "EI",
    "ISL": "IC",
    "ISR": "IS",
    "ITA": "IT",
    "JPN": "JA",
    "LTU": "LH",
    "LUX": "LU",
    "LVA": "LG",
    "MLT": "MT",
    "NLD": "NL",
    "NOR": "NO",
    "NZL": "NZ",
    "POL": "PL",
    "PRT": "PO",
    "ROU": "RO",
    "SVK": "LO",
    "SVN": "SI",
    "SWE": "SW",
    "TUR": "TU",
}
FIPS_TO_PARLGOV = {value: key for key, value in PARLGOV_TO_FIPS.items()}

# GDELT 2.0 Event export column offsets.
IS_ROOT_EVENT = 25
EVENT_ROOT_CODE = 28
QUAD_CLASS = 29
GOLDSTEIN_SCALE = 30
NUM_ARTICLES = 33
AVG_TONE = 34
ACTION_GEO_COUNTRY_CODE = 53
MIN_COLUMNS = 61


@dataclass(slots=True)
class SliceStats:
    events: int = 0
    articles: float = 0.0
    protest_articles: float = 0.0
    verbal_conflict_articles: float = 0.0
    material_conflict_articles: float = 0.0
    cooperation_articles: float = 0.0
    negative_articles: float = 0.0
    tone_sum: float = 0.0
    tone_weight: float = 0.0
    goldstein_sum: float = 0.0
    goldstein_weight: float = 0.0

    def add(
        self,
        *,
        root_code: str,
        quad_class: int,
        articles: float,
        tone: float | None,
        goldstein: float | None,
    ) -> None:
        weight = max(1.0, articles)
        self.events += 1
        self.articles += weight
        if root_code == "14":
            self.protest_articles += weight
        if quad_class == 3:
            self.verbal_conflict_articles += weight
        if quad_class == 4:
            self.material_conflict_articles += weight
        if quad_class in {1, 2}:
            self.cooperation_articles += weight
        if tone is not None:
            self.tone_sum += tone * weight
            self.tone_weight += weight
            if tone < -5.0:
                self.negative_articles += weight
        if goldstein is not None:
            self.goldstein_sum += goldstein * weight
            self.goldstein_weight += weight


@dataclass(frozen=True, slots=True)
class CountrySlice:
    observed_at: datetime
    country_code: str
    source_file: str
    stats: SliceStats


@dataclass(frozen=True, slots=True)
class DownloadRecord:
    requested_at: str
    observed_at: str | None
    url: str | None
    sha256: str | None
    bytes: int
    status: str
    error: str | None = None


def _float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_export_zip(
    payload: bytes,
    observed_at: datetime,
    source_file: str,
    countries: Iterable[str],
) -> list[CountrySlice]:
    """Aggregate one GDELT 2.0 event ZIP into country-level signal slices."""
    wanted = set(countries)
    aggregates = {code: SliceStats() for code in wanted}
    with zipfile.ZipFile(io.BytesIO(payload)) as bundle:
        names = [name for name in bundle.namelist() if not name.endswith("/")]
        if not names:
            raise ValueError("empty GDELT archive")
        with bundle.open(names[0]) as raw:
            text = io.TextIOWrapper(
                raw, encoding="utf-8", errors="replace", newline=""
            )
            reader = csv.reader(text, delimiter="\t")
            for row in reader:
                if len(row) < MIN_COLUMNS or row[IS_ROOT_EVENT] != "1":
                    continue
                iso3 = FIPS_TO_PARLGOV.get(
                    row[ACTION_GEO_COUNTRY_CODE].strip().upper()
                )
                if iso3 not in aggregates:
                    continue
                quad = _int(row[QUAD_CLASS])
                if quad is None:
                    continue
                aggregates[iso3].add(
                    root_code=row[EVENT_ROOT_CODE].strip(),
                    quad_class=quad,
                    articles=_float(row[NUM_ARTICLES]) or 1.0,
                    tone=_float(row[AVG_TONE]),
                    goldstein=_float(row[GOLDSTEIN_SCALE]),
                )
    return [
        CountrySlice(observed_at, code, source_file, stats)
        for code, stats in sorted(aggregates.items())
    ]


def iter_sample_times(
    start: date,
    end: date,
    every_days: int = 7,
    hour: int = 12,
) -> list[datetime]:
    if every_days <= 0:
        raise ValueError("every_days must be positive")
    if not 0 <= hour <= 23:
        raise ValueError("hour must be 0..23")
    cursor = datetime.combine(start, dtime(hour=hour), tzinfo=UTC)
    final = datetime.combine(end, dtime(hour=hour), tzinfo=UTC)
    result: list[datetime] = []
    while cursor <= final:
        result.append(cursor)
        cursor += timedelta(days=every_days)
    return result


def _candidate_urls(
    requested: datetime,
    base_url: str,
) -> list[tuple[datetime, str]]:
    return [
        (
            requested + timedelta(minutes=minutes),
            f"{base_url.rstrip('/')}/"
            f"{(requested + timedelta(minutes=minutes)).strftime('%Y%m%d%H%M%S')}"
            ".export.CSV.zip",
        )
        for minutes in (0, 15, 30, 45)
    ]


def download_one(
    requested: datetime,
    countries: Iterable[str],
    *,
    base_url: str = GDELT2_BASE_URL,
    timeout: int = 90,
    retries: int = 2,
) -> tuple[list[CountrySlice], DownloadRecord]:
    """Download one deterministic slice, trying the next three quarter-hours."""
    last_error = "not found"
    for observed_at, url in _candidate_urls(requested, base_url):
        for attempt in range(retries + 1):
            try:
                request = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": "FENCHA/0.2 historical-forecasting-research"
                    },
                )
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    payload = response.read()
                digest = hashlib.sha256(payload).hexdigest()
                slices = parse_export_zip(
                    payload,
                    observed_at,
                    url.rsplit("/", 1)[-1],
                    countries,
                )
                return slices, DownloadRecord(
                    requested.isoformat(),
                    observed_at.isoformat(),
                    url,
                    digest,
                    len(payload),
                    "ok",
                )
            except urllib.error.HTTPError as exc:
                last_error = f"HTTP {exc.code}"
                if exc.code == 404:
                    break
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt < retries:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break
    return [], DownloadRecord(
        requested.isoformat(), None, None, None, 0, "missing", last_error
    )


def collect_weekly_samples(
    *,
    start: date,
    end: date,
    countries: Iterable[str],
    every_days: int = 7,
    hour: int = 12,
    workers: int = 6,
    base_url: str = GDELT2_BASE_URL,
) -> tuple[list[CountrySlice], list[DownloadRecord]]:
    requested = iter_sample_times(start, end, every_days, hour)
    slices: list[CountrySlice] = []
    records: list[DownloadRecord] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_map = {
            pool.submit(
                download_one,
                sample_time,
                tuple(countries),
                base_url=base_url,
            ): sample_time
            for sample_time in requested
        }
        for future in as_completed(future_map):
            country_slices, record = future.result()
            slices.extend(country_slices)
            records.append(record)
    slices.sort(key=lambda item: (item.observed_at, item.country_code))
    records.sort(key=lambda item: item.requested_at)
    return slices, records


def _window_features(
    items: list[CountrySlice],
    cutoff: datetime,
    window_days: int,
    expected_samples: float,
) -> dict[str, float] | None:
    start = cutoff - timedelta(days=window_days)
    selected = [item for item in items if start <= item.observed_at < cutoff]
    minimum = max(2, math.ceil(expected_samples * 0.55))
    if len(selected) < minimum:
        return None

    total = SliceStats()
    for item in selected:
        stats = item.stats
        total.events += stats.events
        total.articles += stats.articles
        total.protest_articles += stats.protest_articles
        total.verbal_conflict_articles += stats.verbal_conflict_articles
        total.material_conflict_articles += stats.material_conflict_articles
        total.cooperation_articles += stats.cooperation_articles
        total.negative_articles += stats.negative_articles
        total.tone_sum += stats.tone_sum
        total.tone_weight += stats.tone_weight
        total.goldstein_sum += stats.goldstein_sum
        total.goldstein_weight += stats.goldstein_weight

    denominator = max(total.articles, 1.0)
    suffix = f"{window_days}d"
    return {
        f"gdelt_samples_{suffix}": float(len(selected)),
        f"gdelt_coverage_{suffix}": min(1.0, len(selected) / expected_samples),
        f"gdelt_events_per_sample_{suffix}": total.events / len(selected),
        f"gdelt_articles_per_sample_{suffix}": total.articles / len(selected),
        f"gdelt_protest_share_{suffix}": total.protest_articles / denominator,
        f"gdelt_verbal_conflict_share_{suffix}": (
            total.verbal_conflict_articles / denominator
        ),
        f"gdelt_material_conflict_share_{suffix}": (
            total.material_conflict_articles / denominator
        ),
        f"gdelt_cooperation_share_{suffix}": total.cooperation_articles / denominator,
        f"gdelt_negative_tone_share_{suffix}": (
            total.negative_articles / denominator
        ),
        f"gdelt_avg_tone_{suffix}": (
            total.tone_sum / total.tone_weight if total.tone_weight else 0.0
        ),
        f"gdelt_avg_goldstein_{suffix}": (
            total.goldstein_sum / total.goldstein_weight
            if total.goldstein_weight
            else 0.0
        ),
    }


def enrich_cases(
    cases: Iterable[HistoricalCase],
    slices: Iterable[CountrySlice],
    *,
    every_days: int = 7,
    windows: tuple[int, ...] = (30, 90),
) -> list[HistoricalCase]:
    """Attach only past GDELT windows; incomplete windows are excluded."""
    by_country: dict[str, list[CountrySlice]] = {}
    for item in slices:
        by_country.setdefault(item.country_code, []).append(item)
    for items in by_country.values():
        items.sort(key=lambda item: item.observed_at)

    enriched: list[HistoricalCase] = []
    for case in cases:
        country = case.tags[0] if case.tags else ""
        items = by_country.get(country, [])
        derived: dict[str, float] = {}
        complete = True
        for window in windows:
            values = _window_features(
                items,
                case.cutoff_at,
                window,
                window / every_days,
            )
            if values is None:
                complete = False
                break
            derived.update(values)
        if not complete:
            continue

        features = dict(case.features)
        for name, value in derived.items():
            features[name] = FeatureValue(
                float(value), case.cutoff_at, "numeric"
            )
        enriched.append(
            HistoricalCase(
                case_id=case.case_id,
                domain=case.domain,
                question=case.question,
                cutoff_at=case.cutoff_at,
                resolved_at=case.resolved_at,
                outcome=case.outcome,
                features=features,
                tags=case.tags + (BUILDER_VERSION,),
            )
        )
    return enriched


def write_samples(slices: Iterable[CountrySlice], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for item in slices:
            payload = {
                "observed_at": item.observed_at.isoformat(),
                "country_code": item.country_code,
                "source_file": item.source_file,
                "stats": asdict(item.stats),
            }
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            count += 1
    return count
