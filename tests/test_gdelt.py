import io
import zipfile
from datetime import datetime, timedelta, timezone

from fencha.datasets.gdelt import (
    CountrySlice,
    SliceStats,
    enrich_cases,
    parse_export_zip,
)
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc


def _row(
    country: str = "UK",
    root: str = "14",
    quad: str = "3",
    articles: str = "4",
    tone: str = "-6",
    goldstein: str = "-2",
) -> str:
    values = [""] * 61
    values[25] = "1"
    values[28] = root
    values[29] = quad
    values[30] = goldstein
    values[33] = articles
    values[34] = tone
    values[53] = country
    return "\t".join(values) + "\n"


def _zipped(text: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("sample.export.CSV", text)
    return buffer.getvalue()


def test_parser_aggregates_event_classes_and_maps_fips() -> None:
    observed = datetime(2022, 1, 2, 12, tzinfo=UTC)
    payload = _zipped(
        _row()
        + _row(root="19", quad="4", articles="2", tone="1", goldstein="-8")
        + _row(country="GM", root="03", quad="1", articles="3", tone="2", goldstein="5")
    )
    slices = parse_export_zip(payload, observed, "sample.zip", ["GBR", "DEU"])
    by_country = {item.country_code: item.stats for item in slices}

    assert by_country["GBR"].events == 2
    assert by_country["GBR"].protest_articles == 4
    assert by_country["GBR"].material_conflict_articles == 2
    assert by_country["DEU"].cooperation_articles == 3


def test_enrichment_uses_only_pre_cutoff_slices() -> None:
    cutoff = datetime(2022, 4, 1, tzinfo=UTC)
    case = HistoricalCase(
        case_id="q",
        domain="government_leader_exit_180d",
        question="Will the leader leave?",
        cutoff_at=cutoff,
        resolved_at=cutoff + timedelta(days=180),
        outcome=False,
        features={
            "country_code": FeatureValue("GBR", cutoff, "categorical")
        },
        tags=("GBR",),
    )
    slices = [
        CountrySlice(
            cutoff - timedelta(days=days),
            "GBR",
            "past.zip",
            SliceStats(
                events=2,
                articles=4,
                protest_articles=1,
                tone_sum=-8,
                tone_weight=4,
            ),
        )
        for days in range(7, 99, 7)
    ]
    slices.append(
        CountrySlice(
            cutoff + timedelta(days=1),
            "GBR",
            "future.zip",
            SliceStats(events=999, articles=999),
        )
    )

    enriched = enrich_cases([case], slices)

    assert len(enriched) == 1
    assert enriched[0].features["gdelt_events_per_sample_30d"].value == 2
    assert enriched[0].features["gdelt_avg_tone_90d"].value == -2
    assert all(
        feature.observed_at <= enriched[0].cutoff_at
        for feature in enriched[0].features.values()
    )
