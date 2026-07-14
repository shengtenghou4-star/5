from datetime import datetime, timedelta, timezone
from math import log1p

import pytest

from fencha.m2_1_features import (
    M21_FEATURE_VERSION,
    add_time_safe_volume_features,
)
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc


def _case(
    case_id: str,
    cutoff: datetime,
    value: float,
    *,
    country: str = "GBR",
) -> HistoricalCase:
    return HistoricalCase(
        case_id=case_id,
        domain="government_leader_exit_180d",
        question="Will the leader leave?",
        cutoff_at=cutoff,
        resolved_at=cutoff + timedelta(days=180),
        outcome=False,
        features={
            "country_code": FeatureValue(country, cutoff, "categorical"),
            "gdelt_events_per_sample_30d": FeatureValue(value, cutoff, "numeric"),
            "gdelt_articles_per_sample_30d": FeatureValue(
                value * 2, cutoff, "numeric"
            ),
        },
        tags=(country,),
    )


def test_log_and_anomaly_features_use_only_earlier_country_history() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    cases = [
        _case(f"c-{index}", start + timedelta(days=30 * index), value)
        for index, value in enumerate((1.0, 3.0, 7.0, 15.0))
    ]

    enriched = add_time_safe_volume_features(cases, minimum_history=2)
    by_id = {case.case_id: case for case in enriched}

    assert by_id["c-0"].features["gdelt_log_events_per_sample_30d"].value == pytest.approx(
        log1p(1.0)
    )
    assert "gdelt_anomaly_log_events_per_sample_30d" not in by_id["c-0"].features
    assert "gdelt_anomaly_log_events_per_sample_30d" not in by_id["c-1"].features
    assert "gdelt_anomaly_log_events_per_sample_30d" in by_id["c-2"].features
    assert M21_FEATURE_VERSION in by_id["c-3"].tags
    assert all(
        feature.observed_at <= case.cutoff_at
        for case in enriched
        for feature in case.features.values()
    )


def test_future_values_cannot_change_earlier_anomalies() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    original = [
        _case(f"c-{index}", start + timedelta(days=30 * index), value)
        for index, value in enumerate((1.0, 2.0, 4.0, 8.0, 16.0))
    ]
    changed = [*original[:-1], _case("c-4", original[-1].cutoff_at, 1_000_000.0)]

    first = add_time_safe_volume_features(original, minimum_history=2)
    second = add_time_safe_volume_features(changed, minimum_history=2)
    first_by_id = {case.case_id: case for case in first}
    second_by_id = {case.case_id: case for case in second}

    for case_id in ("c-0", "c-1", "c-2", "c-3"):
        assert first_by_id[case_id].features == second_by_id[case_id].features


def test_same_cutoff_cases_do_not_enter_one_anothers_history() -> None:
    start = datetime(2020, 1, 1, tzinfo=UTC)
    history = [
        _case("past-1", start, 1.0),
        _case("past-2", start + timedelta(days=30), 2.0),
    ]
    cutoff = start + timedelta(days=60)
    ordinary = _case("ordinary", cutoff, 3.0)
    extreme = _case("extreme", cutoff, 100_000.0)

    ordinary_alone = add_time_safe_volume_features(
        [*history, ordinary], minimum_history=2
    )[-1]
    together = add_time_safe_volume_features(
        [*history, extreme, ordinary], minimum_history=2
    )
    together_ordinary = next(case for case in together if case.case_id == "ordinary")

    assert together_ordinary.features[
        "gdelt_anomaly_log_events_per_sample_30d"
    ].value == pytest.approx(
        ordinary_alone.features[
            "gdelt_anomaly_log_events_per_sample_30d"
        ].value
    )


def test_feature_configuration_validation() -> None:
    with pytest.raises(ValueError, match="minimum_history"):
        add_time_safe_volume_features([], minimum_history=0)
    with pytest.raises(ValueError, match="anomaly_clip"):
        add_time_safe_volume_features([], anomaly_clip=0)
