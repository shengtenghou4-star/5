from datetime import datetime, timedelta, timezone

import pytest

from fencha.m2_1 import compare_matched_architecture
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc


def _month(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


def _cases() -> list[HistoricalCase]:
    cases: list[HistoricalCase] = []
    for index in range(30):
        year = 2020 + index // 12
        month = index % 12 + 1
        cutoff = _month(year, month)
        outcome = index % 5 == 0
        features = {
            "country_code": FeatureValue("GBR", cutoff, "categorical"),
            "tenure_days": FeatureValue(float(index * 30), cutoff, "numeric"),
            "cabinet_age_days": FeatureValue(float(index * 20), cutoff, "numeric"),
            # These are diagnostics only and must never receive the engine's
            # implicit default weight.
            "gdelt_samples_30d": FeatureValue(float(2 + index % 4), cutoff, "numeric"),
            "gdelt_samples_90d": FeatureValue(float(7 + index % 6), cutoff, "numeric"),
            "gdelt_protest_share_30d": FeatureValue(
                0.4 if outcome else 0.05, cutoff, "numeric"
            ),
            "gdelt_protest_share_90d": FeatureValue(
                0.3 if outcome else 0.04, cutoff, "numeric"
            ),
            "gdelt_coverage_30d": FeatureValue(1.0, cutoff, "numeric"),
            "gdelt_coverage_90d": FeatureValue(1.0, cutoff, "numeric"),
        }
        cases.append(
            HistoricalCase(
                case_id=f"case-{index}",
                domain="government_leader_exit_180d",
                question="Will the leader leave?",
                cutoff_at=cutoff,
                resolved_at=cutoff + timedelta(days=25),
                outcome=outcome,
                features=features,
                tags=("GBR", "test"),
            )
        )
    return cases


def test_zero_gdelt_multiplier_reproduces_structure_model() -> None:
    report, diagnostics = compare_matched_architecture(
        _cases(),
        holdout_start=_month(2021, 1),
        minimum_training_cases=5,
        max_history=100,
        top_k=7,
        gdelt_multiplier=0.0,
    )

    assert report.predictions == len(report.target_ids) == len(diagnostics)
    assert report.structure_analog.predictions == report.gdelt_analog.predictions
    assert report.structure_analog.brier_score == pytest.approx(
        report.gdelt_analog.brier_score
    )
    assert all(
        item.structure_probability == pytest.approx(item.gdelt_probability)
        for item in diagnostics
    )
    assert all(len(item.structure_neighbor_ids) <= 7 for item in diagnostics)
    assert all(len(item.gdelt_neighbor_ids) <= 7 for item in diagnostics)


def test_matched_comparison_records_exact_target_ids() -> None:
    report, diagnostics = compare_matched_architecture(
        _cases(),
        holdout_start=_month(2021, 1),
        minimum_training_cases=5,
        max_history=100,
        top_k=5,
        gdelt_multiplier=1.0,
    )

    assert report.target_ids == tuple(item.case_id for item in diagnostics)
    assert report.first_cutoff == diagnostics[0].cutoff_at
    assert report.last_cutoff == diagnostics[-1].cutoff_at
    assert 0.0 <= report.mean_neighbor_overlap <= 1.0
