from datetime import datetime, timedelta, timezone

from fencha.m2 import compare_structure_and_gdelt
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc


def _month(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


def _cases() -> list[HistoricalCase]:
    cases: list[HistoricalCase] = []
    for index in range(24):
        year = 2020 + index // 12
        month = index % 12 + 1
        cutoff = _month(year, month)
        outcome = index % 5 == 0
        features = {
            "country_code": FeatureValue("GBR", cutoff, "categorical"),
            "tenure_days": FeatureValue(float(index * 30), cutoff, "numeric"),
            "cabinet_age_days": FeatureValue(float(index * 20), cutoff, "numeric"),
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


def test_comparison_uses_identical_targets() -> None:
    report = compare_structure_and_gdelt(
        _cases(),
        holdout_start=_month(2021, 1),
        minimum_training_cases=5,
        max_history=100,
    )
    assert report.predictions > 0
    assert report.structure_analog.predictions == report.gdelt_analog.predictions
    assert report.baseline.predictions == report.predictions
