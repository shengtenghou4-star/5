from datetime import datetime, timezone

import pytest

from fencha.competing_risks import (
    coherent_competing_risk_curve,
    forecast_competing_risks,
)
from fencha.engine import ForecastResult
from fencha.models import FeatureValue


def test_curve_conserves_total_and_each_mechanism_is_monotone() -> None:
    points, _ = coherent_competing_risk_curve(
        {30: 0.3, 90: 0.2, 180: 0.7},
        {
            "electoral": {30: 0.2, 90: 0.4, 180: 0.5},
            "other": {30: 0.1, 90: 0.1, 180: 0.6},
        },
    )

    assert [point.total_exit_probability for point in points] == [0.25, 0.25, 0.7]
    for point in points:
        assert sum(
            item.cumulative_probability for item in point.mechanisms
        ) == pytest.approx(point.total_exit_probability)
        assert sum(
            item.interval_probability for item in point.mechanisms
        ) == pytest.approx(point.interval_exit_probability)

    for index in range(2):
        values = [point.mechanisms[index].cumulative_probability for point in points]
        assert values == sorted(values)


def test_all_zero_scores_split_new_exit_mass_evenly() -> None:
    points, _ = coherent_competing_risk_curve(
        {30: 0.2},
        {"electoral": {}, "other": {}},
    )
    assert points[0].mechanisms[0].interval_probability == pytest.approx(0.1)
    assert points[0].mechanisms[1].interval_probability == pytest.approx(0.1)


def test_rejects_mechanism_horizons_absent_from_total_curve() -> None:
    with pytest.raises(ValueError, match="absent from total"):
        coherent_competing_risk_curve(
            {30: 0.2},
            {"electoral": {90: 0.1}},
        )


class RecordingModel:
    def __init__(self) -> None:
        self.domains: list[str | None] = []

    def fit_predict(
        self,
        history: object,
        *,
        target_features: object,
        target_cutoff: object,
        domain: str | None = None,
    ) -> ForecastResult:
        self.domains.append(domain)
        probability = 0.4 if domain and domain.endswith("_30d") else 0.6
        return ForecastResult(
            probability=probability,
            prior_probability=0.2,
            effective_sample_size=3.0,
            neighbors=(),
            feature_coverage=0.8,
        )


def test_forecast_routes_total_and_mechanism_domains() -> None:
    model = RecordingModel()
    cutoff = datetime(2024, 1, 1, tzinfo=timezone.utc)
    features = {"x": FeatureValue(1.0, cutoff, "numeric")}

    result = forecast_competing_risks(
        [],
        target_features=features,
        target_cutoff=cutoff,
        mechanisms=("other", "electoral"),
        horizons=(90, 30),
        model=model,  # type: ignore[arg-type]
    )

    assert model.domains == [
        "government_leader_exit_30d",
        "government_leader_exit_electoral_30d",
        "government_leader_exit_other_30d",
        "government_leader_exit_90d",
        "government_leader_exit_electoral_90d",
        "government_leader_exit_other_90d",
    ]
    assert result.mechanisms == ("electoral", "other")
    assert sum(
        item.cumulative_probability for item in result.points[-1].mechanisms
    ) == pytest.approx(result.points[-1].total_exit_probability)
