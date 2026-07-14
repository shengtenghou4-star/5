from datetime import datetime, timedelta, timezone

import pytest

from fencha.engine import AnalogForecaster
from fencha.models import FeatureValue, HistoricalCase
from fencha.survival import coherent_survival_curve, forecast_survival_curve

UTC = timezone.utc


def test_coherent_curve_repairs_crossing_horizon_probabilities() -> None:
    points, restricted_mean = coherent_survival_curve(
        {30: 0.30, 90: 0.20, 180: 0.70}
    )

    assert [point.adjusted_exit_probability for point in points] == pytest.approx(
        [0.25, 0.25, 0.70]
    )
    assert [point.interval_exit_probability for point in points] == pytest.approx(
        [0.25, 0.0, 0.45]
    )
    assert [point.survival_probability for point in points] == pytest.approx(
        [0.75, 0.75, 0.30]
    )
    assert restricted_mean == pytest.approx(118.5)
    assert sum(point.interval_exit_probability for point in points) == pytest.approx(
        points[-1].adjusted_exit_probability
    )


def _case(horizon: int, index: int, outcome: bool) -> HistoricalCase:
    cutoff = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index * 30)
    return HistoricalCase(
        case_id=f"history:{horizon}:{index}",
        domain=f"government_leader_exit_{horizon}d",
        question="test",
        cutoff_at=cutoff,
        resolved_at=cutoff + timedelta(days=10),
        outcome=outcome,
        features={
            "tenure_days": FeatureValue(100.0, cutoff, "numeric"),
            "country_code": FeatureValue("TST", cutoff, "categorical"),
        },
    )


def test_forecast_survival_curve_uses_horizon_domains_and_returns_valid_curve() -> None:
    history = [
        _case(30, 0, True),
        _case(30, 1, True),
        _case(30, 2, False),
        _case(90, 0, False),
        _case(90, 1, False),
        _case(90, 2, False),
        _case(180, 0, True),
        _case(180, 1, True),
        _case(180, 2, True),
    ]
    target_cutoff = datetime(2022, 1, 1, tzinfo=UTC)
    target_features = {
        "tenure_days": FeatureValue(100.0, target_cutoff, "numeric"),
        "country_code": FeatureValue("TST", target_cutoff, "categorical"),
    }
    curve = forecast_survival_curve(
        history,
        target_features=target_features,
        target_cutoff=target_cutoff,
        horizons=(180, 30, 90),
        model=AnalogForecaster(prior_strength=1.0, top_k=3),
    )

    assert [point.horizon_days for point in curve.points] == [30, 90, 180]
    adjusted = [point.adjusted_exit_probability for point in curve.points]
    assert adjusted == sorted(adjusted)
    assert all(point.interval_exit_probability >= 0 for point in curve.points)
    assert all(point.effective_sample_size == pytest.approx(3.0) for point in curve.points)
    assert curve.restricted_mean_survival_days > 0


def test_curve_rejects_invalid_probabilities() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        coherent_survival_curve({30: 1.1})
    with pytest.raises(ValueError, match="positive"):
        coherent_survival_curve({0: 0.2})
