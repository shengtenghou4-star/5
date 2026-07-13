from datetime import datetime, timezone

import pytest

from fencha.demo import demo_cases, dt, feature
from fencha.engine import AnalogForecaster
from fencha.models import FeatureValue, HistoricalCase
from fencha.scoring import walk_forward_backtest

UTC = timezone.utc


def test_case_rejects_future_information() -> None:
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="future information leakage"):
        HistoricalCase(
            case_id="leak",
            domain="test",
            question="Leak?",
            cutoff_at=cutoff,
            resolved_at=datetime(2021, 1, 1, tzinfo=UTC),
            outcome=True,
            features={
                "future": FeatureValue(
                    value=1.0,
                    observed_at=datetime(2020, 1, 2, tzinfo=UTC),
                    kind="numeric",
                )
            },
        )


def test_forecaster_only_uses_resolved_past_cases() -> None:
    cases = demo_cases()
    model = AnalogForecaster(top_k=100)
    cutoff = dt(2016, 3, 1)
    result = model.fit_predict(
        cases,
        target_features={
            "pressure_index": feature(0.7, cutoff),
            "coalition_structure": feature("tight", cutoff),
            "recent_trigger": feature(True, cutoff),
        },
        target_cutoff=cutoff,
        domain="world_demo",
    )
    assert result.neighbors
    assert all(neighbor.cutoff_at < cutoff for neighbor in result.neighbors)
    assert {neighbor.case_id for neighbor in result.neighbors} <= {
        "w01", "w02", "w03", "w04", "w05"
    }


def test_walk_forward_backtest_is_deterministic() -> None:
    cases = demo_cases()
    model = AnalogForecaster(top_k=5, prior_strength=3.0)
    first = walk_forward_backtest(cases, model, minimum_training_cases=3)
    second = walk_forward_backtest(reversed(cases), model, minimum_training_cases=3)
    assert first == second
    assert len(first.points) > 0
    assert 0.0 <= first.brier_score <= 1.0
