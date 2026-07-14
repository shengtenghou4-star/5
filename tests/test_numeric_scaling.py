from datetime import datetime, timedelta, timezone

import pytest

from fencha.engine import AnalogForecaster
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc


def _case(index: int, value: float) -> HistoricalCase:
    cutoff = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index * 2)
    return HistoricalCase(
        case_id=f"scale-{index}",
        domain="scale-test",
        question="Scale?",
        cutoff_at=cutoff,
        resolved_at=cutoff + timedelta(days=1),
        outcome=bool(index % 2),
        features={"signal": FeatureValue(value, cutoff, "numeric")},
    )


def test_iqr_scale_uses_history_only() -> None:
    history = [_case(index, float(index)) for index in range(4)]
    target_cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    target = {
        "signal": FeatureValue(10_000.0, target_cutoff, "numeric"),
    }

    robust = AnalogForecaster(numeric_scale="iqr")
    original = AnalogForecaster(numeric_scale="range")

    assert robust._numeric_scales(history, target)["signal"] == pytest.approx(1.5)
    assert original._numeric_scales(history, target)["signal"] == pytest.approx(
        10_000.0
    )


def test_invalid_numeric_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="numeric_scale"):
        AnalogForecaster(numeric_scale="standard-deviation")  # type: ignore[arg-type]
