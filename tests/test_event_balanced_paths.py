from datetime import datetime, timedelta, timezone

import pytest

from fencha.engine import ForecastResult
from fencha.event_balanced_paths import (
    EventBalancedPathForecaster,
    exit_event_id,
    select_event_representatives,
)
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc
DOMAIN = "government_leader_exit_post_election_transition_90d"


class RecordingModel:
    def __init__(self) -> None:
        self.history: tuple[HistoricalCase, ...] = ()

    def fit_predict(
        self,
        history,
        *,
        target_features,
        target_cutoff,
        domain=None,
    ) -> ForecastResult:
        self.history = tuple(history)
        return ForecastResult(
            probability=0.5,
            prior_probability=0.5,
            effective_sample_size=float(len(self.history)),
            neighbors=(),
            feature_coverage=1.0,
        )


def _case(
    name: str,
    *,
    cutoff: datetime,
    exit_at: datetime,
    outcome: bool,
) -> HistoricalCase:
    return HistoricalCase(
        case_id=f"parlgov:TST:Leader A:{cutoff.date()}:90d:post_election_transition:{name}",
        domain=DOMAIN,
        question="Path?",
        cutoff_at=cutoff,
        resolved_at=exit_at,
        outcome=outcome,
        features={
            "country_code": FeatureValue("TST", cutoff, "categorical"),
            "tenure_days": FeatureValue(500.0, cutoff, "numeric"),
        },
        tags=("TST", "horizon:90", "mechanism:post_election_transition"),
    )


def test_one_exit_event_contributes_one_earliest_window_snapshot() -> None:
    exit_at = datetime(2020, 6, 30, tzinfo=UTC)
    early = _case(
        "early",
        cutoff=exit_at - timedelta(days=85),
        exit_at=exit_at,
        outcome=True,
    )
    late = _case(
        "late",
        cutoff=exit_at - timedelta(days=20),
        exit_at=exit_at,
        outcome=True,
    )
    other_exit = _case(
        "other",
        cutoff=datetime(2020, 7, 1, tzinfo=UTC),
        exit_at=datetime(2020, 8, 1, tzinfo=UTC),
        outcome=False,
    )
    no_exit = _case(
        "none",
        cutoff=datetime(2020, 9, 1, tzinfo=UTC),
        exit_at=datetime(2020, 11, 30, tzinfo=UTC),
        outcome=False,
    )

    representatives = select_event_representatives(
        [late, no_exit, other_exit, early],
        horizon_days=90,
        target_cutoff=datetime(2021, 1, 1, tzinfo=UTC),
        domain=DOMAIN,
    )

    assert [item.case_id.rsplit(":", 1)[-1] for item in representatives] == [
        "early",
        "other",
    ]
    assert exit_event_id(early, horizon_days=90) == exit_event_id(
        late,
        horizon_days=90,
    )


def test_event_balanced_wrapper_passes_unique_events_to_base_model() -> None:
    exit_at = datetime(2020, 6, 30, tzinfo=UTC)
    cases = [
        _case(
            "early",
            cutoff=exit_at - timedelta(days=80),
            exit_at=exit_at,
            outcome=True,
        ),
        _case(
            "late",
            cutoff=exit_at - timedelta(days=10),
            exit_at=exit_at,
            outcome=True,
        ),
    ]
    recorder = RecordingModel()
    model = EventBalancedPathForecaster(recorder)
    target_cutoff = datetime(2021, 1, 1, tzinfo=UTC)

    result = model.fit_predict(
        cases,
        target_features={
            "country_code": FeatureValue("TST", target_cutoff, "categorical")
        },
        target_cutoff=target_cutoff,
        domain=DOMAIN,
    )

    assert result.effective_sample_size == pytest.approx(1.0)
    assert len(recorder.history) == 1
    assert recorder.history[0].case_id.endswith(":early")


def test_event_balanced_wrapper_rejects_total_domain() -> None:
    model = EventBalancedPathForecaster(RecordingModel())
    cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError):
        model.fit_predict(
            [],
            target_features={
                "country_code": FeatureValue("TST", cutoff, "categorical")
            },
            target_cutoff=cutoff,
            domain="government_leader_exit_90d",
        )
