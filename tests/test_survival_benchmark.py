from datetime import datetime, timedelta, timezone

import pytest

from fencha.engine import ForecastResult
from fencha.models import FeatureValue, HistoricalCase
from fencha.survival_benchmark import temporal_survival_benchmark

UTC = timezone.utc
HORIZONS = (30, 90, 180)


def _outcomes(index: int) -> dict[int, bool]:
    bucket = index % 4
    return {
        30: bucket == 3,
        90: bucket >= 2,
        180: bucket >= 1,
    }


def _cases() -> list[HistoricalCase]:
    result: list[HistoricalCase] = []
    start = datetime(2019, 1, 1, tzinfo=UTC)
    for index in range(12):
        cutoff = start + timedelta(days=index * 30)
        for horizon, outcome in _outcomes(index).items():
            result.append(
                HistoricalCase(
                    case_id=f"snapshot-{index}:{horizon}d",
                    domain=f"government_leader_exit_{horizon}d",
                    question="Will the leader leave?",
                    cutoff_at=cutoff,
                    resolved_at=cutoff + timedelta(days=5),
                    outcome=outcome,
                    features={
                        "country_code": FeatureValue(
                            "TST", cutoff, "categorical"
                        ),
                        "tenure_days": FeatureValue(
                            float(index * 30), cutoff, "numeric"
                        ),
                    },
                    tags=("TST", f"horizon:{horizon}"),
                )
            )
    return result


class CrossingForecaster:
    probabilities = {
        "government_leader_exit_30d": 0.60,
        "government_leader_exit_90d": 0.30,
        "government_leader_exit_180d": 0.80,
    }

    def __init__(self) -> None:
        self.calls = 0

    def fit_predict(
        self,
        history,
        *,
        target_features,
        target_cutoff,
        domain=None,
    ) -> ForecastResult:
        frozen = tuple(history)
        assert frozen
        assert all(case.resolved_at < target_cutoff for case in frozen)
        self.calls += 1
        return ForecastResult(
            probability=self.probabilities[domain],
            prior_probability=0.25,
            effective_sample_size=float(len(frozen)),
            neighbors=(),
            feature_coverage=1.0,
        )


def test_survival_benchmark_scores_raw_and_reconciled_curves() -> None:
    model = CrossingForecaster()
    report = temporal_survival_benchmark(
        _cases(),
        model,  # type: ignore[arg-type]
        holdout_start=datetime(2019, 7, 1, tzinfo=UTC),
        horizons=HORIZONS,
        minimum_training_cases=2,
        target_stride=1,
        max_history=100,
    )

    assert report.snapshots > 0
    assert report.raw_crossing_curves == report.snapshots
    assert report.raw_crossing_rate == pytest.approx(1.0)
    assert report.mean_crossing_magnitude == pytest.approx(0.30)
    assert report.mean_absolute_adjustment > 0
    assert model.calls == report.snapshots * len(HORIZONS)
    assert len(report.snapshot_ids) == report.snapshots
    assert [item.horizon_days for item in report.horizon_metrics] == list(HORIZONS)
    assert all(item.predictions == report.snapshots for item in report.horizon_metrics)
    assert report.first_cutoff < report.last_cutoff


def test_incomplete_snapshots_are_not_scored() -> None:
    cases = [
        case
        for case in _cases()
        if case.case_id != "snapshot-8:180d"
    ]
    complete = temporal_survival_benchmark(
        _cases(),
        CrossingForecaster(),  # type: ignore[arg-type]
        holdout_start=datetime(2019, 7, 1, tzinfo=UTC),
        horizons=HORIZONS,
        minimum_training_cases=2,
        target_stride=1,
    )
    incomplete = temporal_survival_benchmark(
        cases,
        CrossingForecaster(),  # type: ignore[arg-type]
        holdout_start=datetime(2019, 7, 1, tzinfo=UTC),
        horizons=HORIZONS,
        minimum_training_cases=2,
        target_stride=1,
    )

    assert incomplete.snapshots == complete.snapshots - 1
    assert "snapshot-8" not in incomplete.snapshot_ids


def test_non_monotone_historical_labels_are_rejected() -> None:
    cases = _cases()
    target_index = next(
        index for index, case in enumerate(cases) if case.case_id == "snapshot-8:30d"
    )
    target = cases[target_index]
    cases[target_index] = HistoricalCase(
        case_id=target.case_id,
        domain=target.domain,
        question=target.question,
        cutoff_at=target.cutoff_at,
        resolved_at=target.resolved_at,
        outcome=True,
        features=target.features,
        tags=target.tags,
    )
    target_90_index = next(
        index for index, case in enumerate(cases) if case.case_id == "snapshot-8:90d"
    )
    target_90 = cases[target_90_index]
    cases[target_90_index] = HistoricalCase(
        case_id=target_90.case_id,
        domain=target_90.domain,
        question=target_90.question,
        cutoff_at=target_90.cutoff_at,
        resolved_at=target_90.resolved_at,
        outcome=False,
        features=target_90.features,
        tags=target_90.tags,
    )

    with pytest.raises(ValueError, match="non-monotone outcome"):
        temporal_survival_benchmark(
            cases,
            CrossingForecaster(),  # type: ignore[arg-type]
            holdout_start=datetime(2019, 7, 1, tzinfo=UTC),
            horizons=HORIZONS,
            minimum_training_cases=2,
            target_stride=1,
        )


def test_survival_benchmark_validates_configuration() -> None:
    with pytest.raises(ValueError, match="horizons"):
        temporal_survival_benchmark(
            _cases(),
            CrossingForecaster(),  # type: ignore[arg-type]
            holdout_start=datetime(2019, 7, 1, tzinfo=UTC),
            horizons=(),
        )
    with pytest.raises(ValueError, match="minimum_training_cases"):
        temporal_survival_benchmark(
            _cases(),
            CrossingForecaster(),  # type: ignore[arg-type]
            holdout_start=datetime(2019, 7, 1, tzinfo=UTC),
            minimum_training_cases=0,
        )
