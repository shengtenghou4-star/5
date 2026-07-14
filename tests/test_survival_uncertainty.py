from datetime import datetime, timedelta, timezone
from statistics import fmean

import pytest

from fencha.engine import ForecastResult
from fencha.models import FeatureValue, HistoricalCase
from fencha.survival_uncertainty import (
    SurvivalSnapshotScore,
    country_cluster_bootstrap,
    paired_survival_snapshot_scores,
)

UTC = timezone.utc
HORIZONS = (30, 90)


def _cases() -> list[HistoricalCase]:
    result: list[HistoricalCase] = []
    start = datetime(2019, 1, 1, tzinfo=UTC)
    for index in range(24):
        cutoff = start + timedelta(days=index * 30)
        country = "A" if index % 2 == 0 else "B"
        short_exit = index % 8 == 0
        for horizon in HORIZONS:
            outcome = short_exit if horizon == 30 else (short_exit or index % 5 == 0)
            result.append(
                HistoricalCase(
                    case_id=f"snapshot-{index}:{horizon}d",
                    domain=f"government_leader_exit_{horizon}d",
                    question="Exit?",
                    cutoff_at=cutoff,
                    resolved_at=cutoff + timedelta(days=5),
                    outcome=outcome,
                    features={
                        "country_code": FeatureValue(
                            country, cutoff, "categorical"
                        ),
                        "tenure_days": FeatureValue(
                            float(index * 30), cutoff, "numeric"
                        ),
                    },
                    tags=(country, f"horizon:{horizon}"),
                )
            )
    return result


class ConstantForecaster:
    def __init__(self, probability: float) -> None:
        self.probability = probability
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
            probability=self.probability,
            prior_probability=self.probability,
            effective_sample_size=float(len(frozen)),
            neighbors=(),
            feature_coverage=1.0,
        )


def test_paired_scores_replay_complete_snapshots_without_future_labels() -> None:
    model = ConstantForecaster(0.10)
    scores = paired_survival_snapshot_scores(
        _cases(),
        model,
        holdout_start=datetime(2020, 1, 1, tzinfo=UTC),
        horizons=HORIZONS,
        minimum_training_cases=4,
        target_stride=1,
        max_history=100,
    )

    assert scores
    assert {item.country for item in scores} == {"A", "B"}
    assert all(
        item.paired_delta
        == pytest.approx(
            item.model_integrated_brier - item.baseline_integrated_brier
        )
        for item in scores
    )
    assert model.calls == len(scores) * len(HORIZONS)


def test_country_cluster_bootstrap_is_deterministic_and_paired() -> None:
    scores = [
        SurvivalSnapshotScore("a1", "A", "2020-01-01", 0.10, 0.08, -0.02),
        SurvivalSnapshotScore("a2", "A", "2020-02-01", 0.20, 0.18, -0.02),
        SurvivalSnapshotScore("b1", "B", "2020-01-01", 0.10, 0.12, 0.02),
        SurvivalSnapshotScore("c1", "C", "2020-01-01", 0.30, 0.24, -0.06),
    ]

    first = country_cluster_bootstrap(scores, replicates=500, seed=7)
    second = country_cluster_bootstrap(scores, replicates=500, seed=7)

    assert first == second
    assert first.clusters == 3
    assert first.snapshots == 4
    assert first.observed_baseline_brier == pytest.approx(
        fmean(item.baseline_integrated_brier for item in scores)
    )
    assert first.observed_model_brier == pytest.approx(
        fmean(item.model_integrated_brier for item in scores)
    )
    assert 0.0 <= first.probability_model_better <= 1.0
    assert first.delta_ci_low <= first.observed_paired_delta <= first.delta_ci_high


def test_uncertainty_analysis_validates_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        country_cluster_bootstrap([], replicates=0)
    with pytest.raises(ValueError, match="two country"):
        country_cluster_bootstrap(
            [SurvivalSnapshotScore("a", "A", "2020", 0.1, 0.1, 0.0)],
            replicates=10,
        )
    with pytest.raises(ValueError, match="horizons"):
        paired_survival_snapshot_scores(
            _cases(),
            ConstantForecaster(0.1),
            holdout_start=datetime(2020, 1, 1, tzinfo=UTC),
            horizons=(),
        )
