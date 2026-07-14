from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from fencha.competing_benchmark import temporal_competing_risk_benchmark
from fencha.engine import ForecastResult
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc
MECHANISMS = ("electoral", "other")
HORIZONS = (30, 90)


def _case(
    snapshot: str,
    cutoff: datetime,
    horizon: int,
    *,
    mechanism: str | None,
    outcome: bool,
    resolved_at: datetime,
) -> HistoricalCase:
    suffix = f":{horizon}d" + (f":{mechanism}" if mechanism else "")
    domain = "government_leader_exit_"
    if mechanism:
        domain += f"{mechanism}_{horizon}d"
    else:
        domain += f"{horizon}d"
    tags = ("TST", "test", f"horizon:{horizon}")
    if mechanism:
        tags += (f"mechanism:{mechanism}",)
    return HistoricalCase(
        case_id=f"{snapshot}{suffix}",
        domain=domain,
        question="test",
        cutoff_at=cutoff,
        resolved_at=resolved_at,
        outcome=outcome,
        features={
            "tenure_days": FeatureValue(400.0, cutoff, "numeric"),
            "minority_government": FeatureValue(True, cutoff, "boolean"),
        },
        tags=tags,
    )


def _snapshot(
    name: str,
    cutoff: datetime,
    *,
    exit_horizon: int | None,
    mechanism: str | None,
) -> list[HistoricalCase]:
    result: list[HistoricalCase] = []
    for horizon in HORIZONS:
        exited = exit_horizon is not None and horizon >= exit_horizon
        resolved = cutoff + timedelta(days=(exit_horizon if exited else horizon))
        result.append(
            _case(
                name,
                cutoff,
                horizon,
                mechanism=None,
                outcome=exited,
                resolved_at=resolved,
            )
        )
        for candidate in MECHANISMS:
            result.append(
                _case(
                    name,
                    cutoff,
                    horizon,
                    mechanism=candidate,
                    outcome=exited and candidate == mechanism,
                    resolved_at=resolved,
                )
            )
    return result


class RecordingModel:
    def __init__(self) -> None:
        self.domains: list[str] = []
        self.history_signatures: list[tuple[bool, ...]] = []

    def fit_predict(
        self,
        history: list[HistoricalCase],
        *,
        target_features: object,
        target_cutoff: object,
        domain: str | None = None,
    ) -> ForecastResult:
        assert domain is not None
        self.domains.append(domain)
        self.history_signatures.append(tuple(case.outcome for case in history))
        probabilities = {
            "government_leader_exit_30d": 0.10,
            "government_leader_exit_90d": 0.80,
            "government_leader_exit_electoral_30d": 0.05,
            "government_leader_exit_electoral_90d": 0.20,
            "government_leader_exit_other_30d": 0.05,
            "government_leader_exit_other_90d": 0.70,
        }
        return ForecastResult(
            probability=probabilities[domain],
            prior_probability=0.2,
            effective_sample_size=1.0,
            neighbors=(),
            feature_coverage=1.0,
        )


def test_benchmark_scores_total_and_conditional_mechanism_forecasts() -> None:
    training_cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    target_cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    cases = [
        *_snapshot(
            "train:TST:Leader A:2020-01-01",
            training_cutoff,
            exit_horizon=90,
            mechanism="electoral",
        ),
        *_snapshot(
            "target:TST:Leader B:2021-01-01",
            target_cutoff,
            exit_horizon=90,
            mechanism="other",
        ),
    ]
    model = RecordingModel()

    report = temporal_competing_risk_benchmark(
        cases,
        model,  # type: ignore[arg-type]
        holdout_start=target_cutoff,
        mechanisms=MECHANISMS,
        horizons=HORIZONS,
        minimum_training_cases=1,
        target_stride=1,
        max_history=None,
    )

    assert report.snapshots == 1
    assert report.exit_observations == 1
    assert report.max_adjusted_conservation_error == pytest.approx(0.0)
    assert report.adjusted_conditional_accuracy == pytest.approx(1.0)
    assert report.adjusted_conditional_log_loss < report.baseline_conditional_log_loss
    assert model.domains == [
        "government_leader_exit_30d",
        "government_leader_exit_electoral_30d",
        "government_leader_exit_other_30d",
        "government_leader_exit_90d",
        "government_leader_exit_electoral_90d",
        "government_leader_exit_other_90d",
    ]
    assert all(item.snapshots == 1 for item in report.horizon_metrics)


def test_benchmark_rejects_mechanism_label_without_total_exit() -> None:
    training_cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    target_cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    cases = [
        *_snapshot(
            "train:TST:Leader A:2020-01-01",
            training_cutoff,
            exit_horizon=90,
            mechanism="electoral",
        ),
        *_snapshot(
            "target:TST:Leader B:2021-01-01",
            target_cutoff,
            exit_horizon=None,
            mechanism=None,
        ),
    ]
    bad_index = next(
        index
        for index, case in enumerate(cases)
        if case.case_id.endswith(":30d:electoral")
        and case.cutoff_at == target_cutoff
    )
    bad = cases[bad_index]
    cases[bad_index] = HistoricalCase(
        case_id=bad.case_id,
        domain=bad.domain,
        question=bad.question,
        cutoff_at=bad.cutoff_at,
        resolved_at=bad.resolved_at,
        outcome=True,
        features=bad.features,
        tags=bad.tags,
    )

    with pytest.raises(ValueError, match="mutually exclusive"):
        temporal_competing_risk_benchmark(
            cases,
            RecordingModel(),  # type: ignore[arg-type]
            holdout_start=target_cutoff,
            mechanisms=MECHANISMS,
            horizons=HORIZONS,
            minimum_training_cases=1,
            target_stride=1,
            max_history=None,
        )


def test_future_outcomes_do_not_change_earlier_forecast_history() -> None:
    training_cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    first_cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    future_cutoff = datetime(2022, 1, 1, tzinfo=UTC)
    common = [
        *_snapshot(
            "train:TST:Leader A:2020-01-01",
            training_cutoff,
            exit_horizon=90,
            mechanism="electoral",
        ),
        *_snapshot(
            "first:TST:Leader B:2021-01-01",
            first_cutoff,
            exit_horizon=90,
            mechanism="other",
        ),
    ]
    model_a = RecordingModel()
    temporal_competing_risk_benchmark(
        [
            *common,
            *_snapshot(
                "future:TST:Leader C:2022-01-01",
                future_cutoff,
                exit_horizon=90,
                mechanism="electoral",
            ),
        ],
        model_a,  # type: ignore[arg-type]
        holdout_start=first_cutoff,
        mechanisms=MECHANISMS,
        horizons=HORIZONS,
        minimum_training_cases=1,
        target_stride=1,
        max_history=None,
    )
    model_b = RecordingModel()
    temporal_competing_risk_benchmark(
        [
            *common,
            *_snapshot(
                "future:TST:Leader C:2022-01-01",
                future_cutoff,
                exit_horizon=None,
                mechanism=None,
            ),
        ],
        model_b,  # type: ignore[arg-type]
        holdout_start=first_cutoff,
        mechanisms=MECHANISMS,
        horizons=HORIZONS,
        minimum_training_cases=1,
        target_stride=1,
        max_history=None,
    )

    assert model_a.history_signatures[:6] == model_b.history_signatures[:6]
