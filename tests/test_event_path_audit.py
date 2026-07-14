from datetime import datetime, timedelta, timezone

import pytest

from fencha.engine import ForecastResult
from fencha.event_path_audit import temporal_event_path_audit
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc
HORIZON = 90
MECHANISMS = ("post_election_transition", "other_recorded_transition")


class SignalModel:
    def fit_predict(
        self,
        history,
        *,
        target_features,
        target_cutoff,
        domain=None,
    ) -> ForecastResult:
        assert domain is not None
        signal = target_features["signal"].value
        is_other = "other_recorded_transition" in domain
        probability = 0.9 if (signal == "other") == is_other else 0.1
        return ForecastResult(
            probability=probability,
            prior_probability=0.5,
            effective_sample_size=float(len(tuple(history))),
            neighbors=(),
            feature_coverage=1.0,
        )


def _snapshot(
    leader: str,
    cutoff: datetime,
    *,
    exit_at: datetime,
    true_mechanism: str,
    signal: str,
) -> list[HistoricalCase]:
    prefix = f"parlgov:TST:{leader}:{cutoff.date()}"
    features = {
        "country_code": FeatureValue("TST", cutoff, "categorical"),
        "tenure_days": FeatureValue(500.0, cutoff, "numeric"),
        "signal": FeatureValue(signal, cutoff, "categorical"),
    }
    total = HistoricalCase(
        case_id=f"{prefix}:{HORIZON}d",
        domain=f"government_leader_exit_{HORIZON}d",
        question="Exit?",
        cutoff_at=cutoff,
        resolved_at=exit_at,
        outcome=True,
        features=features,
        tags=("TST", f"horizon:{HORIZON}"),
    )
    result = [total]
    for mechanism in MECHANISMS:
        result.append(
            HistoricalCase(
                case_id=f"{prefix}:{HORIZON}d:{mechanism}",
                domain=f"government_leader_exit_{mechanism}_{HORIZON}d",
                question="Path?",
                cutoff_at=cutoff,
                resolved_at=exit_at,
                outcome=mechanism == true_mechanism,
                features=features,
                tags=(
                    "TST",
                    f"horizon:{HORIZON}",
                    f"mechanism:{mechanism}",
                ),
            )
        )
    return result


def test_audit_deduplicates_one_exit_across_repeated_snapshots() -> None:
    cases: list[HistoricalCase] = []
    for index, mechanism in enumerate(
        (
            "post_election_transition",
            "post_election_transition",
            "other_recorded_transition",
        )
    ):
        cutoff = datetime(2019, 1, 1, tzinfo=UTC) + timedelta(days=index * 120)
        cases.extend(
            _snapshot(
                f"Train {index}",
                cutoff,
                exit_at=cutoff + timedelta(days=60),
                true_mechanism=mechanism,
                signal="other" if mechanism.startswith("other") else "election",
            )
        )

    target_exit = datetime(2021, 6, 30, tzinfo=UTC)
    cases.extend(
        _snapshot(
            "Target",
            target_exit - timedelta(days=86),
            exit_at=target_exit,
            true_mechanism="other_recorded_transition",
            signal="other",
        )
    )
    cases.extend(
        _snapshot(
            "Target",
            target_exit - timedelta(days=20),
            exit_at=target_exit,
            true_mechanism="other_recorded_transition",
            signal="other",
        )
    )

    report = temporal_event_path_audit(
        cases,
        {"signal_model": SignalModel()},
        holdout_start=datetime(2021, 1, 1, tzinfo=UTC),
        mechanisms=MECHANISMS,
        horizons=(HORIZON,),
        minimum_training_cases=1,
        minimum_exit_events=2,
        max_history=None,
        bootstrap_replicates=100,
        bootstrap_seed=7,
    )

    assert report.candidate_exit_predictions == 2
    assert report.selected_event_horizon_predictions == 1
    assert report.unique_exit_events == 1
    assert report.predictions[0].days_to_exit == pytest.approx(86.0)
    assert report.baseline.accuracy == pytest.approx(0.0)
    assert report.models[0].metrics.accuracy == pytest.approx(1.0)
    assert report.models[0].accuracy_improvement == pytest.approx(1.0)
    assert report.models[0].brier_improvement > 0
    assert report.models[0].brier_improvement_ci95.low > 0


def test_audit_validates_thresholds_and_models() -> None:
    with pytest.raises(ValueError, match="named path model"):
        temporal_event_path_audit(
            [],
            {},
            holdout_start=datetime(2021, 1, 1, tzinfo=UTC),
            mechanisms=MECHANISMS,
            horizons=(90,),
        )
