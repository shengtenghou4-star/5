from datetime import datetime, timedelta, timezone

import pytest

from fencha.conditioned_paths import (
    ExitConditionedPathForecaster,
    is_realized_exit_case,
    mechanism_horizon,
)
from fencha.engine import ForecastResult
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc
DOMAIN = "government_leader_exit_post_election_transition_90d"


class RecordingForecaster:
    def __init__(self) -> None:
        self.histories: list[tuple[HistoricalCase, ...]] = []

    def fit_predict(
        self,
        history,
        *,
        target_features,
        target_cutoff,
        domain=None,
    ) -> ForecastResult:
        frozen = tuple(history)
        self.histories.append(frozen)
        positives = sum(case.outcome for case in frozen)
        probability = (positives + 1.0) / (len(frozen) + 2.0)
        return ForecastResult(
            probability=probability,
            prior_probability=probability,
            effective_sample_size=float(len(frozen)),
            neighbors=(),
            feature_coverage=1.0,
        )


def _case(
    name: str,
    *,
    outcome: bool,
    resolution_days: int,
    domain: str = DOMAIN,
    total_exit: bool | None = None,
) -> HistoricalCase:
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    tags = ("TST", "horizon:90", "mechanism:post_election_transition")
    if total_exit is not None:
        tags += (f"total_exit:{int(total_exit)}",)
    return HistoricalCase(
        case_id=name,
        domain=domain,
        question="Exit through this path?",
        cutoff_at=cutoff,
        resolved_at=cutoff + timedelta(days=resolution_days),
        outcome=outcome,
        features={
            "country_code": FeatureValue("TST", cutoff, "categorical"),
            "tenure_days": FeatureValue(500.0, cutoff, "numeric"),
        },
        tags=tags,
    )


def test_conditioned_model_keeps_only_realized_exits() -> None:
    base = RecordingForecaster()
    model = ExitConditionedPathForecaster(base)
    target_cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    features = {
        "country_code": FeatureValue("TST", target_cutoff, "categorical"),
        "tenure_days": FeatureValue(700.0, target_cutoff, "numeric"),
    }
    positive_path = _case("positive-path", outcome=True, resolution_days=40)
    other_path_exit = _case("other-path-exit", outcome=False, resolution_days=55)
    no_exit = _case("no-exit", outcome=False, resolution_days=90)
    exact_boundary_positive = _case(
        "boundary-positive",
        outcome=True,
        resolution_days=90,
    )
    exact_boundary_other_path = _case(
        "boundary-other-path",
        outcome=False,
        resolution_days=90,
        total_exit=True,
    )

    result = model.fit_predict(
        [
            positive_path,
            other_path_exit,
            no_exit,
            exact_boundary_positive,
            exact_boundary_other_path,
        ],
        target_features=features,
        target_cutoff=target_cutoff,
        domain=DOMAIN,
    )

    assert [case.case_id for case in base.histories[0]] == [
        "positive-path",
        "other-path-exit",
        "boundary-positive",
        "boundary-other-path",
    ]
    assert result.effective_sample_size == pytest.approx(4.0)
    assert result.probability == pytest.approx(3 / 6)


def test_explicit_total_exit_tag_resolves_boundary_ambiguity() -> None:
    assert is_realized_exit_case(
        _case(
            "boundary-other",
            outcome=False,
            resolution_days=90,
            total_exit=True,
        ),
        horizon_days=90,
    )
    assert not is_realized_exit_case(
        _case(
            "boundary-none",
            outcome=False,
            resolution_days=90,
            total_exit=False,
        ),
        horizon_days=90,
    )


def test_exit_filter_falls_back_for_older_datasets() -> None:
    assert is_realized_exit_case(
        _case("other", outcome=False, resolution_days=89),
        horizon_days=90,
    )
    assert not is_realized_exit_case(
        _case("none", outcome=False, resolution_days=90),
        horizon_days=90,
    )
    assert is_realized_exit_case(
        _case("positive", outcome=True, resolution_days=90),
        horizon_days=90,
    )


def test_invalid_total_exit_tags_are_rejected() -> None:
    invalid = _case("invalid", outcome=True, resolution_days=20, total_exit=False)
    with pytest.raises(ValueError, match="positive mechanism"):
        is_realized_exit_case(invalid, horizon_days=90)


def test_mechanism_domain_parser_rejects_total_and_bad_domains() -> None:
    assert mechanism_horizon(DOMAIN, domain_prefix="government_leader_exit") == 90
    for domain in (
        None,
        "government_leader_exit_90d",
        "government_leader_exit_path_xd",
        "other_path_90d",
    ):
        with pytest.raises(ValueError):
            mechanism_horizon(domain, domain_prefix="government_leader_exit")


def test_conditioned_model_rejects_empty_prefix() -> None:
    with pytest.raises(ValueError, match="domain_prefix"):
        ExitConditionedPathForecaster(RecordingForecaster(), domain_prefix="__")
