from datetime import datetime, timedelta, timezone

import pytest

from fencha.engine import ForecastResult
from fencha.hybrid_competing import HybridCompetingForecaster
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc


class RecordingForecaster:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.domains: list[str | None] = []

    def fit_predict(
        self,
        history,
        *,
        target_features,
        target_cutoff,
        domain=None,
    ) -> ForecastResult:
        frozen = tuple(history)
        assert all(case.resolved_at < target_cutoff for case in frozen)
        self.domains.append(domain)
        return ForecastResult(
            probability=self.probability,
            prior_probability=self.probability,
            effective_sample_size=float(len(frozen)),
            neighbors=(),
            feature_coverage=1.0,
        )


def _case(domain: str) -> HistoricalCase:
    cutoff = datetime(2020, 1, 1, tzinfo=UTC)
    return HistoricalCase(
        case_id=domain,
        domain=domain,
        question="Exit?",
        cutoff_at=cutoff,
        resolved_at=cutoff + timedelta(days=1),
        outcome=False,
        features={
            "country_code": FeatureValue("GBR", cutoff, "categorical")
        },
        tags=("GBR",),
    )


def test_hybrid_routes_total_and_mechanism_domains_separately() -> None:
    total = RecordingForecaster(0.10)
    mechanism = RecordingForecaster(0.70)
    hybrid = HybridCompetingForecaster(
        total_model=total,
        mechanism_model=mechanism,
    )
    target_cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    features = {
        "country_code": FeatureValue("GBR", target_cutoff, "categorical")
    }

    total_result = hybrid.fit_predict(
        [_case("government_leader_exit_90d")],
        target_features=features,
        target_cutoff=target_cutoff,
        domain="government_leader_exit_90d",
    )
    mechanism_result = hybrid.fit_predict(
        [_case("government_leader_exit_post_election_transition_90d")],
        target_features=features,
        target_cutoff=target_cutoff,
        domain="government_leader_exit_post_election_transition_90d",
    )

    assert total_result.probability == pytest.approx(0.10)
    assert mechanism_result.probability == pytest.approx(0.70)
    assert total.domains == ["government_leader_exit_90d"]
    assert mechanism.domains == [
        "government_leader_exit_post_election_transition_90d"
    ]


def test_hybrid_rejects_missing_or_malformed_domains() -> None:
    hybrid = HybridCompetingForecaster(
        total_model=RecordingForecaster(0.1),
        mechanism_model=RecordingForecaster(0.2),
    )
    cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    features = {"country_code": FeatureValue("GBR", cutoff, "categorical")}

    for domain in (None, "other_90d", "government_leader_exit_0d"):
        with pytest.raises(ValueError):
            hybrid.fit_predict(
                [],
                target_features=features,
                target_cutoff=cutoff,
                domain=domain,
            )


def test_hybrid_validates_prefix() -> None:
    with pytest.raises(ValueError, match="domain_prefix"):
        HybridCompetingForecaster(
            total_model=RecordingForecaster(0.1),
            mechanism_model=RecordingForecaster(0.2),
            domain_prefix="__",
        )
