from datetime import datetime, timedelta, timezone

import pytest

from fencha.hierarchical import HierarchicalRiskConfig, HierarchicalRiskForecaster
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc


def _case(
    index: int,
    *,
    country: str,
    tenure_days: float,
    outcome: bool,
    resolved_before_target: bool = True,
) -> HistoricalCase:
    cutoff = datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=index)
    resolved = (
        datetime(2021, 1, 1, tzinfo=UTC) - timedelta(days=1)
        if resolved_before_target
        else datetime(2021, 1, 1, tzinfo=UTC) + timedelta(days=1)
    )
    return HistoricalCase(
        case_id=f"case-{index}",
        domain="government_leader_exit_30d",
        question="Will the leader leave?",
        cutoff_at=cutoff,
        resolved_at=max(resolved, cutoff + timedelta(days=1)),
        outcome=outcome,
        features={
            "country_code": FeatureValue(country, cutoff, "categorical"),
            "tenure_days": FeatureValue(tenure_days, cutoff, "numeric"),
            "caretaker": FeatureValue(False, cutoff, "boolean"),
            "cabinet_type": FeatureValue("majority", cutoff, "categorical"),
        },
        tags=(country,),
    )


def _target(country: str, tenure_days: float):
    cutoff = datetime(2021, 1, 1, tzinfo=UTC)
    return cutoff, {
        "country_code": FeatureValue(country, cutoff, "categorical"),
        "tenure_days": FeatureValue(tenure_days, cutoff, "numeric"),
        "caretaker": FeatureValue(False, cutoff, "boolean"),
        "cabinet_type": FeatureValue("majority", cutoff, "categorical"),
    }


def test_global_configuration_reproduces_smoothed_base_rate() -> None:
    history = [
        _case(index, country="A", tenure_days=100, outcome=index < 2)
        for index in range(10)
    ]
    cutoff, features = _target("A", 100)
    model = HierarchicalRiskForecaster(
        HierarchicalRiskConfig(
            use_country=False,
            use_tenure=False,
            use_context=False,
        )
    )

    result = model.fit_predict(
        history,
        target_features=features,
        target_cutoff=cutoff,
        domain="government_leader_exit_30d",
    )

    assert result.probability == pytest.approx(3 / 12)
    assert result.prior_probability == pytest.approx(3 / 12)
    assert result.feature_coverage == pytest.approx(1.0)


def test_sparse_country_rate_is_shrunk_toward_global_rate() -> None:
    history = [
        _case(index, country="A", tenure_days=100, outcome=False)
        for index in range(100)
    ]
    history.extend(
        _case(200 + index, country="B", tenure_days=100, outcome=True)
        for index in range(2)
    )
    cutoff, features = _target("B", 100)
    model = HierarchicalRiskForecaster(
        HierarchicalRiskConfig(
            use_country=True,
            use_tenure=False,
            country_strength=50,
        )
    )

    result = model.fit_predict(
        history,
        target_features=features,
        target_cutoff=cutoff,
        domain="government_leader_exit_30d",
    )

    assert result.prior_probability < result.probability < 0.10
    assert result.effective_sample_size == pytest.approx(2.0)
    assert result.feature_coverage == pytest.approx(1.0)


def test_hierarchy_ignores_unresolved_future_labels() -> None:
    history = [
        _case(index, country="A", tenure_days=100, outcome=False)
        for index in range(20)
    ]
    history.append(
        _case(
            100,
            country="A",
            tenure_days=100,
            outcome=True,
            resolved_before_target=False,
        )
    )
    cutoff, features = _target("A", 100)
    model = HierarchicalRiskForecaster(
        HierarchicalRiskConfig(use_country=False, use_tenure=False)
    )

    result = model.fit_predict(
        history,
        target_features=features,
        target_cutoff=cutoff,
        domain="government_leader_exit_30d",
    )

    assert result.probability == pytest.approx(1 / 22)


def test_tenure_cell_is_nested_inside_country() -> None:
    history = []
    for index in range(20):
        history.append(
            _case(
                index,
                country="A",
                tenure_days=50 if index < 10 else 500,
                outcome=index in {0, 1},
            )
        )
    for index in range(20):
        history.append(
            _case(
                100 + index,
                country="B",
                tenure_days=50,
                outcome=True,
            )
        )
    cutoff, features = _target("A", 50)
    model = HierarchicalRiskForecaster(
        HierarchicalRiskConfig(
            use_country=True,
            use_tenure=True,
            tenure_bucket_days=365,
            country_strength=10,
            tenure_strength=10,
        )
    )

    result = model.fit_predict(
        history,
        target_features=features,
        target_cutoff=cutoff,
        domain="government_leader_exit_30d",
    )

    # Country B's high short-tenure risk must not directly populate country A's
    # nested tenure cell.
    assert result.probability < 0.35
    assert result.effective_sample_size == pytest.approx(10.0)
    assert result.feature_coverage == pytest.approx(1.0)


def test_hierarchy_rejects_future_target_features_and_bad_config() -> None:
    cutoff, features = _target("A", 100)
    features["future"] = FeatureValue(1.0, cutoff + timedelta(days=1), "numeric")
    model = HierarchicalRiskForecaster()

    with pytest.raises(ValueError, match="future information"):
        model.fit_predict(
            [_case(0, country="A", tenure_days=100, outcome=False)],
            target_features=features,
            target_cutoff=cutoff,
        )

    with pytest.raises(ValueError, match="tenure_bucket_days"):
        HierarchicalRiskConfig(tenure_bucket_days=0)
    with pytest.raises(ValueError, match="country_strength"):
        HierarchicalRiskConfig(country_strength=0)
