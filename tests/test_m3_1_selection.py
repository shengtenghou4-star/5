from datetime import datetime, timedelta, timezone

import pytest

from fencha.hierarchical import HierarchicalRiskConfig
from fencha.m3_1_select import (
    default_candidate_configs,
    select_hierarchical_survival_model,
)
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc
HORIZONS = (30, 90)


def _month(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


def _cases(*, alternate_holdout: bool = False) -> list[HistoricalCase]:
    result: list[HistoricalCase] = []
    start = _month(2008, 1)
    for index in range(120):
        cutoff = start + timedelta(days=index * 30)
        country = "A" if index % 2 == 0 else "B"
        base_exit = index % (10 if country == "A" else 6) == 0
        for horizon in HORIZONS:
            outcome = base_exit if horizon == 30 else (base_exit or index % 8 == 0)
            if alternate_holdout and cutoff >= _month(2015, 1):
                short_exit = index % 5 == 0
                outcome = short_exit if horizon == 30 else (short_exit or index % 3 == 0)
            result.append(
                HistoricalCase(
                    case_id=f"snapshot-{index}:{horizon}d",
                    domain=f"government_leader_exit_{horizon}d",
                    question="Will the leader leave?",
                    cutoff_at=cutoff,
                    resolved_at=cutoff + timedelta(days=10),
                    outcome=outcome,
                    features={
                        "country_code": FeatureValue(
                            country, cutoff, "categorical"
                        ),
                        "tenure_days": FeatureValue(
                            float((index % 24) * 30), cutoff, "numeric"
                        ),
                        "caretaker": FeatureValue(False, cutoff, "boolean"),
                        "cabinet_type": FeatureValue(
                            "majority", cutoff, "categorical"
                        ),
                    },
                    tags=(country, f"horizon:{horizon}"),
                )
            )
    return result


def _small_candidates():
    return (
        (
            "global",
            HierarchicalRiskConfig(
                use_country=False,
                use_tenure=False,
                use_context=False,
            ),
        ),
        (
            "country",
            HierarchicalRiskConfig(
                use_country=True,
                use_tenure=False,
                use_context=False,
                country_strength=25,
            ),
        ),
        (
            "country_tenure",
            HierarchicalRiskConfig(
                use_country=True,
                use_tenure=True,
                use_context=False,
                tenure_bucket_days=365,
                country_strength=100,
                tenure_strength=100,
            ),
        ),
    )


def _select(cases: list[HistoricalCase]):
    return select_hierarchical_survival_model(
        cases,
        validation_start=_month(2012, 1),
        validation_end=_month(2015, 1),
        holdout_start=_month(2015, 1),
        horizons=HORIZONS,
        minimum_training_cases=20,
        validation_target_stride=2,
        holdout_target_stride=2,
        max_history=500,
        candidates=_small_candidates(),
    )


def test_default_grid_is_predeclared_and_unique() -> None:
    candidates = default_candidate_configs()
    assert len(candidates) == 31
    assert len(set(candidates)) == len(candidates)
    assert candidates[0][0] == "global"
    assert {name for name, _ in candidates} == {
        "global",
        "country",
        "tenure",
        "country_tenure",
        "country_tenure_context",
    }


def test_hierarchy_selection_is_independent_of_holdout_outcomes() -> None:
    original = _select(_cases())
    alternate = _select(_cases(alternate_holdout=True))

    assert original.selected_template in {
        "global",
        "country",
        "country_tenure",
    }
    assert len(original.candidates) == 3
    assert original.final_holdout.snapshots > 0
    assert len(original.final_holdout.snapshot_ids) == original.final_holdout.snapshots

    # The alternate holdout remains logically monotone, but changes the final
    # labels. It may change the final score, never the pre-2015 selection.
    assert alternate.selected_template == original.selected_template
    assert alternate.selected_config == original.selected_config
    assert alternate.selected_validation == original.selected_validation
    assert alternate.final_holdout.adjusted_integrated_brier != pytest.approx(
        original.final_holdout.adjusted_integrated_brier
    )


def test_hierarchy_selection_validates_design() -> None:
    with pytest.raises(ValueError, match="validation_start"):
        select_hierarchical_survival_model(
            _cases(),
            validation_start=_month(2015, 1),
            validation_end=_month(2014, 1),
            holdout_start=_month(2015, 1),
            candidates=_small_candidates(),
        )

    with pytest.raises(ValueError, match="at least one hierarchy"):
        select_hierarchical_survival_model(
            _cases(),
            validation_start=_month(2012, 1),
            validation_end=_month(2015, 1),
            holdout_start=_month(2015, 1),
            candidates=(),
        )

    duplicate = (_small_candidates()[0], _small_candidates()[0])
    with pytest.raises(ValueError, match="unique"):
        select_hierarchical_survival_model(
            _cases(),
            validation_start=_month(2012, 1),
            validation_end=_month(2015, 1),
            holdout_start=_month(2015, 1),
            candidates=duplicate,
        )
