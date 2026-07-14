from datetime import datetime, timedelta, timezone

import pytest

from fencha.m2_1_select import select_and_evaluate
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc


def _month(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


def _cases(*, invert_holdout: bool = False) -> list[HistoricalCase]:
    cases: list[HistoricalCase] = []
    for index in range(48):
        year = 2019 + index // 12
        month = index % 12 + 1
        cutoff = _month(year, month)
        outcome = index % 6 == 0
        if invert_holdout and cutoff >= _month(2022, 1):
            outcome = not outcome
        cases.append(
            HistoricalCase(
                case_id=f"select-{index}",
                domain="government_leader_exit_180d",
                question="Will the leader leave?",
                cutoff_at=cutoff,
                resolved_at=cutoff + timedelta(days=20),
                outcome=outcome,
                features={
                    "country_code": FeatureValue("GBR", cutoff, "categorical"),
                    "tenure_days": FeatureValue(float(index * 30), cutoff, "numeric"),
                    "cabinet_age_days": FeatureValue(
                        float(index * 20), cutoff, "numeric"
                    ),
                    "gdelt_events_per_sample_30d": FeatureValue(
                        float(10 + index * 2), cutoff, "numeric"
                    ),
                    "gdelt_articles_per_sample_30d": FeatureValue(
                        float(20 + index * 3), cutoff, "numeric"
                    ),
                    "gdelt_protest_share_30d": FeatureValue(
                        0.35 if outcome else 0.03, cutoff, "numeric"
                    ),
                    "gdelt_avg_tone_30d": FeatureValue(
                        -4.0 if outcome else -0.5, cutoff, "numeric"
                    ),
                    "gdelt_coverage_30d": FeatureValue(1.0, cutoff, "numeric"),
                    "gdelt_coverage_90d": FeatureValue(1.0, cutoff, "numeric"),
                },
                tags=("GBR", "selection-test"),
            )
        )
    return cases


def _run(cases: list[HistoricalCase]):
    return select_and_evaluate(
        cases,
        validation_start=_month(2020, 1),
        validation_end=_month(2022, 1),
        holdout_start=_month(2022, 1),
        minimum_training_cases=5,
        validation_target_stride=1,
        holdout_target_stride=1,
        max_history=100,
        top_k_candidates=(3, 5),
        numeric_scale_candidates=("range",),
        signal_family_candidates=("conflict",),
        gdelt_multiplier_candidates=(0.1, 1.0),
    )


def test_selection_is_frozen_before_holdout_outcomes() -> None:
    original, original_diagnostics = _run(_cases())
    inverted, inverted_diagnostics = _run(_cases(invert_holdout=True))

    assert original.selected_top_k in {3, 5}
    assert original.selected_numeric_scale == "range"
    assert original.selected_signal_family == "conflict"
    assert original.selected_gdelt_multiplier in {0.1, 1.0}
    assert len(original.structure_candidates) == 2
    assert len(original.gdelt_candidates) == 2
    assert original.final_holdout.predictions == len(original_diagnostics)
    assert original.final_holdout.first_cutoff.startswith("2022-")

    # Changing every final-holdout outcome may change the final score, but it
    # must not change any setting selected from pre-2022 validation data.
    assert inverted.selected_top_k == original.selected_top_k
    assert inverted.selected_numeric_scale == original.selected_numeric_scale
    assert inverted.selected_signal_family == original.selected_signal_family
    assert inverted.selected_gdelt_multiplier == original.selected_gdelt_multiplier
    assert len(inverted_diagnostics) == len(original_diagnostics)


def test_selection_rejects_leaky_or_empty_designs() -> None:
    with pytest.raises(ValueError, match="validation_start"):
        select_and_evaluate(
            _cases(),
            validation_start=_month(2022, 1),
            validation_end=_month(2021, 1),
            holdout_start=_month(2022, 1),
        )

    with pytest.raises(ValueError, match="non-empty"):
        select_and_evaluate(
            _cases(),
            validation_start=_month(2020, 1),
            validation_end=_month(2022, 1),
            holdout_start=_month(2022, 1),
            top_k_candidates=(),
        )

    with pytest.raises(ValueError, match="cannot include 'none'"):
        select_and_evaluate(
            _cases(),
            validation_start=_month(2020, 1),
            validation_end=_month(2022, 1),
            holdout_start=_month(2022, 1),
            minimum_training_cases=5,
            top_k_candidates=(3,),
            numeric_scale_candidates=("range",),
            signal_family_candidates=("none",),
            gdelt_multiplier_candidates=(0.1,),
        )
