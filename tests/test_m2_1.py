from datetime import datetime, timedelta, timezone

import pytest

from fencha.m2_1 import compare_matched_architecture, summarize_diagnostics
from fencha.models import FeatureValue, HistoricalCase

UTC = timezone.utc


def _month(year: int, month: int) -> datetime:
    return datetime(year, month, 1, tzinfo=UTC)


def _cases() -> list[HistoricalCase]:
    cases: list[HistoricalCase] = []
    for index in range(30):
        year = 2020 + index // 12
        month = index % 12 + 1
        cutoff = _month(year, month)
        outcome = index % 5 == 0
        features = {
            "country_code": FeatureValue("GBR", cutoff, "categorical"),
            "tenure_days": FeatureValue(float(index * 30), cutoff, "numeric"),
            "cabinet_age_days": FeatureValue(float(index * 20), cutoff, "numeric"),
            # Sample counts are diagnostics only and must never receive the
            # engine's implicit default weight.
            "gdelt_samples_30d": FeatureValue(float(2 + index % 4), cutoff, "numeric"),
            "gdelt_samples_90d": FeatureValue(float(7 + index % 6), cutoff, "numeric"),
            "gdelt_events_per_sample_30d": FeatureValue(
                float(20 + index * 3), cutoff, "numeric"
            ),
            "gdelt_articles_per_sample_30d": FeatureValue(
                float(30 + index * 4), cutoff, "numeric"
            ),
            "gdelt_protest_share_30d": FeatureValue(
                0.4 if outcome else 0.05, cutoff, "numeric"
            ),
            "gdelt_protest_share_90d": FeatureValue(
                0.3 if outcome else 0.04, cutoff, "numeric"
            ),
            "gdelt_avg_tone_30d": FeatureValue(
                -4.0 if outcome else -0.5, cutoff, "numeric"
            ),
            "gdelt_avg_goldstein_30d": FeatureValue(
                -3.0 if outcome else 1.0, cutoff, "numeric"
            ),
            "gdelt_coverage_30d": FeatureValue(1.0, cutoff, "numeric"),
            "gdelt_coverage_90d": FeatureValue(1.0, cutoff, "numeric"),
        }
        cases.append(
            HistoricalCase(
                case_id=f"case-{index}",
                domain="government_leader_exit_180d",
                question="Will the leader leave?",
                cutoff_at=cutoff,
                resolved_at=cutoff + timedelta(days=25),
                outcome=outcome,
                features=features,
                tags=("GBR", "test"),
            )
        )
    return cases


def test_zero_gdelt_multiplier_reproduces_structure_model() -> None:
    report, diagnostics = compare_matched_architecture(
        _cases(),
        holdout_start=_month(2021, 1),
        minimum_training_cases=5,
        max_history=100,
        top_k=7,
        gdelt_multiplier=0.0,
    )

    assert report.predictions == len(report.target_ids) == len(diagnostics)
    assert report.structure_analog.predictions == report.gdelt_analog.predictions
    assert report.structure_analog.brier_score == pytest.approx(
        report.gdelt_analog.brier_score
    )
    assert all(
        item.structure_probability == pytest.approx(item.gdelt_probability)
        for item in diagnostics
    )
    assert all(len(item.structure_neighbor_ids) <= 7 for item in diagnostics)
    assert all(len(item.gdelt_neighbor_ids) <= 7 for item in diagnostics)


def test_none_signal_family_is_an_exact_structure_control() -> None:
    report, diagnostics = compare_matched_architecture(
        _cases(),
        holdout_start=_month(2021, 1),
        minimum_training_cases=5,
        max_history=100,
        top_k=6,
        gdelt_multiplier=1.0,
        signal_family="none",
        numeric_scale="iqr",
    )

    assert report.signal_family == "none"
    assert report.numeric_scale == "iqr"
    assert report.gdelt_brier_skill_vs_structure == pytest.approx(0.0)
    assert all(item.probability_delta == pytest.approx(0.0) for item in diagnostics)
    assert all(item.neighbor_overlap == pytest.approx(1.0) for item in diagnostics)


def test_matched_comparison_records_exact_target_ids_and_error_deltas() -> None:
    report, diagnostics = compare_matched_architecture(
        _cases(),
        holdout_start=_month(2021, 1),
        minimum_training_cases=5,
        max_history=100,
        top_k=5,
        gdelt_multiplier=1.0,
        signal_family="conflict",
    )

    assert report.target_ids == tuple(item.case_id for item in diagnostics)
    assert report.first_cutoff == diagnostics[0].cutoff_at
    assert report.last_cutoff == diagnostics[-1].cutoff_at
    assert 0.0 <= report.mean_neighbor_overlap <= 1.0
    assert report.mean_squared_error_delta == pytest.approx(
        sum(item.squared_error_delta for item in diagnostics) / len(diagnostics)
    )
    for item in diagnostics:
        assert item.squared_error_delta == pytest.approx(
            item.gdelt_squared_error - item.structure_squared_error
        )


def test_signal_family_and_scale_validation() -> None:
    with pytest.raises(ValueError, match="signal family"):
        compare_matched_architecture(
            _cases(),
            holdout_start=_month(2021, 1),
            minimum_training_cases=5,
            signal_family="mystery",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="numeric_scale"):
        compare_matched_architecture(
            _cases(),
            holdout_start=_month(2021, 1),
            minimum_training_cases=5,
            numeric_scale="standard",  # type: ignore[arg-type]
        )


def test_subgroup_summary_reports_year_and_country() -> None:
    _, diagnostics = compare_matched_architecture(
        _cases(),
        holdout_start=_month(2021, 1),
        minimum_training_cases=5,
        max_history=100,
        signal_family="tone",
    )
    summary = summarize_diagnostics(diagnostics, minimum_group_predictions=2)

    assert [item["group"] for item in summary["by_country"]] == ["GBR"]
    assert summary["by_country"][0]["predictions"] == len(diagnostics)
    assert {item["group"] for item in summary["by_year"]} == {"2021", "2022"}

    with pytest.raises(ValueError, match="minimum_group_predictions"):
        summarize_diagnostics(diagnostics, minimum_group_predictions=0)
