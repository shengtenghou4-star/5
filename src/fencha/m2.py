from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from .benchmark import MetricSet, temporal_holdout_benchmark
from .engine import AnalogForecaster
from .models import HistoricalCase

STRUCTURE_WEIGHTS = {
    "country_code": 0.25,
    "tenure_days": 2.0,
    "cabinet_age_days": 0.75,
    "coalition_size": 0.5,
    "caretaker": 1.5,
    "cabinet_type": 0.75,
    "election_age_days": 1.25,
    "government_seat_share": 1.25,
    "minority_government": 1.0,
}

GDELT_WEIGHTS = {
    "gdelt_events_per_sample_30d": 0.35,
    "gdelt_articles_per_sample_30d": 0.35,
    "gdelt_protest_share_30d": 1.5,
    "gdelt_verbal_conflict_share_30d": 0.75,
    "gdelt_material_conflict_share_30d": 1.5,
    "gdelt_cooperation_share_30d": 0.75,
    "gdelt_negative_tone_share_30d": 1.0,
    "gdelt_avg_tone_30d": 1.0,
    "gdelt_avg_goldstein_30d": 1.0,
    "gdelt_events_per_sample_90d": 0.25,
    "gdelt_articles_per_sample_90d": 0.25,
    "gdelt_protest_share_90d": 1.0,
    "gdelt_verbal_conflict_share_90d": 0.5,
    "gdelt_material_conflict_share_90d": 1.0,
    "gdelt_cooperation_share_90d": 0.5,
    "gdelt_negative_tone_share_90d": 0.75,
    "gdelt_avg_tone_90d": 0.75,
    "gdelt_avg_goldstein_90d": 0.75,
    "gdelt_coverage_30d": 0.1,
    "gdelt_coverage_90d": 0.1,
}


@dataclass(frozen=True, slots=True)
class M2ComparisonReport:
    holdout_start: str
    predictions: int
    baseline: MetricSet
    structure_analog: MetricSet
    gdelt_analog: MetricSet
    structure_brier_skill_vs_baseline: float
    gdelt_brier_skill_vs_baseline: float
    gdelt_brier_skill_vs_structure: float
    gdelt_log_loss_skill_vs_structure: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _without_gdelt(case: HistoricalCase) -> HistoricalCase:
    return HistoricalCase(
        case_id=case.case_id,
        domain=case.domain,
        question=case.question,
        cutoff_at=case.cutoff_at,
        resolved_at=case.resolved_at,
        outcome=case.outcome,
        features={
            name: feature
            for name, feature in case.features.items()
            if not name.startswith("gdelt_")
        },
        tags=case.tags,
    )


def compare_structure_and_gdelt(
    cases: list[HistoricalCase],
    *,
    holdout_start: datetime,
    minimum_training_cases: int = 500,
    target_stride: int = 1,
    max_history: int | None = 3000,
) -> M2ComparisonReport:
    enriched = [
        case
        for case in cases
        if any(name.startswith("gdelt_") for name in case.features)
    ]
    if not enriched:
        raise ValueError("no GDELT-enriched cases were supplied")
    structure_cases = [_without_gdelt(case) for case in enriched]

    structure_model = AnalogForecaster(
        feature_weights=STRUCTURE_WEIGHTS,
        prior_strength=8.0,
        top_k=50,
    )
    gdelt_model = AnalogForecaster(
        feature_weights={**STRUCTURE_WEIGHTS, **GDELT_WEIGHTS},
        prior_strength=8.0,
        top_k=75,
    )
    common = dict(
        holdout_start=holdout_start,
        minimum_training_cases=minimum_training_cases,
        target_stride=target_stride,
        max_history=max_history,
    )
    structure = temporal_holdout_benchmark(
        structure_cases, structure_model, **common
    )
    gdelt = temporal_holdout_benchmark(enriched, gdelt_model, **common)
    if structure.analog.predictions != gdelt.analog.predictions:
        raise RuntimeError("structure and GDELT benchmarks used different targets")

    def skill(score: float, reference: float) -> float:
        return 0.0 if reference == 0 else 1.0 - score / reference

    return M2ComparisonReport(
        holdout_start=holdout_start.isoformat(),
        predictions=gdelt.analog.predictions,
        baseline=gdelt.baseline,
        structure_analog=structure.analog,
        gdelt_analog=gdelt.analog,
        structure_brier_skill_vs_baseline=structure.analog_brier_skill,
        gdelt_brier_skill_vs_baseline=gdelt.analog_brier_skill,
        gdelt_brier_skill_vs_structure=skill(
            gdelt.analog.brier_score, structure.analog.brier_score
        ),
        gdelt_log_loss_skill_vs_structure=skill(
            gdelt.analog.log_loss, structure.analog.log_loss
        ),
    )
