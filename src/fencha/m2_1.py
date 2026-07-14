from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import fmean

from .benchmark import MetricSet
from .engine import AnalogForecaster
from .m2 import GDELT_WEIGHTS, STRUCTURE_WEIGHTS
from .models import HistoricalCase, ensure_aware
from .scoring import BacktestPoint, binary_log_loss, brier_score, calibration_error


@dataclass(frozen=True, slots=True)
class M21PredictionDiagnostic:
    case_id: str
    cutoff_at: str
    country: str
    outcome: bool
    baseline_probability: float
    structure_probability: float
    gdelt_probability: float
    probability_delta: float
    structure_effective_sample_size: float
    gdelt_effective_sample_size: float
    structure_feature_coverage: float
    gdelt_feature_coverage: float
    neighbor_overlap: float
    structure_neighbor_ids: tuple[str, ...]
    gdelt_neighbor_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class M21MatchedReport:
    holdout_start: str
    predictions: int
    positives: int
    top_k: int
    prior_strength: float
    minimum_similarity: float
    gdelt_multiplier: float
    target_stride: int
    max_history: int | None
    baseline: MetricSet
    structure_analog: MetricSet
    gdelt_analog: MetricSet
    structure_brier_skill_vs_baseline: float
    gdelt_brier_skill_vs_baseline: float
    gdelt_brier_skill_vs_structure: float
    gdelt_log_loss_skill_vs_structure: float
    mean_signed_probability_delta: float
    mean_absolute_probability_delta: float
    mean_neighbor_overlap: float
    first_cutoff: str
    last_cutoff: str
    target_ids: tuple[str, ...]

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


def _metrics(points: list[BacktestPoint]) -> MetricSet:
    return MetricSet(
        predictions=len(points),
        brier_score=fmean(
            brier_score(point.probability, point.outcome) for point in points
        ),
        log_loss=fmean(
            binary_log_loss(point.probability, point.outcome) for point in points
        ),
        calibration_error=calibration_error(points, bins=10),
    )


def _skill(score: float, reference: float) -> float:
    return 0.0 if reference == 0 else 1.0 - score / reference


def _neighbor_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    return 1.0 if not union else len(left_set & right_set) / len(union)


def compare_matched_architecture(
    cases: list[HistoricalCase],
    *,
    holdout_start: datetime,
    minimum_training_cases: int = 500,
    target_stride: int = 1,
    max_history: int | None = 3000,
    top_k: int = 50,
    prior_strength: float = 8.0,
    minimum_similarity: float = 0.05,
    gdelt_multiplier: float = 1.0,
) -> tuple[M21MatchedReport, list[M21PredictionDiagnostic]]:
    """Compare structure and GDELT using identical targets and hyperparameters.

    This is the first M2.1 diagnostic ablation. It removes the original pilot's
    top-k confound while preserving the frozen holdout and time-safety rules.
    """
    holdout_start = ensure_aware(holdout_start)
    if minimum_training_cases <= 0:
        raise ValueError("minimum_training_cases must be positive")
    if target_stride <= 0:
        raise ValueError("target_stride must be positive")
    if max_history is not None and max_history <= 0:
        raise ValueError("max_history must be positive when provided")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if prior_strength <= 0:
        raise ValueError("prior_strength must be positive")
    if minimum_similarity < 0:
        raise ValueError("minimum_similarity cannot be negative")
    if gdelt_multiplier < 0:
        raise ValueError("gdelt_multiplier cannot be negative")

    enriched = [
        case
        for case in cases
        if any(name.startswith("gdelt_") for name in case.features)
    ]
    if not enriched:
        raise ValueError("no GDELT-enriched cases were supplied")

    structure_by_id = {case.case_id: _without_gdelt(case) for case in enriched}
    if len(structure_by_id) != len(enriched):
        raise ValueError("case_id values must be unique")

    gdelt_weights = {
        name: weight * gdelt_multiplier for name, weight in GDELT_WEIGHTS.items()
    }
    shared_model_args = {
        "prior_strength": prior_strength,
        "top_k": top_k,
        "minimum_similarity": minimum_similarity,
    }
    structure_model = AnalogForecaster(
        feature_weights=STRUCTURE_WEIGHTS,
        **shared_model_args,
    )
    gdelt_model = AnalogForecaster(
        feature_weights={**STRUCTURE_WEIGHTS, **gdelt_weights},
        **shared_model_args,
    )

    ordered = sorted(enriched, key=lambda case: (case.cutoff_at, case.case_id))
    resolved = sorted(enriched, key=lambda case: (case.resolved_at, case.case_id))
    targets = [case for case in ordered if case.cutoff_at >= holdout_start]
    if not targets:
        raise ValueError("no cases exist in the requested holdout period")

    history: list[HistoricalCase] = []
    resolved_pointer = 0
    seen_by_country: dict[str, int] = {}
    baseline_points: list[BacktestPoint] = []
    structure_points: list[BacktestPoint] = []
    gdelt_points: list[BacktestPoint] = []
    diagnostics: list[M21PredictionDiagnostic] = []

    for target in targets:
        while (
            resolved_pointer < len(resolved)
            and resolved[resolved_pointer].resolved_at < target.cutoff_at
        ):
            history.append(resolved[resolved_pointer])
            resolved_pointer += 1

        country = target.tags[0] if target.tags else "all"
        seen = seen_by_country.get(country, 0)
        seen_by_country[country] = seen + 1
        if seen % target_stride:
            continue

        eligible_gdelt = [case for case in history if case.domain == target.domain]
        if len(eligible_gdelt) < minimum_training_cases:
            continue
        if max_history is not None:
            eligible_gdelt = eligible_gdelt[-max_history:]
        eligible_structure = [
            structure_by_id[case.case_id] for case in eligible_gdelt
        ]

        yes_count = sum(case.outcome for case in eligible_gdelt)
        baseline_probability = (yes_count + 1.0) / (len(eligible_gdelt) + 2.0)
        structure_target = structure_by_id[target.case_id]

        structure_result = structure_model.fit_predict(
            eligible_structure,
            target_features=structure_target.features,
            target_cutoff=target.cutoff_at,
            domain=target.domain,
        )
        gdelt_result = gdelt_model.fit_predict(
            eligible_gdelt,
            target_features=target.features,
            target_cutoff=target.cutoff_at,
            domain=target.domain,
        )

        baseline_points.append(
            BacktestPoint(
                case_id=target.case_id,
                probability=baseline_probability,
                outcome=target.outcome,
                training_cases=len(eligible_gdelt),
            )
        )
        structure_points.append(
            BacktestPoint(
                case_id=target.case_id,
                probability=structure_result.probability,
                outcome=target.outcome,
                training_cases=len(eligible_structure),
            )
        )
        gdelt_points.append(
            BacktestPoint(
                case_id=target.case_id,
                probability=gdelt_result.probability,
                outcome=target.outcome,
                training_cases=len(eligible_gdelt),
            )
        )

        structure_neighbor_ids = tuple(
            neighbor.case_id for neighbor in structure_result.neighbors
        )
        gdelt_neighbor_ids = tuple(
            neighbor.case_id for neighbor in gdelt_result.neighbors
        )
        diagnostics.append(
            M21PredictionDiagnostic(
                case_id=target.case_id,
                cutoff_at=target.cutoff_at.isoformat(),
                country=country,
                outcome=target.outcome,
                baseline_probability=baseline_probability,
                structure_probability=structure_result.probability,
                gdelt_probability=gdelt_result.probability,
                probability_delta=(
                    gdelt_result.probability - structure_result.probability
                ),
                structure_effective_sample_size=(
                    structure_result.effective_sample_size
                ),
                gdelt_effective_sample_size=gdelt_result.effective_sample_size,
                structure_feature_coverage=structure_result.feature_coverage,
                gdelt_feature_coverage=gdelt_result.feature_coverage,
                neighbor_overlap=_neighbor_overlap(
                    structure_neighbor_ids, gdelt_neighbor_ids
                ),
                structure_neighbor_ids=structure_neighbor_ids,
                gdelt_neighbor_ids=gdelt_neighbor_ids,
            )
        )

    if not diagnostics:
        raise ValueError("no holdout predictions were generated")

    target_ids = tuple(point.case_id for point in structure_points)
    if target_ids != tuple(point.case_id for point in gdelt_points):
        raise RuntimeError("structure and GDELT target IDs differ")
    if target_ids != tuple(point.case_id for point in baseline_points):
        raise RuntimeError("baseline and analog target IDs differ")

    baseline_metrics = _metrics(baseline_points)
    structure_metrics = _metrics(structure_points)
    gdelt_metrics = _metrics(gdelt_points)
    deltas = [item.probability_delta for item in diagnostics]

    return (
        M21MatchedReport(
            holdout_start=holdout_start.isoformat(),
            predictions=len(diagnostics),
            positives=sum(item.outcome for item in diagnostics),
            top_k=top_k,
            prior_strength=prior_strength,
            minimum_similarity=minimum_similarity,
            gdelt_multiplier=gdelt_multiplier,
            target_stride=target_stride,
            max_history=max_history,
            baseline=baseline_metrics,
            structure_analog=structure_metrics,
            gdelt_analog=gdelt_metrics,
            structure_brier_skill_vs_baseline=_skill(
                structure_metrics.brier_score, baseline_metrics.brier_score
            ),
            gdelt_brier_skill_vs_baseline=_skill(
                gdelt_metrics.brier_score, baseline_metrics.brier_score
            ),
            gdelt_brier_skill_vs_structure=_skill(
                gdelt_metrics.brier_score, structure_metrics.brier_score
            ),
            gdelt_log_loss_skill_vs_structure=_skill(
                gdelt_metrics.log_loss, structure_metrics.log_loss
            ),
            mean_signed_probability_delta=fmean(deltas),
            mean_absolute_probability_delta=fmean(abs(value) for value in deltas),
            mean_neighbor_overlap=fmean(
                item.neighbor_overlap for item in diagnostics
            ),
            first_cutoff=diagnostics[0].cutoff_at,
            last_cutoff=diagnostics[-1].cutoff_at,
            target_ids=target_ids,
        ),
        diagnostics,
    )
