from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import fmean
from typing import Literal

from .benchmark import MetricSet
from .engine import AnalogForecaster, NumericScale
from .m2 import GDELT_WEIGHTS, STRUCTURE_WEIGHTS
from .models import HistoricalCase, ensure_aware
from .scoring import BacktestPoint, binary_log_loss, brier_score, calibration_error

GdeltSignalFamily = Literal[
    "none",
    "volume",
    "raw_volume",
    "log_volume",
    "anomaly",
    "conflict",
    "tone",
    "all",
]

# M2.1-specific additions. The frozen M2 weights remain unchanged in m2.py.
M21_GDELT_WEIGHTS = {
    **GDELT_WEIGHTS,
    "gdelt_log_events_per_sample_30d": 0.35,
    "gdelt_log_articles_per_sample_30d": 0.35,
    "gdelt_log_events_per_sample_90d": 0.25,
    "gdelt_log_articles_per_sample_90d": 0.25,
    "gdelt_anomaly_log_events_per_sample_30d": 0.75,
    "gdelt_anomaly_log_articles_per_sample_30d": 0.75,
    "gdelt_anomaly_log_events_per_sample_90d": 0.50,
    "gdelt_anomaly_log_articles_per_sample_90d": 0.50,
}

_RAW_VOLUME_FEATURES = {
    name
    for name in GDELT_WEIGHTS
    if "events_per_sample" in name or "articles_per_sample" in name
}
_LOG_VOLUME_FEATURES = {
    name
    for name in M21_GDELT_WEIGHTS
    if name.startswith("gdelt_log_")
}
_ANOMALY_FEATURES = {
    name
    for name in M21_GDELT_WEIGHTS
    if name.startswith("gdelt_anomaly_")
}
_CONFLICT_FEATURES = {
    name
    for name in GDELT_WEIGHTS
    if any(
        token in name
        for token in (
            "protest_share",
            "verbal_conflict_share",
            "material_conflict_share",
            "cooperation_share",
        )
    )
}
_TONE_FEATURES = {
    name
    for name in GDELT_WEIGHTS
    if any(
        token in name
        for token in ("negative_tone_share", "avg_tone", "avg_goldstein")
    )
}
_COVERAGE_FEATURES = {name for name in GDELT_WEIGHTS if "coverage" in name}


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
    structure_squared_error: float
    gdelt_squared_error: float
    squared_error_delta: float
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
class M21SubgroupReport:
    group: str
    predictions: int
    positives: int
    structure_brier: float
    gdelt_brier: float
    gdelt_brier_skill_vs_structure: float
    mean_absolute_probability_delta: float
    mean_neighbor_overlap: float

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
    signal_family: GdeltSignalFamily
    numeric_scale: NumericScale
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
    mean_squared_error_delta: float
    mean_neighbor_overlap: float
    mean_structure_effective_sample_size: float
    mean_gdelt_effective_sample_size: float
    mean_structure_feature_coverage: float
    mean_gdelt_feature_coverage: float
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


def _family_features(signal_family: GdeltSignalFamily) -> set[str]:
    if signal_family == "none":
        return set()
    if signal_family in {"volume", "raw_volume"}:
        return _RAW_VOLUME_FEATURES | _COVERAGE_FEATURES
    if signal_family == "log_volume":
        return _LOG_VOLUME_FEATURES | _COVERAGE_FEATURES
    if signal_family == "anomaly":
        return _ANOMALY_FEATURES | _COVERAGE_FEATURES
    if signal_family == "conflict":
        return _CONFLICT_FEATURES | _COVERAGE_FEATURES
    if signal_family == "tone":
        return _TONE_FEATURES | _COVERAGE_FEATURES
    if signal_family == "all":
        # Raw and log volumes encode the same quantity. The combined model uses
        # the stabilized log representation plus country anomalies, not both.
        return (
            _LOG_VOLUME_FEATURES
            | _ANOMALY_FEATURES
            | _CONFLICT_FEATURES
            | _TONE_FEATURES
            | _COVERAGE_FEATURES
        )
    raise ValueError(f"unknown GDELT signal family: {signal_family}")


def _with_signal_family(
    case: HistoricalCase,
    signal_family: GdeltSignalFamily,
) -> HistoricalCase:
    selected = _family_features(signal_family)
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
            if not name.startswith("gdelt_") or name in selected
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


def _subgroup_report(
    group: str,
    items: list[M21PredictionDiagnostic],
) -> M21SubgroupReport:
    structure_brier = fmean(item.structure_squared_error for item in items)
    gdelt_brier = fmean(item.gdelt_squared_error for item in items)
    return M21SubgroupReport(
        group=group,
        predictions=len(items),
        positives=sum(item.outcome for item in items),
        structure_brier=structure_brier,
        gdelt_brier=gdelt_brier,
        gdelt_brier_skill_vs_structure=_skill(gdelt_brier, structure_brier),
        mean_absolute_probability_delta=fmean(
            abs(item.probability_delta) for item in items
        ),
        mean_neighbor_overlap=fmean(item.neighbor_overlap for item in items),
    )


def summarize_diagnostics(
    diagnostics: list[M21PredictionDiagnostic],
    *,
    minimum_group_predictions: int = 10,
) -> dict[str, list[dict[str, object]]]:
    if minimum_group_predictions <= 0:
        raise ValueError("minimum_group_predictions must be positive")

    countries: dict[str, list[M21PredictionDiagnostic]] = {}
    years: dict[str, list[M21PredictionDiagnostic]] = {}
    for item in diagnostics:
        countries.setdefault(item.country, []).append(item)
        years.setdefault(item.cutoff_at[:4], []).append(item)

    def build(groups: dict[str, list[M21PredictionDiagnostic]]) -> list[dict[str, object]]:
        return [
            _subgroup_report(group, items).to_dict()
            for group, items in sorted(groups.items())
            if len(items) >= minimum_group_predictions
        ]

    return {
        "by_country": build(countries),
        "by_year": build(years),
    }


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
    signal_family: GdeltSignalFamily = "all",
    numeric_scale: NumericScale = "range",
) -> tuple[M21MatchedReport, list[M21PredictionDiagnostic]]:
    """Compare structure and GDELT using identical targets and hyperparameters.

    This diagnostic removes the original pilot's top-k confound. Signal-family
    filtering prevents omitted GDELT fields from receiving the engine's implicit
    default weight, while both models use the same numeric-scaling rule.
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
    selected_features = _family_features(signal_family)

    raw_enriched = [
        case
        for case in cases
        if any(name.startswith("gdelt_") for name in case.features)
    ]
    if not raw_enriched:
        raise ValueError("no GDELT-enriched cases were supplied")

    enriched = [_with_signal_family(case, signal_family) for case in raw_enriched]
    structure_by_id = {case.case_id: _without_gdelt(case) for case in raw_enriched}
    if len(structure_by_id) != len(raw_enriched):
        raise ValueError("case_id values must be unique")

    gdelt_weights = {
        name: weight * gdelt_multiplier
        for name, weight in M21_GDELT_WEIGHTS.items()
        if name in selected_features
    }
    shared_model_args = {
        "prior_strength": prior_strength,
        "top_k": top_k,
        "minimum_similarity": minimum_similarity,
        "numeric_scale": numeric_scale,
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
        outcome_value = 1.0 if target.outcome else 0.0
        structure_squared_error = (structure_result.probability - outcome_value) ** 2
        gdelt_squared_error = (gdelt_result.probability - outcome_value) ** 2
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
                structure_squared_error=structure_squared_error,
                gdelt_squared_error=gdelt_squared_error,
                squared_error_delta=gdelt_squared_error - structure_squared_error,
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
            signal_family=signal_family,
            numeric_scale=numeric_scale,
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
            mean_squared_error_delta=fmean(
                item.squared_error_delta for item in diagnostics
            ),
            mean_neighbor_overlap=fmean(
                item.neighbor_overlap for item in diagnostics
            ),
            mean_structure_effective_sample_size=fmean(
                item.structure_effective_sample_size for item in diagnostics
            ),
            mean_gdelt_effective_sample_size=fmean(
                item.gdelt_effective_sample_size for item in diagnostics
            ),
            mean_structure_feature_coverage=fmean(
                item.structure_feature_coverage for item in diagnostics
            ),
            mean_gdelt_feature_coverage=fmean(
                item.gdelt_feature_coverage for item in diagnostics
            ),
            first_cutoff=diagnostics[0].cutoff_at,
            last_cutoff=diagnostics[-1].cutoff_at,
            target_ids=target_ids,
        ),
        diagnostics,
    )
