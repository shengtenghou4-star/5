from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

from .engine import NumericScale
from .m2_1 import (
    GdeltSignalFamily,
    M21MatchedReport,
    M21PredictionDiagnostic,
    compare_matched_architecture,
)
from .m2_1_features import M21_FEATURE_VERSION, add_time_safe_volume_features
from .models import HistoricalCase, ensure_aware


@dataclass(frozen=True, slots=True)
class M21ValidationCandidate:
    stage: str
    top_k: int
    numeric_scale: NumericScale
    signal_family: GdeltSignalFamily
    gdelt_multiplier: float
    predictions: int
    structure_brier: float
    gdelt_brier: float
    gdelt_brier_skill_vs_structure: float
    structure_log_loss: float
    gdelt_log_loss: float
    structure_calibration_error: float
    gdelt_calibration_error: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class M21SelectionReport:
    feature_version: str
    validation_start: str
    validation_end: str
    holdout_start: str
    minimum_training_cases: int
    validation_target_stride: int
    holdout_target_stride: int
    max_history: int | None
    selected_top_k: int
    selected_numeric_scale: NumericScale
    selected_signal_family: GdeltSignalFamily
    selected_gdelt_multiplier: float
    structure_candidates: tuple[M21ValidationCandidate, ...]
    gdelt_candidates: tuple[M21ValidationCandidate, ...]
    selected_validation_candidate: M21ValidationCandidate
    final_holdout: M21MatchedReport

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _candidate(
    stage: str,
    report: M21MatchedReport,
) -> M21ValidationCandidate:
    return M21ValidationCandidate(
        stage=stage,
        top_k=report.top_k,
        numeric_scale=report.numeric_scale,
        signal_family=report.signal_family,
        gdelt_multiplier=report.gdelt_multiplier,
        predictions=report.predictions,
        structure_brier=report.structure_analog.brier_score,
        gdelt_brier=report.gdelt_analog.brier_score,
        gdelt_brier_skill_vs_structure=report.gdelt_brier_skill_vs_structure,
        structure_log_loss=report.structure_analog.log_loss,
        gdelt_log_loss=report.gdelt_analog.log_loss,
        structure_calibration_error=report.structure_analog.calibration_error,
        gdelt_calibration_error=report.gdelt_analog.calibration_error,
    )


def _structure_key(candidate: M21ValidationCandidate) -> tuple[float, float, float, int, str]:
    return (
        candidate.structure_brier,
        candidate.structure_calibration_error,
        candidate.structure_log_loss,
        candidate.top_k,
        candidate.numeric_scale,
    )


def _gdelt_key(
    candidate: M21ValidationCandidate,
) -> tuple[float, float, float, float, str]:
    return (
        candidate.gdelt_brier,
        candidate.gdelt_calibration_error,
        candidate.gdelt_log_loss,
        candidate.gdelt_multiplier,
        candidate.signal_family,
    )


def select_and_evaluate(
    cases: list[HistoricalCase],
    *,
    validation_start: datetime,
    validation_end: datetime,
    holdout_start: datetime,
    minimum_training_cases: int = 150,
    validation_target_stride: int = 2,
    holdout_target_stride: int = 1,
    max_history: int | None = 3000,
    top_k_candidates: Iterable[int] = (25, 50, 75),
    numeric_scale_candidates: Iterable[NumericScale] = ("range", "iqr"),
    signal_family_candidates: Iterable[GdeltSignalFamily] = (
        "raw_volume",
        "log_volume",
        "anomaly",
        "conflict",
        "tone",
        "all",
    ),
    gdelt_multiplier_candidates: Iterable[float] = (0.1, 0.25, 0.5, 1.0),
    prior_strength: float = 8.0,
    minimum_similarity: float = 0.05,
) -> tuple[M21SelectionReport, list[M21PredictionDiagnostic]]:
    """Select M2.1 settings before the holdout, then evaluate it exactly once.

    Selection sees only cases with cutoffs strictly before ``validation_end``.
    The 2022+ holdout is not used to choose any hyperparameter or signal family.
    Log-volume and country-anomaly features use only earlier same-country cutoffs.
    """
    validation_start = ensure_aware(validation_start)
    validation_end = ensure_aware(validation_end)
    holdout_start = ensure_aware(holdout_start)
    if not validation_start < validation_end <= holdout_start:
        raise ValueError(
            "dates must satisfy validation_start < validation_end <= holdout_start"
        )

    top_ks = tuple(top_k_candidates)
    scales = tuple(numeric_scale_candidates)
    families = tuple(signal_family_candidates)
    multipliers = tuple(gdelt_multiplier_candidates)
    if not top_ks or not scales or not families or not multipliers:
        raise ValueError("all candidate collections must be non-empty")
    if any(value <= 0 for value in top_ks):
        raise ValueError("top_k candidates must be positive")
    if any(value <= 0 for value in multipliers):
        raise ValueError("GDELT multiplier candidates must be positive")

    enhanced_cases = add_time_safe_volume_features(cases)
    validation_cases = [
        case
        for case in enhanced_cases
        if ensure_aware(case.cutoff_at) < validation_end
    ]
    if not validation_cases:
        raise ValueError("no cases exist before validation_end")

    structure_candidates: list[M21ValidationCandidate] = []
    structure_target_ids: tuple[str, ...] | None = None
    for top_k in top_ks:
        for numeric_scale in scales:
            report, _ = compare_matched_architecture(
                validation_cases,
                holdout_start=validation_start,
                minimum_training_cases=minimum_training_cases,
                target_stride=validation_target_stride,
                max_history=max_history,
                top_k=top_k,
                prior_strength=prior_strength,
                minimum_similarity=minimum_similarity,
                gdelt_multiplier=0.0,
                signal_family="none",
                numeric_scale=numeric_scale,
            )
            if structure_target_ids is None:
                structure_target_ids = report.target_ids
            elif structure_target_ids != report.target_ids:
                raise RuntimeError("structure candidates used different validation targets")
            structure_candidates.append(_candidate("structure", report))

    selected_structure = min(structure_candidates, key=_structure_key)

    gdelt_candidates: list[M21ValidationCandidate] = []
    for signal_family in families:
        if signal_family == "none":
            raise ValueError("GDELT signal-family candidates cannot include 'none'")
        for multiplier in multipliers:
            report, _ = compare_matched_architecture(
                validation_cases,
                holdout_start=validation_start,
                minimum_training_cases=minimum_training_cases,
                target_stride=validation_target_stride,
                max_history=max_history,
                top_k=selected_structure.top_k,
                prior_strength=prior_strength,
                minimum_similarity=minimum_similarity,
                gdelt_multiplier=multiplier,
                signal_family=signal_family,
                numeric_scale=selected_structure.numeric_scale,
            )
            if report.target_ids != structure_target_ids:
                raise RuntimeError("GDELT candidate used different validation targets")
            gdelt_candidates.append(_candidate("gdelt", report))

    selected_gdelt = min(gdelt_candidates, key=_gdelt_key)

    final_report, final_diagnostics = compare_matched_architecture(
        enhanced_cases,
        holdout_start=holdout_start,
        minimum_training_cases=minimum_training_cases,
        target_stride=holdout_target_stride,
        max_history=max_history,
        top_k=selected_structure.top_k,
        prior_strength=prior_strength,
        minimum_similarity=minimum_similarity,
        gdelt_multiplier=selected_gdelt.gdelt_multiplier,
        signal_family=selected_gdelt.signal_family,
        numeric_scale=selected_structure.numeric_scale,
    )

    return (
        M21SelectionReport(
            feature_version=M21_FEATURE_VERSION,
            validation_start=validation_start.isoformat(),
            validation_end=validation_end.isoformat(),
            holdout_start=holdout_start.isoformat(),
            minimum_training_cases=minimum_training_cases,
            validation_target_stride=validation_target_stride,
            holdout_target_stride=holdout_target_stride,
            max_history=max_history,
            selected_top_k=selected_structure.top_k,
            selected_numeric_scale=selected_structure.numeric_scale,
            selected_signal_family=selected_gdelt.signal_family,
            selected_gdelt_multiplier=selected_gdelt.gdelt_multiplier,
            structure_candidates=tuple(structure_candidates),
            gdelt_candidates=tuple(gdelt_candidates),
            selected_validation_candidate=selected_gdelt,
            final_holdout=final_report,
        ),
        final_diagnostics,
    )
