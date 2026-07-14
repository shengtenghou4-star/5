from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable

from .hierarchical import HierarchicalRiskConfig, HierarchicalRiskForecaster
from .models import HistoricalCase, ensure_aware
from .survival_benchmark import SurvivalBenchmarkReport, temporal_survival_benchmark


@dataclass(frozen=True, slots=True)
class M31CandidateResult:
    template: str
    config: HierarchicalRiskConfig
    snapshots: int
    adjusted_integrated_brier: float
    raw_integrated_brier: float
    baseline_integrated_brier: float
    adjusted_skill_vs_baseline: float
    raw_crossing_rate: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class M31SelectionReport:
    validation_start: str
    validation_end: str
    holdout_start: str
    horizons: tuple[int, ...]
    minimum_training_cases: int
    validation_target_stride: int
    holdout_target_stride: int
    max_history: int | None
    candidates: tuple[M31CandidateResult, ...]
    selected_template: str
    selected_config: HierarchicalRiskConfig
    selected_validation: M31CandidateResult
    final_holdout: SurvivalBenchmarkReport

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_candidate_configs() -> tuple[tuple[str, HierarchicalRiskConfig], ...]:
    """Predeclared transparent hierarchy grid, from simplest to richest."""
    candidates: list[tuple[str, HierarchicalRiskConfig]] = [
        (
            "global",
            HierarchicalRiskConfig(
                use_country=False,
                use_tenure=False,
                use_context=False,
            ),
        )
    ]
    strengths = (25.0, 100.0, 400.0)
    tenure_buckets = (180, 365, 730)
    for strength in strengths:
        candidates.append(
            (
                "country",
                HierarchicalRiskConfig(
                    use_country=True,
                    use_tenure=False,
                    use_context=False,
                    country_strength=strength,
                ),
            )
        )
    for bucket in tenure_buckets:
        for strength in strengths:
            candidates.append(
                (
                    "tenure",
                    HierarchicalRiskConfig(
                        use_country=False,
                        use_tenure=True,
                        use_context=False,
                        tenure_bucket_days=bucket,
                        tenure_strength=strength,
                    ),
                )
            )
            candidates.append(
                (
                    "country_tenure",
                    HierarchicalRiskConfig(
                        use_country=True,
                        use_tenure=True,
                        use_context=False,
                        tenure_bucket_days=bucket,
                        country_strength=strength,
                        tenure_strength=strength,
                    ),
                )
            )
            candidates.append(
                (
                    "country_tenure_context",
                    HierarchicalRiskConfig(
                        use_country=True,
                        use_tenure=True,
                        use_context=True,
                        tenure_bucket_days=bucket,
                        country_strength=strength,
                        tenure_strength=strength,
                        context_strength=strength,
                    ),
                )
            )
    return tuple(candidates)


def _candidate_result(
    template: str,
    config: HierarchicalRiskConfig,
    report: SurvivalBenchmarkReport,
) -> M31CandidateResult:
    return M31CandidateResult(
        template=template,
        config=config,
        snapshots=report.snapshots,
        adjusted_integrated_brier=report.adjusted_integrated_brier,
        raw_integrated_brier=report.raw_integrated_brier,
        baseline_integrated_brier=report.baseline_integrated_brier,
        adjusted_skill_vs_baseline=report.adjusted_integrated_brier_skill_vs_baseline,
        raw_crossing_rate=report.raw_crossing_rate,
    )


def _complexity(candidate: M31CandidateResult) -> int:
    config = candidate.config
    return int(config.use_country) + int(config.use_tenure) + int(config.use_context)


def _selection_key(
    candidate: M31CandidateResult,
) -> tuple[float, float, int, float, int]:
    # Proper score is authoritative. Ties prefer less raw distortion, a simpler
    # hierarchy, stronger shrinkage, and a wider tenure bucket.
    config = candidate.config
    active_strengths = []
    if config.use_country:
        active_strengths.append(config.country_strength)
    if config.use_tenure:
        active_strengths.append(config.tenure_strength)
    if config.use_context:
        active_strengths.append(config.context_strength)
    mean_strength = (
        sum(active_strengths) / len(active_strengths)
        if active_strengths
        else float("inf")
    )
    return (
        candidate.adjusted_integrated_brier,
        candidate.raw_integrated_brier,
        _complexity(candidate),
        -mean_strength,
        -config.tenure_bucket_days,
    )


def select_hierarchical_survival_model(
    cases: list[HistoricalCase],
    *,
    validation_start: datetime,
    validation_end: datetime,
    holdout_start: datetime,
    horizons: Iterable[int] = (30, 90, 180, 365),
    minimum_training_cases: int = 500,
    validation_target_stride: int = 6,
    holdout_target_stride: int = 3,
    max_history: int | None = 3000,
    candidates: Iterable[tuple[str, HierarchicalRiskConfig]] | None = None,
) -> M31SelectionReport:
    """Select shrinkage settings before 2015, then evaluate the holdout once."""
    validation_start = ensure_aware(validation_start)
    validation_end = ensure_aware(validation_end)
    holdout_start = ensure_aware(holdout_start)
    if not validation_start < validation_end <= holdout_start:
        raise ValueError(
            "dates must satisfy validation_start < validation_end <= holdout_start"
        )
    normalized_horizons = tuple(sorted(set(horizons)))
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError("horizons must contain positive values")

    candidate_configs = (
        default_candidate_configs() if candidates is None else tuple(candidates)
    )
    if not candidate_configs:
        raise ValueError("at least one hierarchy candidate is required")
    if len({(name, config) for name, config in candidate_configs}) != len(
        candidate_configs
    ):
        raise ValueError("hierarchy candidates must be unique")

    validation_cases = [
        case for case in cases if ensure_aware(case.cutoff_at) < validation_end
    ]
    if not validation_cases:
        raise ValueError("no cases exist before validation_end")

    results: list[M31CandidateResult] = []
    target_ids: tuple[str, ...] | None = None
    for template, config in candidate_configs:
        report = temporal_survival_benchmark(
            validation_cases,
            HierarchicalRiskForecaster(config),  # type: ignore[arg-type]
            holdout_start=validation_start,
            horizons=normalized_horizons,
            minimum_training_cases=minimum_training_cases,
            target_stride=validation_target_stride,
            max_history=max_history,
        )
        if target_ids is None:
            target_ids = report.snapshot_ids
        elif report.snapshot_ids != target_ids:
            raise RuntimeError("hierarchy candidates used different validation targets")
        results.append(_candidate_result(template, config, report))

    selected = min(results, key=_selection_key)
    final_report = temporal_survival_benchmark(
        cases,
        HierarchicalRiskForecaster(selected.config),  # type: ignore[arg-type]
        holdout_start=holdout_start,
        horizons=normalized_horizons,
        minimum_training_cases=minimum_training_cases,
        target_stride=holdout_target_stride,
        max_history=max_history,
    )

    return M31SelectionReport(
        validation_start=validation_start.isoformat(),
        validation_end=validation_end.isoformat(),
        holdout_start=holdout_start.isoformat(),
        horizons=normalized_horizons,
        minimum_training_cases=minimum_training_cases,
        validation_target_stride=validation_target_stride,
        holdout_target_stride=holdout_target_stride,
        max_history=max_history,
        candidates=tuple(results),
        selected_template=selected.template,
        selected_config=selected.config,
        selected_validation=selected,
        final_holdout=final_report,
    )
