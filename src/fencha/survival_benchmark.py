from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import fmean
from typing import Iterable

from .benchmark import MetricSet
from .engine import AnalogForecaster
from .models import HistoricalCase, ensure_aware
from .scoring import BacktestPoint, binary_log_loss, brier_score, calibration_error
from .survival import coherent_survival_curve


@dataclass(frozen=True, slots=True)
class HorizonSurvivalMetrics:
    horizon_days: int
    predictions: int
    positives: int
    baseline: MetricSet
    raw_analog: MetricSet
    adjusted_analog: MetricSet
    adjusted_brier_skill_vs_raw: float
    adjusted_brier_skill_vs_baseline: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SurvivalBenchmarkReport:
    holdout_start: str
    horizons: tuple[int, ...]
    snapshots: int
    target_stride: int
    minimum_training_cases: int
    max_history: int | None
    raw_crossing_curves: int
    raw_crossing_rate: float
    mean_crossing_magnitude: float
    mean_absolute_adjustment: float
    baseline_integrated_brier: float
    raw_integrated_brier: float
    adjusted_integrated_brier: float
    adjusted_integrated_brier_skill_vs_raw: float
    adjusted_integrated_brier_skill_vs_baseline: float
    mean_restricted_survival_days: float
    first_cutoff: str
    last_cutoff: str
    snapshot_ids: tuple[str, ...]
    horizon_metrics: tuple[HorizonSurvivalMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


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


def _horizon_from_domain(case: HistoricalCase, domain_prefix: str) -> int | None:
    prefix = f"{domain_prefix}_"
    if not case.domain.startswith(prefix) or not case.domain.endswith("d"):
        return None
    value = case.domain[len(prefix) : -1]
    try:
        horizon = int(value)
    except ValueError:
        return None
    return horizon if horizon > 0 else None


def _snapshot_id(case: HistoricalCase, horizon: int) -> str:
    suffix = f":{horizon}d"
    if case.case_id.endswith(suffix):
        return case.case_id[: -len(suffix)]
    return f"{case.tags[0] if case.tags else 'all'}:{case.cutoff_at.isoformat()}:{case.case_id}"


def temporal_survival_benchmark(
    cases: list[HistoricalCase],
    model: AnalogForecaster,
    *,
    holdout_start: datetime,
    horizons: Iterable[int] = (30, 90, 180, 365),
    domain_prefix: str = "government_leader_exit",
    minimum_training_cases: int = 100,
    target_stride: int = 3,
    max_history: int | None = 3000,
) -> SurvivalBenchmarkReport:
    """Walk forward through complete multi-horizon snapshots without leakage."""
    holdout_start = ensure_aware(holdout_start)
    normalized_horizons = tuple(sorted(set(horizons)))
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError("horizons must contain at least one positive value")
    if minimum_training_cases <= 0:
        raise ValueError("minimum_training_cases must be positive")
    if target_stride <= 0:
        raise ValueError("target_stride must be positive")
    if max_history is not None and max_history <= 0:
        raise ValueError("max_history must be positive when provided")

    requested = set(normalized_horizons)
    snapshots: dict[str, dict[int, HistoricalCase]] = {}
    relevant: list[HistoricalCase] = []
    for case in cases:
        horizon = _horizon_from_domain(case, domain_prefix)
        if horizon not in requested:
            continue
        relevant.append(case)
        snapshot_id = _snapshot_id(case, horizon)
        bucket = snapshots.setdefault(snapshot_id, {})
        if horizon in bucket:
            raise ValueError(f"duplicate horizon {horizon} for snapshot {snapshot_id}")
        bucket[horizon] = case

    complete_targets: list[tuple[str, dict[int, HistoricalCase]]] = []
    for snapshot_id, bucket in snapshots.items():
        if set(bucket) != requested:
            continue
        cutoffs = {case.cutoff_at for case in bucket.values()}
        if len(cutoffs) != 1:
            raise ValueError(f"snapshot {snapshot_id} contains different cutoffs")
        if next(iter(cutoffs)) >= holdout_start:
            complete_targets.append((snapshot_id, bucket))
    complete_targets.sort(
        key=lambda item: (
            item[1][normalized_horizons[0]].cutoff_at,
            item[0],
        )
    )
    if not complete_targets:
        raise ValueError("no complete survival snapshots exist in the holdout")

    resolved = sorted(relevant, key=lambda case: (case.resolved_at, case.case_id))
    resolved_pointer = 0
    history_by_horizon: dict[int, list[HistoricalCase]] = {
        horizon: [] for horizon in normalized_horizons
    }
    seen_by_country: dict[str, int] = {}
    baseline_points: dict[int, list[BacktestPoint]] = {
        horizon: [] for horizon in normalized_horizons
    }
    raw_points: dict[int, list[BacktestPoint]] = {
        horizon: [] for horizon in normalized_horizons
    }
    adjusted_points: dict[int, list[BacktestPoint]] = {
        horizon: [] for horizon in normalized_horizons
    }
    evaluated_ids: list[str] = []
    evaluated_cutoffs: list[datetime] = []
    crossing_count = 0
    crossing_magnitudes: list[float] = []
    adjustments: list[float] = []
    restricted_survival: list[float] = []
    baseline_curve_scores: list[float] = []
    raw_curve_scores: list[float] = []
    adjusted_curve_scores: list[float] = []

    for snapshot_id, targets_by_horizon in complete_targets:
        target = targets_by_horizon[normalized_horizons[0]]
        cutoff = target.cutoff_at
        while (
            resolved_pointer < len(resolved)
            and resolved[resolved_pointer].resolved_at < cutoff
        ):
            historical = resolved[resolved_pointer]
            historical_horizon = _horizon_from_domain(historical, domain_prefix)
            if historical_horizon in history_by_horizon:
                history_by_horizon[historical_horizon].append(historical)
            resolved_pointer += 1

        country = target.tags[0] if target.tags else "all"
        seen = seen_by_country.get(country, 0)
        seen_by_country[country] = seen + 1
        if seen % target_stride:
            continue

        reference_features = target.features
        outcomes = [
            bool(targets_by_horizon[horizon].outcome)
            for horizon in normalized_horizons
        ]
        if [int(value) for value in outcomes] != sorted(int(value) for value in outcomes):
            raise ValueError(f"snapshot {snapshot_id} has non-monotone outcome labels")
        for horizon in normalized_horizons[1:]:
            if targets_by_horizon[horizon].features != reference_features:
                raise ValueError(f"snapshot {snapshot_id} has inconsistent features")

        eligible_by_horizon: dict[int, list[HistoricalCase]] = {}
        ready = True
        for horizon in normalized_horizons:
            eligible = history_by_horizon[horizon]
            if len(eligible) < minimum_training_cases:
                ready = False
                break
            eligible_by_horizon[horizon] = (
                eligible[-max_history:] if max_history is not None else eligible
            )
        if not ready:
            continue

        raw_probabilities: dict[int, float] = {}
        evidence = {}
        baseline_probabilities: dict[int, float] = {}
        for horizon in normalized_horizons:
            eligible = eligible_by_horizon[horizon]
            yes_count = sum(case.outcome for case in eligible)
            baseline_probabilities[horizon] = (yes_count + 1.0) / (
                len(eligible) + 2.0
            )
            result = model.fit_predict(
                eligible,
                target_features=reference_features,
                target_cutoff=cutoff,
                domain=f"{domain_prefix}_{horizon}d",
            )
            raw_probabilities[horizon] = result.probability
            evidence[horizon] = result

        curve, restricted_mean = coherent_survival_curve(
            raw_probabilities,
            evidence=evidence,
        )
        raw_values = [raw_probabilities[horizon] for horizon in normalized_horizons]
        crossing_magnitude = sum(
            max(0.0, left - right)
            for left, right in zip(raw_values, raw_values[1:])
        )
        if crossing_magnitude > 0:
            crossing_count += 1
        crossing_magnitudes.append(crossing_magnitude)
        restricted_survival.append(restricted_mean)

        baseline_snapshot_errors: list[float] = []
        raw_snapshot_errors: list[float] = []
        adjusted_snapshot_errors: list[float] = []
        for point, outcome in zip(curve, outcomes, strict=True):
            horizon = point.horizon_days
            training_cases = len(eligible_by_horizon[horizon])
            baseline_probability = baseline_probabilities[horizon]
            baseline_points[horizon].append(
                BacktestPoint(
                    case_id=snapshot_id,
                    probability=baseline_probability,
                    outcome=outcome,
                    training_cases=training_cases,
                )
            )
            raw_points[horizon].append(
                BacktestPoint(
                    case_id=snapshot_id,
                    probability=point.raw_exit_probability,
                    outcome=outcome,
                    training_cases=training_cases,
                )
            )
            adjusted_points[horizon].append(
                BacktestPoint(
                    case_id=snapshot_id,
                    probability=point.adjusted_exit_probability,
                    outcome=outcome,
                    training_cases=training_cases,
                )
            )
            outcome_value = 1.0 if outcome else 0.0
            baseline_snapshot_errors.append(
                (baseline_probability - outcome_value) ** 2
            )
            raw_snapshot_errors.append(
                (point.raw_exit_probability - outcome_value) ** 2
            )
            adjusted_snapshot_errors.append(
                (point.adjusted_exit_probability - outcome_value) ** 2
            )
            adjustments.append(
                abs(point.adjusted_exit_probability - point.raw_exit_probability)
            )

        baseline_curve_scores.append(fmean(baseline_snapshot_errors))
        raw_curve_scores.append(fmean(raw_snapshot_errors))
        adjusted_curve_scores.append(fmean(adjusted_snapshot_errors))
        evaluated_ids.append(snapshot_id)
        evaluated_cutoffs.append(cutoff)

    if not evaluated_ids:
        raise ValueError("no survival forecasts met the training requirement")

    horizon_reports: list[HorizonSurvivalMetrics] = []
    for horizon in normalized_horizons:
        baseline = _metrics(baseline_points[horizon])
        raw = _metrics(raw_points[horizon])
        adjusted = _metrics(adjusted_points[horizon])
        horizon_reports.append(
            HorizonSurvivalMetrics(
                horizon_days=horizon,
                predictions=adjusted.predictions,
                positives=sum(point.outcome for point in adjusted_points[horizon]),
                baseline=baseline,
                raw_analog=raw,
                adjusted_analog=adjusted,
                adjusted_brier_skill_vs_raw=_skill(
                    adjusted.brier_score, raw.brier_score
                ),
                adjusted_brier_skill_vs_baseline=_skill(
                    adjusted.brier_score, baseline.brier_score
                ),
            )
        )

    baseline_integrated = fmean(baseline_curve_scores)
    raw_integrated = fmean(raw_curve_scores)
    adjusted_integrated = fmean(adjusted_curve_scores)
    return SurvivalBenchmarkReport(
        holdout_start=holdout_start.isoformat(),
        horizons=normalized_horizons,
        snapshots=len(evaluated_ids),
        target_stride=target_stride,
        minimum_training_cases=minimum_training_cases,
        max_history=max_history,
        raw_crossing_curves=crossing_count,
        raw_crossing_rate=crossing_count / len(evaluated_ids),
        mean_crossing_magnitude=fmean(crossing_magnitudes),
        mean_absolute_adjustment=fmean(adjustments),
        baseline_integrated_brier=baseline_integrated,
        raw_integrated_brier=raw_integrated,
        adjusted_integrated_brier=adjusted_integrated,
        adjusted_integrated_brier_skill_vs_raw=_skill(
            adjusted_integrated, raw_integrated
        ),
        adjusted_integrated_brier_skill_vs_baseline=_skill(
            adjusted_integrated, baseline_integrated
        ),
        mean_restricted_survival_days=fmean(restricted_survival),
        first_cutoff=min(evaluated_cutoffs).isoformat(),
        last_cutoff=max(evaluated_cutoffs).isoformat(),
        snapshot_ids=tuple(evaluated_ids),
        horizon_metrics=tuple(horizon_reports),
    )
