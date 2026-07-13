from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import fmean

from .engine import AnalogForecaster
from .models import HistoricalCase, ensure_aware
from .scoring import (
    BacktestPoint,
    binary_log_loss,
    brier_score,
    calibration_error,
)


@dataclass(frozen=True, slots=True)
class MetricSet:
    predictions: int
    brier_score: float
    log_loss: float
    calibration_error: float


@dataclass(frozen=True, slots=True)
class TemporalBenchmarkReport:
    holdout_start: str
    target_stride: int
    max_history: int | None
    baseline: MetricSet
    analog: MetricSet
    analog_brier_skill: float
    first_cutoff: str
    last_cutoff: str

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


def temporal_holdout_benchmark(
    cases: list[HistoricalCase],
    forecaster: AnalogForecaster,
    *,
    holdout_start: datetime,
    minimum_training_cases: int = 100,
    target_stride: int = 1,
    max_history: int | None = 5_000,
) -> TemporalBenchmarkReport:
    """Compare analog forecasts with a smoothed base rate on a later time block.

    Earlier holdout cases may enter training only after they have resolved. This
    mirrors live operation and prevents the final time block from being treated
    as a conventional random test split.
    """
    holdout_start = ensure_aware(holdout_start)
    if target_stride <= 0:
        raise ValueError("target_stride must be positive")
    if minimum_training_cases <= 0:
        raise ValueError("minimum_training_cases must be positive")
    if max_history is not None and max_history <= 0:
        raise ValueError("max_history must be positive when provided")

    ordered = sorted(cases, key=lambda case: (case.cutoff_at, case.case_id))
    resolved = sorted(cases, key=lambda case: (case.resolved_at, case.case_id))
    targets = [case for case in ordered if case.cutoff_at >= holdout_start]
    if not targets:
        raise ValueError("no cases exist in the requested holdout period")

    history: list[HistoricalCase] = []
    resolved_pointer = 0
    seen_by_country: dict[str, int] = {}
    baseline_points: list[BacktestPoint] = []
    analog_points: list[BacktestPoint] = []

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

        eligible = [case for case in history if case.domain == target.domain]
        if len(eligible) < minimum_training_cases:
            continue
        if max_history is not None:
            eligible = eligible[-max_history:]

        yes_count = sum(case.outcome for case in eligible)
        baseline_probability = (yes_count + 1.0) / (len(eligible) + 2.0)
        baseline_points.append(
            BacktestPoint(
                case_id=target.case_id,
                probability=baseline_probability,
                outcome=target.outcome,
                training_cases=len(eligible),
            )
        )

        result = forecaster.fit_predict(
            eligible,
            target_features=target.features,
            target_cutoff=target.cutoff_at,
            domain=target.domain,
        )
        analog_points.append(
            BacktestPoint(
                case_id=target.case_id,
                probability=result.probability,
                outcome=target.outcome,
                training_cases=len(eligible),
            )
        )

    if not analog_points:
        raise ValueError("no holdout predictions were generated")

    baseline_metrics = _metrics(baseline_points)
    analog_metrics = _metrics(analog_points)
    if baseline_metrics.brier_score:
        brier_skill = 1.0 - (
            analog_metrics.brier_score / baseline_metrics.brier_score
        )
    else:
        brier_skill = 0.0

    return TemporalBenchmarkReport(
        holdout_start=holdout_start.isoformat(),
        target_stride=target_stride,
        max_history=max_history,
        baseline=baseline_metrics,
        analog=analog_metrics,
        analog_brier_skill=brier_skill,
        first_cutoff=targets[0].cutoff_at.isoformat(),
        last_cutoff=targets[-1].cutoff_at.isoformat(),
    )
