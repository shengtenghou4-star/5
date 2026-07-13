from __future__ import annotations

from dataclasses import dataclass
from math import log
from statistics import fmean
from typing import Iterable

from .engine import AnalogForecaster
from .models import HistoricalCase


@dataclass(frozen=True, slots=True)
class BacktestPoint:
    case_id: str
    probability: float
    outcome: bool
    training_cases: int


@dataclass(frozen=True, slots=True)
class BacktestReport:
    points: tuple[BacktestPoint, ...]
    brier_score: float
    log_loss: float
    calibration_error: float


def brier_score(probability: float, outcome: bool) -> float:
    return (probability - float(outcome)) ** 2


def binary_log_loss(probability: float, outcome: bool) -> float:
    clipped = min(1.0 - 1e-12, max(1e-12, probability))
    return -(float(outcome) * log(clipped) + (1.0 - float(outcome)) * log(1.0 - clipped))


def calibration_error(points: Iterable[BacktestPoint], bins: int = 5) -> float:
    point_list = list(points)
    if not point_list:
        return 0.0
    total = len(point_list)
    error = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        bucket = [
            point
            for point in point_list
            if lower <= point.probability < upper
            or (index == bins - 1 and point.probability == 1.0)
        ]
        if not bucket:
            continue
        confidence = fmean(point.probability for point in bucket)
        accuracy = fmean(float(point.outcome) for point in bucket)
        error += (len(bucket) / total) * abs(confidence - accuracy)
    return error


def walk_forward_backtest(
    cases: Iterable[HistoricalCase],
    forecaster: AnalogForecaster,
    *,
    minimum_training_cases: int = 3,
    same_domain_only: bool = True,
) -> BacktestReport:
    ordered = sorted(cases, key=lambda case: (case.cutoff_at, case.case_id))
    points: list[BacktestPoint] = []
    for target in ordered:
        history = [case for case in ordered if case.resolved_at < target.cutoff_at]
        if same_domain_only:
            history = [case for case in history if case.domain == target.domain]
        if len(history) < minimum_training_cases:
            continue
        result = forecaster.fit_predict(
            history,
            target_features=target.features,
            target_cutoff=target.cutoff_at,
            domain=target.domain if same_domain_only else None,
        )
        points.append(
            BacktestPoint(
                case_id=target.case_id,
                probability=result.probability,
                outcome=target.outcome,
                training_cases=len(history),
            )
        )

    if not points:
        return BacktestReport((), 0.0, 0.0, 0.0)
    return BacktestReport(
        points=tuple(points),
        brier_score=fmean(brier_score(p.probability, p.outcome) for p in points),
        log_loss=fmean(binary_log_loss(p.probability, p.outcome) for p in points),
        calibration_error=calibration_error(points),
    )
