from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from .engine import AnalogForecaster, ForecastResult
from .models import FeatureValue, HistoricalCase, ensure_aware


@dataclass(frozen=True, slots=True)
class SurvivalPoint:
    horizon_days: int
    raw_exit_probability: float
    adjusted_exit_probability: float
    survival_probability: float
    interval_exit_probability: float
    prior_probability: float | None = None
    effective_sample_size: float | None = None
    feature_coverage: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "horizon_days": self.horizon_days,
            "raw_exit_probability": self.raw_exit_probability,
            "adjusted_exit_probability": self.adjusted_exit_probability,
            "survival_probability": self.survival_probability,
            "interval_exit_probability": self.interval_exit_probability,
            "prior_probability": self.prior_probability,
            "effective_sample_size": self.effective_sample_size,
            "feature_coverage": self.feature_coverage,
        }


@dataclass(frozen=True, slots=True)
class LeaderSurvivalForecast:
    cutoff_at: datetime
    domain_prefix: str
    points: tuple[SurvivalPoint, ...]
    restricted_mean_survival_days: float

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoff_at": self.cutoff_at.isoformat(),
            "domain_prefix": self.domain_prefix,
            "restricted_mean_survival_days": self.restricted_mean_survival_days,
            "points": [point.to_dict() for point in self.points],
        }


def _validate_probability(value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probabilities must be between 0 and 1")
    return probability


def _isotonic_non_decreasing(values: Iterable[float]) -> tuple[float, ...]:
    """Pool adjacent violators using equal weights.

    Independently trained horizon models can produce impossible curves such as
    P(exit by 30d) > P(exit by 90d). PAVA changes the probabilities as little
    as possible in squared-error terms while enforcing chronological order.
    """
    blocks: list[list[float | int]] = []
    for raw in values:
        value = _validate_probability(raw)
        blocks.append([value, 1, 1])  # weighted sum, weight, represented items
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_mean = float(left[0]) / int(left[1])
            right_mean = float(right[0]) / int(right[1])
            if left_mean <= right_mean:
                break
            blocks[-2:] = [
                [
                    float(left[0]) + float(right[0]),
                    int(left[1]) + int(right[1]),
                    int(left[2]) + int(right[2]),
                ]
            ]

    adjusted: list[float] = []
    for weighted_sum, weight, count in blocks:
        adjusted.extend([float(weighted_sum) / int(weight)] * int(count))
    return tuple(adjusted)


def coherent_survival_curve(
    exit_probabilities: Mapping[int, float],
    *,
    evidence: Mapping[int, ForecastResult] | None = None,
) -> tuple[tuple[SurvivalPoint, ...], float]:
    """Turn horizon-specific exit probabilities into a valid survival curve."""
    if not exit_probabilities:
        raise ValueError("at least one horizon probability is required")
    horizons = tuple(sorted(exit_probabilities))
    if any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")

    raw = tuple(_validate_probability(exit_probabilities[h]) for h in horizons)
    adjusted = _isotonic_non_decreasing(raw)
    points: list[SurvivalPoint] = []
    previous_exit_probability = 0.0
    previous_horizon = 0
    previous_survival = 1.0
    restricted_mean = 0.0

    for horizon, raw_probability, adjusted_probability in zip(
        horizons, raw, adjusted, strict=True
    ):
        survival_probability = 1.0 - adjusted_probability
        interval_exit_probability = adjusted_probability - previous_exit_probability
        restricted_mean += (horizon - previous_horizon) * (
            previous_survival + survival_probability
        ) / 2.0
        forecast = evidence.get(horizon) if evidence else None
        points.append(
            SurvivalPoint(
                horizon_days=horizon,
                raw_exit_probability=raw_probability,
                adjusted_exit_probability=adjusted_probability,
                survival_probability=survival_probability,
                interval_exit_probability=interval_exit_probability,
                prior_probability=(forecast.prior_probability if forecast else None),
                effective_sample_size=(
                    forecast.effective_sample_size if forecast else None
                ),
                feature_coverage=(forecast.feature_coverage if forecast else None),
            )
        )
        previous_exit_probability = adjusted_probability
        previous_horizon = horizon
        previous_survival = survival_probability

    return tuple(points), restricted_mean


def forecast_survival_curve(
    history: Iterable[HistoricalCase],
    *,
    target_features: dict[str, FeatureValue],
    target_cutoff: datetime,
    horizons: Iterable[int] = (30, 90, 180, 365),
    domain_prefix: str = "government_leader_exit",
    model: AnalogForecaster | None = None,
) -> LeaderSurvivalForecast:
    """Forecast several exit horizons and reconcile them into one curve."""
    target_cutoff = ensure_aware(target_cutoff)
    normalized_horizons = tuple(sorted(set(horizons)))
    if not normalized_horizons:
        raise ValueError("at least one horizon is required")
    if any(horizon <= 0 for horizon in normalized_horizons):
        raise ValueError("horizons must be positive")

    forecaster = model or AnalogForecaster()
    frozen_history = tuple(history)
    raw: dict[int, float] = {}
    evidence: dict[int, ForecastResult] = {}
    for horizon in normalized_horizons:
        result = forecaster.fit_predict(
            frozen_history,
            target_features=target_features,
            target_cutoff=target_cutoff,
            domain=f"{domain_prefix}_{horizon}d",
        )
        raw[horizon] = result.probability
        evidence[horizon] = result

    points, restricted_mean = coherent_survival_curve(raw, evidence=evidence)
    return LeaderSurvivalForecast(
        cutoff_at=target_cutoff,
        domain_prefix=domain_prefix,
        points=points,
        restricted_mean_survival_days=restricted_mean,
    )
