from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Mapping

from .engine import AnalogForecaster, ForecastResult
from .models import FeatureValue, HistoricalCase, ensure_aware
from .survival import coherent_survival_curve


@dataclass(frozen=True, slots=True)
class MechanismRisk:
    mechanism: str
    raw_score: float
    cumulative_probability: float
    interval_probability: float
    prior_probability: float | None = None
    effective_sample_size: float | None = None
    feature_coverage: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "mechanism": self.mechanism,
            "raw_score": self.raw_score,
            "cumulative_probability": self.cumulative_probability,
            "interval_probability": self.interval_probability,
            "prior_probability": self.prior_probability,
            "effective_sample_size": self.effective_sample_size,
            "feature_coverage": self.feature_coverage,
        }


@dataclass(frozen=True, slots=True)
class CompetingRiskPoint:
    horizon_days: int
    total_exit_probability: float
    survival_probability: float
    interval_exit_probability: float
    mechanisms: tuple[MechanismRisk, ...]
    prior_probability: float | None = None
    effective_sample_size: float | None = None
    feature_coverage: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "horizon_days": self.horizon_days,
            "total_exit_probability": self.total_exit_probability,
            "survival_probability": self.survival_probability,
            "interval_exit_probability": self.interval_exit_probability,
            "prior_probability": self.prior_probability,
            "effective_sample_size": self.effective_sample_size,
            "feature_coverage": self.feature_coverage,
            "mechanisms": [item.to_dict() for item in self.mechanisms],
        }


@dataclass(frozen=True, slots=True)
class LeaderCompetingRiskForecast:
    cutoff_at: datetime
    domain_prefix: str
    mechanisms: tuple[str, ...]
    points: tuple[CompetingRiskPoint, ...]
    restricted_mean_survival_days: float

    def to_dict(self) -> dict[str, object]:
        return {
            "cutoff_at": self.cutoff_at.isoformat(),
            "domain_prefix": self.domain_prefix,
            "mechanisms": list(self.mechanisms),
            "restricted_mean_survival_days": self.restricted_mean_survival_days,
            "points": [point.to_dict() for point in self.points],
        }


def _probability(value: float) -> float:
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probabilities must be between 0 and 1")
    return probability


def _mechanism_names(
    mechanism_scores: Mapping[str, Mapping[int, float]],
) -> tuple[str, ...]:
    normalized = [name.strip() for name in mechanism_scores]
    if not normalized:
        raise ValueError("at least one exit mechanism is required")
    if any(not name for name in normalized) or len(set(normalized)) != len(normalized):
        raise ValueError("exit mechanism names must be non-empty and unique")
    return tuple(sorted(normalized))


def coherent_competing_risk_curve(
    total_exit_probabilities: Mapping[int, float],
    mechanism_scores: Mapping[str, Mapping[int, float]],
    *,
    total_evidence: Mapping[int, ForecastResult] | None = None,
    mechanism_evidence: Mapping[str, Mapping[int, ForecastResult]] | None = None,
) -> tuple[tuple[CompetingRiskPoint, ...], float]:
    """Reconcile total and cause-specific forecasts into one valid curve.

    The total exit curve is made monotone by the survival module. Each interval's
    newly added exit mass is then allocated across mechanisms. Positive changes
    in cause-specific scores receive priority; flat or falling scores fall back
    to the current score mix, and an all-zero mix is split evenly. This makes
    every cumulative cause curve non-decreasing while preserving the exact total.
    """
    names = _mechanism_names(mechanism_scores)
    total_points, restricted_mean = coherent_survival_curve(
        total_exit_probabilities,
        evidence=total_evidence,
    )
    horizons = tuple(point.horizon_days for point in total_points)
    unknown_horizons = {
        horizon
        for scores in mechanism_scores.values()
        for horizon in scores
        if horizon not in horizons
    }
    if unknown_horizons:
        raise ValueError(
            "mechanism scores contain horizons absent from total exit curve: "
            + ", ".join(str(value) for value in sorted(unknown_horizons))
        )

    previous_scores = {name: 0.0 for name in names}
    cumulative = {name: 0.0 for name in names}
    points: list[CompetingRiskPoint] = []

    for total_point in total_points:
        horizon = total_point.horizon_days
        current_scores = {
            name: _probability(mechanism_scores[name].get(horizon, 0.0))
            for name in names
        }
        positive_changes = {
            name: max(0.0, current_scores[name] - previous_scores[name])
            for name in names
        }
        basis = positive_changes
        basis_sum = sum(basis.values())
        if basis_sum <= 1e-15:
            basis = current_scores
            basis_sum = sum(basis.values())
        if basis_sum <= 1e-15:
            weights = {name: 1.0 / len(names) for name in names}
        else:
            weights = {name: basis[name] / basis_sum for name in names}

        interval_mass = total_point.interval_exit_probability
        allocations: dict[str, float] = {}
        remaining = interval_mass
        for name in names[:-1]:
            allocation = interval_mass * weights[name]
            allocations[name] = allocation
            remaining -= allocation
        allocations[names[-1]] = max(0.0, remaining)

        mechanism_points: list[MechanismRisk] = []
        for name in names:
            cumulative[name] += allocations[name]
            evidence = (
                mechanism_evidence.get(name, {}).get(horizon)
                if mechanism_evidence
                else None
            )
            mechanism_points.append(
                MechanismRisk(
                    mechanism=name,
                    raw_score=current_scores[name],
                    cumulative_probability=cumulative[name],
                    interval_probability=allocations[name],
                    prior_probability=(evidence.prior_probability if evidence else None),
                    effective_sample_size=(
                        evidence.effective_sample_size if evidence else None
                    ),
                    feature_coverage=(evidence.feature_coverage if evidence else None),
                )
            )
            previous_scores[name] = current_scores[name]

        points.append(
            CompetingRiskPoint(
                horizon_days=horizon,
                total_exit_probability=total_point.adjusted_exit_probability,
                survival_probability=total_point.survival_probability,
                interval_exit_probability=interval_mass,
                mechanisms=tuple(mechanism_points),
                prior_probability=total_point.prior_probability,
                effective_sample_size=total_point.effective_sample_size,
                feature_coverage=total_point.feature_coverage,
            )
        )

    return tuple(points), restricted_mean


def forecast_competing_risks(
    history: Iterable[HistoricalCase],
    *,
    target_features: dict[str, FeatureValue],
    target_cutoff: datetime,
    mechanisms: Iterable[str],
    horizons: Iterable[int] = (30, 90, 180, 365),
    domain_prefix: str = "government_leader_exit",
    model: AnalogForecaster | None = None,
) -> LeaderCompetingRiskForecast:
    """Forecast total exit risk and its mutually exclusive transition channels."""
    target_cutoff = ensure_aware(target_cutoff)
    normalized_horizons = tuple(sorted(set(horizons)))
    normalized_mechanisms = tuple(sorted(set(item.strip() for item in mechanisms)))
    if not normalized_horizons:
        raise ValueError("at least one horizon is required")
    if any(horizon <= 0 for horizon in normalized_horizons):
        raise ValueError("horizons must be positive")
    if not normalized_mechanisms or any(not item for item in normalized_mechanisms):
        raise ValueError("at least one non-empty exit mechanism is required")

    forecaster = model or AnalogForecaster()
    frozen_history = tuple(history)
    total_raw: dict[int, float] = {}
    total_evidence: dict[int, ForecastResult] = {}
    mechanism_raw = {name: {} for name in normalized_mechanisms}
    mechanism_evidence: dict[str, dict[int, ForecastResult]] = {
        name: {} for name in normalized_mechanisms
    }

    for horizon in normalized_horizons:
        total = forecaster.fit_predict(
            frozen_history,
            target_features=target_features,
            target_cutoff=target_cutoff,
            domain=f"{domain_prefix}_{horizon}d",
        )
        total_raw[horizon] = total.probability
        total_evidence[horizon] = total
        for mechanism in normalized_mechanisms:
            result = forecaster.fit_predict(
                frozen_history,
                target_features=target_features,
                target_cutoff=target_cutoff,
                domain=f"{domain_prefix}_{mechanism}_{horizon}d",
            )
            mechanism_raw[mechanism][horizon] = result.probability
            mechanism_evidence[mechanism][horizon] = result

    points, restricted_mean = coherent_competing_risk_curve(
        total_raw,
        mechanism_raw,
        total_evidence=total_evidence,
        mechanism_evidence=mechanism_evidence,
    )
    return LeaderCompetingRiskForecast(
        cutoff_at=target_cutoff,
        domain_prefix=domain_prefix,
        mechanisms=normalized_mechanisms,
        points=points,
        restricted_mean_survival_days=restricted_mean,
    )
