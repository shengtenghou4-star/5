from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from statistics import fmean
from typing import Iterable, Protocol

from .engine import ForecastResult
from .models import FeatureValue, HistoricalCase, ensure_aware
from .survival import coherent_survival_curve


class BinaryForecaster(Protocol):
    def fit_predict(
        self,
        history: Iterable[HistoricalCase],
        *,
        target_features: dict[str, FeatureValue],
        target_cutoff: datetime,
        domain: str | None = None,
    ) -> ForecastResult: ...


@dataclass(frozen=True, slots=True)
class SurvivalSnapshotScore:
    snapshot_id: str
    country: str
    cutoff_at: str
    baseline_integrated_brier: float
    model_integrated_brier: float
    paired_delta: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ClusterBootstrapReport:
    clusters: int
    snapshots: int
    replicates: int
    seed: int
    observed_baseline_brier: float
    observed_model_brier: float
    observed_paired_delta: float
    observed_brier_skill: float
    delta_ci_low: float
    delta_ci_high: float
    skill_ci_low: float
    skill_ci_high: float
    probability_model_better: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _horizon_from_domain(case: HistoricalCase, domain_prefix: str) -> int | None:
    prefix = f"{domain_prefix}_"
    if not case.domain.startswith(prefix) or not case.domain.endswith("d"):
        return None
    body = case.domain[len(prefix) : -1]
    try:
        horizon = int(body)
    except ValueError:
        return None
    return horizon if horizon > 0 else None


def _snapshot_id(case: HistoricalCase, horizon: int) -> str:
    suffix = f":{horizon}d"
    if case.case_id.endswith(suffix):
        return case.case_id[: -len(suffix)]
    country = case.tags[0] if case.tags else "all"
    return f"{country}:{case.cutoff_at.isoformat()}:{case.case_id}"


def paired_survival_snapshot_scores(
    cases: list[HistoricalCase],
    model: BinaryForecaster,
    *,
    holdout_start: datetime,
    horizons: Iterable[int] = (30, 90, 180, 365),
    domain_prefix: str = "government_leader_exit",
    minimum_training_cases: int = 500,
    target_stride: int = 3,
    max_history: int | None = 3000,
) -> list[SurvivalSnapshotScore]:
    """Return paired per-snapshot Brier scores from a time-safe replay."""
    holdout_start = ensure_aware(holdout_start)
    normalized_horizons = tuple(sorted(set(horizons)))
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError("horizons must contain positive values")
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
    scores: list[SurvivalSnapshotScore] = []

    for snapshot_id, targets_by_horizon in complete_targets:
        target = targets_by_horizon[normalized_horizons[0]]
        cutoff = target.cutoff_at
        while (
            resolved_pointer < len(resolved)
            and resolved[resolved_pointer].resolved_at < cutoff
        ):
            historical = resolved[resolved_pointer]
            horizon = _horizon_from_domain(historical, domain_prefix)
            if horizon in history_by_horizon:
                history_by_horizon[horizon].append(historical)
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
        if any(
            targets_by_horizon[horizon].features != reference_features
            for horizon in normalized_horizons[1:]
        ):
            raise ValueError(f"snapshot {snapshot_id} has inconsistent features")

        eligible_by_horizon: dict[int, list[HistoricalCase]] = {}
        for horizon in normalized_horizons:
            eligible = history_by_horizon[horizon]
            if len(eligible) < minimum_training_cases:
                eligible_by_horizon = {}
                break
            eligible_by_horizon[horizon] = (
                eligible[-max_history:] if max_history is not None else eligible
            )
        if not eligible_by_horizon:
            continue

        baseline_probabilities: dict[int, float] = {}
        model_probabilities: dict[int, float] = {}
        evidence: dict[int, ForecastResult] = {}
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
            model_probabilities[horizon] = result.probability
            evidence[horizon] = result

        curve, _ = coherent_survival_curve(model_probabilities, evidence=evidence)
        adjusted = {
            point.horizon_days: point.adjusted_exit_probability for point in curve
        }
        baseline_score = fmean(
            (baseline_probabilities[horizon] - float(outcome)) ** 2
            for horizon, outcome in zip(normalized_horizons, outcomes, strict=True)
        )
        model_score = fmean(
            (adjusted[horizon] - float(outcome)) ** 2
            for horizon, outcome in zip(normalized_horizons, outcomes, strict=True)
        )
        scores.append(
            SurvivalSnapshotScore(
                snapshot_id=snapshot_id,
                country=country,
                cutoff_at=cutoff.isoformat(),
                baseline_integrated_brier=baseline_score,
                model_integrated_brier=model_score,
                paired_delta=model_score - baseline_score,
            )
        )

    if not scores:
        raise ValueError("no survival snapshots met the training requirement")
    return scores


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a percentile of an empty collection")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def country_cluster_bootstrap(
    scores: list[SurvivalSnapshotScore],
    *,
    replicates: int = 10_000,
    seed: int = 20260714,
    confidence: float = 0.95,
) -> ClusterBootstrapReport:
    """Paired country-cluster bootstrap for the integrated Brier difference."""
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    by_country: dict[str, list[SurvivalSnapshotScore]] = defaultdict(list)
    for score in scores:
        by_country[score.country].append(score)
    countries = sorted(by_country)
    if len(countries) < 2:
        raise ValueError("at least two country clusters are required")

    baseline_observed = fmean(item.baseline_integrated_brier for item in scores)
    model_observed = fmean(item.model_integrated_brier for item in scores)
    delta_observed = model_observed - baseline_observed
    skill_observed = (
        0.0 if baseline_observed == 0 else 1.0 - model_observed / baseline_observed
    )

    rng = random.Random(seed)
    deltas: list[float] = []
    skills: list[float] = []
    for _ in range(replicates):
        sampled_countries = [rng.choice(countries) for _ in countries]
        sampled = [
            item
            for country in sampled_countries
            for item in by_country[country]
        ]
        baseline = fmean(item.baseline_integrated_brier for item in sampled)
        model_score = fmean(item.model_integrated_brier for item in sampled)
        deltas.append(model_score - baseline)
        skills.append(0.0 if baseline == 0 else 1.0 - model_score / baseline)

    tail = (1.0 - confidence) / 2.0
    return ClusterBootstrapReport(
        clusters=len(countries),
        snapshots=len(scores),
        replicates=replicates,
        seed=seed,
        observed_baseline_brier=baseline_observed,
        observed_model_brier=model_observed,
        observed_paired_delta=delta_observed,
        observed_brier_skill=skill_observed,
        delta_ci_low=_percentile(deltas, tail),
        delta_ci_high=_percentile(deltas, 1.0 - tail),
        skill_ci_low=_percentile(skills, tail),
        skill_ci_high=_percentile(skills, 1.0 - tail),
        probability_model_better=sum(value < 0.0 for value in deltas) / replicates,
    )
