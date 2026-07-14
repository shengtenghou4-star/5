from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import log
from statistics import fmean
from typing import Iterable

from .benchmark import MetricSet
from .competing_risks import CompetingRiskPoint, coherent_competing_risk_curve
from .engine import AnalogForecaster
from .models import HistoricalCase, ensure_aware
from .scoring import BacktestPoint, binary_log_loss, brier_score, calibration_error


@dataclass(frozen=True, slots=True)
class ConditionalMetricSet:
    predictions: int
    brier_score: float
    log_loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class MechanismMetrics:
    mechanism: str
    positives: int
    baseline: MetricSet
    adjusted_model: MetricSet
    brier_skill_vs_baseline: float


@dataclass(frozen=True, slots=True)
class HorizonCompetingMetrics:
    horizon_days: int
    snapshots: int
    exits: int
    baseline_total: MetricSet
    adjusted_total: MetricSet
    total_brier_skill_vs_baseline: float
    baseline_mean_mechanism_brier: float
    adjusted_mean_mechanism_brier: float
    mechanism_brier_skill_vs_baseline: float
    baseline_conditional: ConditionalMetricSet
    adjusted_conditional: ConditionalMetricSet
    conditional_brier_skill_vs_baseline: float
    conditional_log_loss_skill_vs_baseline: float
    mechanisms: tuple[MechanismMetrics, ...]


@dataclass(frozen=True, slots=True)
class CompetingBenchmarkReport:
    holdout_start: str
    horizons: tuple[int, ...]
    mechanisms: tuple[str, ...]
    snapshots: int
    exit_observations: int
    target_stride: int
    minimum_training_cases: int
    max_history: int | None
    baseline_integrated_total_brier: float
    adjusted_integrated_total_brier: float
    total_brier_skill_vs_baseline: float
    baseline_integrated_mechanism_brier: float
    adjusted_integrated_mechanism_brier: float
    mechanism_brier_skill_vs_baseline: float
    baseline_conditional_brier: float
    adjusted_conditional_brier: float
    conditional_brier_skill_vs_baseline: float
    baseline_conditional_log_loss: float
    adjusted_conditional_log_loss: float
    conditional_log_loss_skill_vs_baseline: float
    baseline_conditional_accuracy: float
    adjusted_conditional_accuracy: float
    max_baseline_conservation_error: float
    max_adjusted_conservation_error: float
    first_cutoff: str
    last_cutoff: str
    snapshot_ids: tuple[str, ...]
    horizon_metrics: tuple[HorizonCompetingMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _metrics(points: list[BacktestPoint]) -> MetricSet:
    if not points:
        return MetricSet(
            predictions=0,
            brier_score=0.0,
            log_loss=0.0,
            calibration_error=0.0,
        )
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


def _conditional_metrics(
    values: list[tuple[float, float, bool]],
) -> ConditionalMetricSet:
    if not values:
        return ConditionalMetricSet(0, 0.0, 0.0, 0.0)
    return ConditionalMetricSet(
        predictions=len(values),
        brier_score=fmean(value[0] for value in values),
        log_loss=fmean(value[1] for value in values),
        accuracy=fmean(float(value[2]) for value in values),
    )


def _skill(score: float, reference: float) -> float:
    return 0.0 if reference == 0 else 1.0 - score / reference


def _normalized_names(values: Iterable[str]) -> tuple[str, ...]:
    names = tuple(sorted(set(value.strip() for value in values)))
    if not names or any(not value for value in names):
        raise ValueError("mechanisms must contain at least one non-empty name")
    return names


def _case_role(
    case: HistoricalCase,
    *,
    domain_prefix: str,
) -> tuple[int, str | None] | None:
    horizon: int | None = None
    mechanism: str | None = None
    for tag in case.tags:
        if tag.startswith("horizon:"):
            try:
                horizon = int(tag.split(":", 1)[1])
            except ValueError:
                return None
        elif tag.startswith("mechanism:"):
            mechanism = tag.split(":", 1)[1].strip() or None

    prefix = f"{domain_prefix}_"
    if not case.domain.startswith(prefix) or not case.domain.endswith("d"):
        return None
    body = case.domain[len(prefix) : -1]
    if mechanism is None:
        try:
            parsed_horizon = int(body)
        except ValueError:
            if "_" not in body:
                return None
            mechanism_body, horizon_text = body.rsplit("_", 1)
            try:
                parsed_horizon = int(horizon_text)
            except ValueError:
                return None
            mechanism = mechanism_body or None
        if horizon is None:
            horizon = parsed_horizon
    if horizon is None or horizon <= 0:
        return None
    return horizon, mechanism


def _snapshot_id(
    case: HistoricalCase,
    *,
    horizon: int,
    mechanism: str | None,
) -> str:
    suffix = f":{horizon}d" + (f":{mechanism}" if mechanism else "")
    if case.case_id.endswith(suffix):
        return case.case_id[: -len(suffix)]
    country = case.tags[0] if case.tags else "all"
    return f"{country}:{case.cutoff_at.isoformat()}:{case.case_id}"


def _curve_by_horizon(
    points: tuple[CompetingRiskPoint, ...],
) -> dict[int, CompetingRiskPoint]:
    return {point.horizon_days: point for point in points}


def _mechanism_probabilities(
    point: CompetingRiskPoint,
) -> dict[str, float]:
    return {
        item.mechanism: item.cumulative_probability for item in point.mechanisms
    }


def _conditional_score(
    point: CompetingRiskPoint,
    *,
    true_mechanism: str,
    mechanisms: tuple[str, ...],
) -> tuple[float, float, bool]:
    values = _mechanism_probabilities(point)
    total = sum(values.get(name, 0.0) for name in mechanisms)
    if total <= 1e-15:
        shares = {name: 1.0 / len(mechanisms) for name in mechanisms}
    else:
        shares = {name: values.get(name, 0.0) / total for name in mechanisms}
    brier = fmean(
        (shares[name] - float(name == true_mechanism)) ** 2
        for name in mechanisms
    )
    probability = min(1.0 - 1e-12, max(1e-12, shares[true_mechanism]))
    predicted = max(mechanisms, key=lambda name: (shares[name], name))
    return brier, -log(probability), predicted == true_mechanism


def temporal_competing_risk_benchmark(
    cases: list[HistoricalCase],
    model: AnalogForecaster,
    *,
    holdout_start: datetime,
    mechanisms: Iterable[str],
    horizons: Iterable[int] = (30, 90, 180, 365),
    domain_prefix: str = "government_leader_exit",
    minimum_training_cases: int = 100,
    target_stride: int = 3,
    max_history: int | None = 3000,
) -> CompetingBenchmarkReport:
    """Walk forward through complete cause-specific snapshots without leakage."""
    holdout_start = ensure_aware(holdout_start)
    normalized_horizons = tuple(sorted(set(horizons)))
    normalized_mechanisms = _normalized_names(mechanisms)
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError("horizons must contain at least one positive value")
    if minimum_training_cases <= 0:
        raise ValueError("minimum_training_cases must be positive")
    if target_stride <= 0:
        raise ValueError("target_stride must be positive")
    if max_history is not None and max_history <= 0:
        raise ValueError("max_history must be positive when provided")

    requested_horizons = set(normalized_horizons)
    requested_mechanisms = set(normalized_mechanisms)
    expected_roles = {None, *normalized_mechanisms}
    snapshots: dict[str, dict[int, dict[str | None, HistoricalCase]]] = {}
    relevant: list[HistoricalCase] = []

    for case in cases:
        role = _case_role(case, domain_prefix=domain_prefix)
        if role is None:
            continue
        horizon, mechanism = role
        if horizon not in requested_horizons:
            continue
        if mechanism is not None and mechanism not in requested_mechanisms:
            continue
        relevant.append(case)
        snapshot_id = _snapshot_id(
            case,
            horizon=horizon,
            mechanism=mechanism,
        )
        horizon_bucket = snapshots.setdefault(snapshot_id, {}).setdefault(horizon, {})
        if mechanism in horizon_bucket:
            label = mechanism or "total"
            raise ValueError(
                f"duplicate {label} case at horizon {horizon} for {snapshot_id}"
            )
        horizon_bucket[mechanism] = case

    complete_targets: list[
        tuple[str, dict[int, dict[str | None, HistoricalCase]]]
    ] = []
    for snapshot_id, horizons_by_role in snapshots.items():
        if set(horizons_by_role) != requested_horizons:
            continue
        if any(set(bucket) != expected_roles for bucket in horizons_by_role.values()):
            continue
        cutoffs = {
            case.cutoff_at
            for bucket in horizons_by_role.values()
            for case in bucket.values()
        }
        if len(cutoffs) != 1:
            raise ValueError(f"snapshot {snapshot_id} contains different cutoffs")
        if next(iter(cutoffs)) >= holdout_start:
            complete_targets.append((snapshot_id, horizons_by_role))
    complete_targets.sort(
        key=lambda item: (
            item[1][normalized_horizons[0]][None].cutoff_at,
            item[0],
        )
    )
    if not complete_targets:
        raise ValueError("no complete competing-risk snapshots exist in the holdout")

    resolved = sorted(relevant, key=lambda case: (case.resolved_at, case.case_id))
    resolved_pointer = 0
    history_by_domain: dict[str, list[HistoricalCase]] = {}
    seen_by_country: dict[str, int] = {}

    baseline_total_points = {horizon: [] for horizon in normalized_horizons}
    adjusted_total_points = {horizon: [] for horizon in normalized_horizons}
    baseline_mechanism_points = {
        horizon: {name: [] for name in normalized_mechanisms}
        for horizon in normalized_horizons
    }
    adjusted_mechanism_points = {
        horizon: {name: [] for name in normalized_mechanisms}
        for horizon in normalized_horizons
    }
    baseline_conditional = {horizon: [] for horizon in normalized_horizons}
    adjusted_conditional = {horizon: [] for horizon in normalized_horizons}

    baseline_total_snapshot_errors: list[float] = []
    adjusted_total_snapshot_errors: list[float] = []
    baseline_mechanism_snapshot_errors: list[float] = []
    adjusted_mechanism_snapshot_errors: list[float] = []
    baseline_conservation_errors: list[float] = []
    adjusted_conservation_errors: list[float] = []
    evaluated_ids: list[str] = []
    evaluated_cutoffs: list[datetime] = []

    for snapshot_id, targets in complete_targets:
        total_target = targets[normalized_horizons[0]][None]
        cutoff = total_target.cutoff_at
        while (
            resolved_pointer < len(resolved)
            and resolved[resolved_pointer].resolved_at < cutoff
        ):
            historical = resolved[resolved_pointer]
            history_by_domain.setdefault(historical.domain, []).append(historical)
            resolved_pointer += 1

        country = total_target.tags[0] if total_target.tags else "all"
        seen = seen_by_country.get(country, 0)
        seen_by_country[country] = seen + 1
        if seen % target_stride:
            continue

        reference_features = total_target.features
        total_outcomes: list[bool] = []
        mechanism_outcomes: dict[int, dict[str, bool]] = {}
        for horizon in normalized_horizons:
            bucket = targets[horizon]
            total_case = bucket[None]
            if total_case.features != reference_features:
                raise ValueError(f"snapshot {snapshot_id} has inconsistent features")
            outcomes: dict[str, bool] = {}
            for name in normalized_mechanisms:
                mechanism_case = bucket[name]
                if mechanism_case.features != reference_features:
                    raise ValueError(
                        f"snapshot {snapshot_id} has inconsistent features"
                    )
                outcomes[name] = bool(mechanism_case.outcome)
            total_outcome = bool(total_case.outcome)
            if sum(outcomes.values()) != int(total_outcome):
                raise ValueError(
                    f"snapshot {snapshot_id} horizon {horizon} has invalid "
                    "mutually exclusive labels"
                )
            total_outcomes.append(total_outcome)
            mechanism_outcomes[horizon] = outcomes

        if [int(value) for value in total_outcomes] != sorted(
            int(value) for value in total_outcomes
        ):
            raise ValueError(f"snapshot {snapshot_id} has non-monotone total labels")
        for name in normalized_mechanisms:
            values = [
                int(mechanism_outcomes[horizon][name])
                for horizon in normalized_horizons
            ]
            if values != sorted(values):
                raise ValueError(
                    f"snapshot {snapshot_id} has non-monotone {name} labels"
                )

        required_domains = []
        for horizon in normalized_horizons:
            required_domains.append(f"{domain_prefix}_{horizon}d")
            required_domains.extend(
                f"{domain_prefix}_{name}_{horizon}d"
                for name in normalized_mechanisms
            )
        if any(
            len(history_by_domain.get(domain, ())) < minimum_training_cases
            for domain in required_domains
        ):
            continue

        eligible_by_domain = {
            domain: (
                history_by_domain[domain][-max_history:]
                if max_history is not None
                else history_by_domain[domain]
            )
            for domain in required_domains
        }
        baseline_total_raw: dict[int, float] = {}
        adjusted_total_raw: dict[int, float] = {}
        baseline_mechanism_raw = {
            name: {} for name in normalized_mechanisms
        }
        adjusted_mechanism_raw = {
            name: {} for name in normalized_mechanisms
        }
        total_evidence = {}
        mechanism_evidence = {name: {} for name in normalized_mechanisms}

        for horizon in normalized_horizons:
            total_domain = f"{domain_prefix}_{horizon}d"
            total_history = eligible_by_domain[total_domain]
            baseline_total_raw[horizon] = (
                sum(case.outcome for case in total_history) + 1.0
            ) / (len(total_history) + 2.0)
            total_result = model.fit_predict(
                total_history,
                target_features=reference_features,
                target_cutoff=cutoff,
                domain=total_domain,
            )
            adjusted_total_raw[horizon] = total_result.probability
            total_evidence[horizon] = total_result

            for name in normalized_mechanisms:
                domain = f"{domain_prefix}_{name}_{horizon}d"
                history = eligible_by_domain[domain]
                baseline_mechanism_raw[name][horizon] = (
                    sum(case.outcome for case in history) + 1.0
                ) / (len(history) + 2.0)
                result = model.fit_predict(
                    history,
                    target_features=reference_features,
                    target_cutoff=cutoff,
                    domain=domain,
                )
                adjusted_mechanism_raw[name][horizon] = result.probability
                mechanism_evidence[name][horizon] = result

        baseline_curve, _ = coherent_competing_risk_curve(
            baseline_total_raw,
            baseline_mechanism_raw,
        )
        adjusted_curve, _ = coherent_competing_risk_curve(
            adjusted_total_raw,
            adjusted_mechanism_raw,
            total_evidence=total_evidence,
            mechanism_evidence=mechanism_evidence,
        )
        baseline_by_horizon = _curve_by_horizon(baseline_curve)
        adjusted_by_horizon = _curve_by_horizon(adjusted_curve)
        baseline_total_errors: list[float] = []
        adjusted_total_errors: list[float] = []
        baseline_mechanism_errors: list[float] = []
        adjusted_mechanism_errors: list[float] = []

        for horizon in normalized_horizons:
            total_outcome = bool(targets[horizon][None].outcome)
            training_cases = len(
                eligible_by_domain[f"{domain_prefix}_{horizon}d"]
            )
            baseline_point = baseline_by_horizon[horizon]
            adjusted_point = adjusted_by_horizon[horizon]
            baseline_total_points[horizon].append(
                BacktestPoint(
                    case_id=snapshot_id,
                    probability=baseline_point.total_exit_probability,
                    outcome=total_outcome,
                    training_cases=training_cases,
                )
            )
            adjusted_total_points[horizon].append(
                BacktestPoint(
                    case_id=snapshot_id,
                    probability=adjusted_point.total_exit_probability,
                    outcome=total_outcome,
                    training_cases=training_cases,
                )
            )
            total_value = float(total_outcome)
            baseline_total_errors.append(
                (baseline_point.total_exit_probability - total_value) ** 2
            )
            adjusted_total_errors.append(
                (adjusted_point.total_exit_probability - total_value) ** 2
            )

            baseline_values = _mechanism_probabilities(baseline_point)
            adjusted_values = _mechanism_probabilities(adjusted_point)
            baseline_conservation_errors.append(
                abs(sum(baseline_values.values()) - baseline_point.total_exit_probability)
            )
            adjusted_conservation_errors.append(
                abs(sum(adjusted_values.values()) - adjusted_point.total_exit_probability)
            )
            for name in normalized_mechanisms:
                outcome = mechanism_outcomes[horizon][name]
                domain = f"{domain_prefix}_{name}_{horizon}d"
                mechanism_training_cases = len(eligible_by_domain[domain])
                baseline_probability = baseline_values[name]
                adjusted_probability = adjusted_values[name]
                baseline_mechanism_points[horizon][name].append(
                    BacktestPoint(
                        case_id=snapshot_id,
                        probability=baseline_probability,
                        outcome=outcome,
                        training_cases=mechanism_training_cases,
                    )
                )
                adjusted_mechanism_points[horizon][name].append(
                    BacktestPoint(
                        case_id=snapshot_id,
                        probability=adjusted_probability,
                        outcome=outcome,
                        training_cases=mechanism_training_cases,
                    )
                )
                outcome_value = float(outcome)
                baseline_mechanism_errors.append(
                    (baseline_probability - outcome_value) ** 2
                )
                adjusted_mechanism_errors.append(
                    (adjusted_probability - outcome_value) ** 2
                )

            if total_outcome:
                true_mechanism = next(
                    name
                    for name in normalized_mechanisms
                    if mechanism_outcomes[horizon][name]
                )
                baseline_conditional[horizon].append(
                    _conditional_score(
                        baseline_point,
                        true_mechanism=true_mechanism,
                        mechanisms=normalized_mechanisms,
                    )
                )
                adjusted_conditional[horizon].append(
                    _conditional_score(
                        adjusted_point,
                        true_mechanism=true_mechanism,
                        mechanisms=normalized_mechanisms,
                    )
                )

        baseline_total_snapshot_errors.append(fmean(baseline_total_errors))
        adjusted_total_snapshot_errors.append(fmean(adjusted_total_errors))
        baseline_mechanism_snapshot_errors.append(
            fmean(baseline_mechanism_errors)
        )
        adjusted_mechanism_snapshot_errors.append(
            fmean(adjusted_mechanism_errors)
        )
        evaluated_ids.append(snapshot_id)
        evaluated_cutoffs.append(cutoff)

    if not evaluated_ids:
        raise ValueError("no competing-risk forecasts met the training requirement")

    horizon_reports: list[HorizonCompetingMetrics] = []
    for horizon in normalized_horizons:
        baseline_total = _metrics(baseline_total_points[horizon])
        adjusted_total = _metrics(adjusted_total_points[horizon])
        mechanism_reports: list[MechanismMetrics] = []
        baseline_mechanism_briers: list[float] = []
        adjusted_mechanism_briers: list[float] = []
        for name in normalized_mechanisms:
            baseline = _metrics(baseline_mechanism_points[horizon][name])
            adjusted = _metrics(adjusted_mechanism_points[horizon][name])
            baseline_mechanism_briers.append(baseline.brier_score)
            adjusted_mechanism_briers.append(adjusted.brier_score)
            mechanism_reports.append(
                MechanismMetrics(
                    mechanism=name,
                    positives=sum(
                        point.outcome
                        for point in adjusted_mechanism_points[horizon][name]
                    ),
                    baseline=baseline,
                    adjusted_model=adjusted,
                    brier_skill_vs_baseline=_skill(
                        adjusted.brier_score,
                        baseline.brier_score,
                    ),
                )
            )
        baseline_conditional_metrics = _conditional_metrics(
            baseline_conditional[horizon]
        )
        adjusted_conditional_metrics = _conditional_metrics(
            adjusted_conditional[horizon]
        )
        baseline_mean_mechanism = fmean(baseline_mechanism_briers)
        adjusted_mean_mechanism = fmean(adjusted_mechanism_briers)
        horizon_reports.append(
            HorizonCompetingMetrics(
                horizon_days=horizon,
                snapshots=adjusted_total.predictions,
                exits=sum(point.outcome for point in adjusted_total_points[horizon]),
                baseline_total=baseline_total,
                adjusted_total=adjusted_total,
                total_brier_skill_vs_baseline=_skill(
                    adjusted_total.brier_score,
                    baseline_total.brier_score,
                ),
                baseline_mean_mechanism_brier=baseline_mean_mechanism,
                adjusted_mean_mechanism_brier=adjusted_mean_mechanism,
                mechanism_brier_skill_vs_baseline=_skill(
                    adjusted_mean_mechanism,
                    baseline_mean_mechanism,
                ),
                baseline_conditional=baseline_conditional_metrics,
                adjusted_conditional=adjusted_conditional_metrics,
                conditional_brier_skill_vs_baseline=_skill(
                    adjusted_conditional_metrics.brier_score,
                    baseline_conditional_metrics.brier_score,
                ),
                conditional_log_loss_skill_vs_baseline=_skill(
                    adjusted_conditional_metrics.log_loss,
                    baseline_conditional_metrics.log_loss,
                ),
                mechanisms=tuple(mechanism_reports),
            )
        )

    baseline_total_integrated = fmean(baseline_total_snapshot_errors)
    adjusted_total_integrated = fmean(adjusted_total_snapshot_errors)
    baseline_mechanism_integrated = fmean(baseline_mechanism_snapshot_errors)
    adjusted_mechanism_integrated = fmean(adjusted_mechanism_snapshot_errors)
    all_baseline_conditional = [
        value
        for horizon in normalized_horizons
        for value in baseline_conditional[horizon]
    ]
    all_adjusted_conditional = [
        value
        for horizon in normalized_horizons
        for value in adjusted_conditional[horizon]
    ]
    baseline_conditional_metrics = _conditional_metrics(all_baseline_conditional)
    adjusted_conditional_metrics = _conditional_metrics(all_adjusted_conditional)

    return CompetingBenchmarkReport(
        holdout_start=holdout_start.isoformat(),
        horizons=normalized_horizons,
        mechanisms=normalized_mechanisms,
        snapshots=len(evaluated_ids),
        exit_observations=adjusted_conditional_metrics.predictions,
        target_stride=target_stride,
        minimum_training_cases=minimum_training_cases,
        max_history=max_history,
        baseline_integrated_total_brier=baseline_total_integrated,
        adjusted_integrated_total_brier=adjusted_total_integrated,
        total_brier_skill_vs_baseline=_skill(
            adjusted_total_integrated,
            baseline_total_integrated,
        ),
        baseline_integrated_mechanism_brier=baseline_mechanism_integrated,
        adjusted_integrated_mechanism_brier=adjusted_mechanism_integrated,
        mechanism_brier_skill_vs_baseline=_skill(
            adjusted_mechanism_integrated,
            baseline_mechanism_integrated,
        ),
        baseline_conditional_brier=baseline_conditional_metrics.brier_score,
        adjusted_conditional_brier=adjusted_conditional_metrics.brier_score,
        conditional_brier_skill_vs_baseline=_skill(
            adjusted_conditional_metrics.brier_score,
            baseline_conditional_metrics.brier_score,
        ),
        baseline_conditional_log_loss=baseline_conditional_metrics.log_loss,
        adjusted_conditional_log_loss=adjusted_conditional_metrics.log_loss,
        conditional_log_loss_skill_vs_baseline=_skill(
            adjusted_conditional_metrics.log_loss,
            baseline_conditional_metrics.log_loss,
        ),
        baseline_conditional_accuracy=baseline_conditional_metrics.accuracy,
        adjusted_conditional_accuracy=adjusted_conditional_metrics.accuracy,
        max_baseline_conservation_error=max(
            baseline_conservation_errors,
            default=0.0,
        ),
        max_adjusted_conservation_error=max(
            adjusted_conservation_errors,
            default=0.0,
        ),
        first_cutoff=min(evaluated_cutoffs).isoformat(),
        last_cutoff=max(evaluated_cutoffs).isoformat(),
        snapshot_ids=tuple(evaluated_ids),
        horizon_metrics=tuple(horizon_reports),
    )
