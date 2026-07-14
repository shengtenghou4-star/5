from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from math import log
from random import Random
from statistics import fmean
from typing import Iterable, Mapping

from .conditioned_paths import BinaryForecaster
from .event_balanced_paths import exit_event_id, select_event_representatives
from .models import HistoricalCase, ensure_aware


@dataclass(frozen=True, slots=True)
class PathMetricSummary:
    observations: int
    unique_events: int
    brier_score: float
    log_loss: float
    accuracy: float


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class ModelPathAudit:
    model: str
    metrics: PathMetricSummary
    brier_improvement: float
    log_loss_improvement: float
    accuracy_improvement: float
    brier_improvement_ci95: ConfidenceInterval
    log_loss_improvement_ci95: ConfidenceInterval
    accuracy_improvement_ci95: ConfidenceInterval
    countries_with_brier_improvement: int
    countries_evaluated: int


@dataclass(frozen=True, slots=True)
class HorizonPathAudit:
    horizon_days: int
    records: int
    unique_events: int
    baseline: PathMetricSummary
    models: tuple[ModelPathAudit, ...]


@dataclass(frozen=True, slots=True)
class SelectedPathPrediction:
    event_id: str
    snapshot_id: str
    country: str
    cutoff_at: str
    exit_at: str
    horizon_days: int
    days_to_exit: float
    true_mechanism: str
    baseline_shares: dict[str, float]
    model_shares: dict[str, dict[str, float]]


@dataclass(frozen=True, slots=True)
class EventPathAuditReport:
    holdout_start: str
    horizons: tuple[int, ...]
    mechanisms: tuple[str, ...]
    model_names: tuple[str, ...]
    candidate_exit_predictions: int
    selected_event_horizon_predictions: int
    unique_exit_events: int
    countries: int
    minimum_training_cases: int
    minimum_exit_events: int
    selection_rule: str
    bootstrap_replicates: int
    baseline: PathMetricSummary
    models: tuple[ModelPathAudit, ...]
    horizons_detail: tuple[HorizonPathAudit, ...]
    predictions: tuple[SelectedPathPrediction, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _CandidatePrediction:
    event_id: str
    snapshot_id: str
    country: str
    cutoff_at: datetime
    exit_at: datetime
    horizon_days: int
    days_to_exit: float
    true_mechanism: str
    baseline_shares: dict[str, float]
    model_shares: dict[str, dict[str, float]]


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


def _normalize(values: Mapping[str, float], names: tuple[str, ...]) -> dict[str, float]:
    clipped = {name: max(0.0, float(values.get(name, 0.0))) for name in names}
    total = sum(clipped.values())
    if total <= 1e-15:
        return {name: 1.0 / len(names) for name in names}
    return {name: clipped[name] / total for name in names}


def _score(
    shares: Mapping[str, float],
    *,
    true_mechanism: str,
    mechanisms: tuple[str, ...],
) -> tuple[float, float, float]:
    normalized = _normalize(shares, mechanisms)
    brier = fmean(
        (normalized[name] - float(name == true_mechanism)) ** 2
        for name in mechanisms
    )
    probability = min(
        1.0 - 1e-12,
        max(1e-12, normalized[true_mechanism]),
    )
    predicted = max(mechanisms, key=lambda name: (normalized[name], name))
    return brier, -log(probability), float(predicted == true_mechanism)


def _event_macro_metrics(
    records: Iterable[_CandidatePrediction],
    *,
    mechanisms: tuple[str, ...],
    model_name: str | None,
) -> PathMetricSummary:
    by_event: dict[str, list[tuple[float, float, float]]] = {}
    frozen = tuple(records)
    for record in frozen:
        shares = (
            record.baseline_shares
            if model_name is None
            else record.model_shares[model_name]
        )
        by_event.setdefault(record.event_id, []).append(
            _score(
                shares,
                true_mechanism=record.true_mechanism,
                mechanisms=mechanisms,
            )
        )
    if not by_event:
        return PathMetricSummary(0, 0, 0.0, 0.0, 0.0)
    event_scores = [
        (
            fmean(value[0] for value in values),
            fmean(value[1] for value in values),
            fmean(value[2] for value in values),
        )
        for values in by_event.values()
    ]
    return PathMetricSummary(
        observations=len(frozen),
        unique_events=len(by_event),
        brier_score=fmean(value[0] for value in event_scores),
        log_loss=fmean(value[1] for value in event_scores),
        accuracy=fmean(value[2] for value in event_scores),
    )


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _cluster_intervals(
    records: Iterable[_CandidatePrediction],
    *,
    mechanisms: tuple[str, ...],
    model_name: str,
    replicates: int,
    seed: int,
) -> tuple[ConfidenceInterval, ConfidenceInterval, ConfidenceInterval]:
    by_event: dict[str, list[tuple[tuple[float, float, float], tuple[float, float, float]]]] = {}
    for record in records:
        baseline = _score(
            record.baseline_shares,
            true_mechanism=record.true_mechanism,
            mechanisms=mechanisms,
        )
        model = _score(
            record.model_shares[model_name],
            true_mechanism=record.true_mechanism,
            mechanisms=mechanisms,
        )
        by_event.setdefault(record.event_id, []).append((baseline, model))
    event_deltas = []
    for values in by_event.values():
        event_deltas.append(
            (
                fmean(item[0][0] - item[1][0] for item in values),
                fmean(item[0][1] - item[1][1] for item in values),
                fmean(item[1][2] - item[0][2] for item in values),
            )
        )
    if not event_deltas or replicates <= 0:
        zero = ConfidenceInterval(0.0, 0.0)
        return zero, zero, zero
    rng = Random(seed)
    brier_samples: list[float] = []
    log_samples: list[float] = []
    accuracy_samples: list[float] = []
    count = len(event_deltas)
    for _ in range(replicates):
        sample = [event_deltas[rng.randrange(count)] for _ in range(count)]
        brier_samples.append(fmean(item[0] for item in sample))
        log_samples.append(fmean(item[1] for item in sample))
        accuracy_samples.append(fmean(item[2] for item in sample))
    return (
        ConfidenceInterval(
            _percentile(brier_samples, 0.025),
            _percentile(brier_samples, 0.975),
        ),
        ConfidenceInterval(
            _percentile(log_samples, 0.025),
            _percentile(log_samples, 0.975),
        ),
        ConfidenceInterval(
            _percentile(accuracy_samples, 0.025),
            _percentile(accuracy_samples, 0.975),
        ),
    )


def _country_wins(
    records: Iterable[_CandidatePrediction],
    *,
    mechanisms: tuple[str, ...],
    model_name: str,
) -> tuple[int, int]:
    by_country: dict[str, list[_CandidatePrediction]] = {}
    for record in records:
        by_country.setdefault(record.country, []).append(record)
    wins = 0
    for country_records in by_country.values():
        baseline = _event_macro_metrics(
            country_records,
            mechanisms=mechanisms,
            model_name=None,
        )
        model = _event_macro_metrics(
            country_records,
            mechanisms=mechanisms,
            model_name=model_name,
        )
        wins += model.brier_score < baseline.brier_score
    return wins, len(by_country)


def _model_audit(
    records: tuple[_CandidatePrediction, ...],
    *,
    mechanisms: tuple[str, ...],
    model_name: str,
    baseline: PathMetricSummary,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> ModelPathAudit:
    metrics = _event_macro_metrics(
        records,
        mechanisms=mechanisms,
        model_name=model_name,
    )
    brier_ci, log_ci, accuracy_ci = _cluster_intervals(
        records,
        mechanisms=mechanisms,
        model_name=model_name,
        replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    wins, countries = _country_wins(
        records,
        mechanisms=mechanisms,
        model_name=model_name,
    )
    return ModelPathAudit(
        model=model_name,
        metrics=metrics,
        brier_improvement=baseline.brier_score - metrics.brier_score,
        log_loss_improvement=baseline.log_loss - metrics.log_loss,
        accuracy_improvement=metrics.accuracy - baseline.accuracy,
        brier_improvement_ci95=brier_ci,
        log_loss_improvement_ci95=log_ci,
        accuracy_improvement_ci95=accuracy_ci,
        countries_with_brier_improvement=wins,
        countries_evaluated=countries,
    )


def _select_fixed_lead_records(
    candidates: Iterable[_CandidatePrediction],
) -> tuple[_CandidatePrediction, ...]:
    selected: dict[tuple[str, int], _CandidatePrediction] = {}
    for record in candidates:
        key = (record.event_id, record.horizon_days)
        current = selected.get(key)
        if current is None or (
            record.days_to_exit,
            -record.cutoff_at.timestamp(),
            record.snapshot_id,
        ) > (
            current.days_to_exit,
            -current.cutoff_at.timestamp(),
            current.snapshot_id,
        ):
            selected[key] = record
    return tuple(
        sorted(
            selected.values(),
            key=lambda item: (item.exit_at, item.event_id, item.horizon_days),
        )
    )


def temporal_event_path_audit(
    cases: list[HistoricalCase],
    models: Mapping[str, BinaryForecaster],
    *,
    holdout_start: datetime,
    mechanisms: Iterable[str],
    horizons: Iterable[int] = (30, 90, 180, 365),
    domain_prefix: str = "government_leader_exit",
    minimum_training_cases: int = 500,
    minimum_exit_events: int = 20,
    max_history: int | None = 3000,
    bootstrap_replicates: int = 2000,
    bootstrap_seed: int = 20260715,
) -> EventPathAuditReport:
    """Audit path prediction once per unique exit event at fixed lead windows."""
    holdout_start = ensure_aware(holdout_start)
    normalized_horizons = tuple(sorted(set(horizons)))
    normalized_mechanisms = tuple(sorted(set(item.strip() for item in mechanisms)))
    model_names = tuple(models)
    if not normalized_horizons or any(value <= 0 for value in normalized_horizons):
        raise ValueError("horizons must contain positive values")
    if not normalized_mechanisms or any(not item for item in normalized_mechanisms):
        raise ValueError("mechanisms must contain non-empty names")
    if not model_names or any(not item.strip() for item in model_names):
        raise ValueError("at least one named path model is required")
    if minimum_training_cases <= 0 or minimum_exit_events <= 0:
        raise ValueError("training thresholds must be positive")
    if max_history is not None and max_history <= 0:
        raise ValueError("max_history must be positive when provided")
    if bootstrap_replicates <= 0:
        raise ValueError("bootstrap_replicates must be positive")

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
        snapshot_id = _snapshot_id(case, horizon=horizon, mechanism=mechanism)
        bucket = snapshots.setdefault(snapshot_id, {}).setdefault(horizon, {})
        if mechanism in bucket:
            raise ValueError(
                f"duplicate role for snapshot {snapshot_id} at horizon {horizon}"
            )
        bucket[mechanism] = case

    targets = []
    for snapshot_id, by_horizon in snapshots.items():
        if set(by_horizon) != requested_horizons:
            continue
        if any(set(bucket) != expected_roles for bucket in by_horizon.values()):
            continue
        cutoffs = {
            case.cutoff_at
            for bucket in by_horizon.values()
            for case in bucket.values()
        }
        if len(cutoffs) != 1:
            raise ValueError(f"snapshot {snapshot_id} contains different cutoffs")
        if next(iter(cutoffs)) >= holdout_start:
            targets.append((snapshot_id, by_horizon))
    targets.sort(
        key=lambda item: (
            item[1][normalized_horizons[0]][None].cutoff_at,
            item[0],
        )
    )
    if not targets:
        raise ValueError("no complete path snapshots exist in the holdout")

    resolved = sorted(relevant, key=lambda item: (item.resolved_at, item.case_id))
    resolved_pointer = 0
    history_by_domain: dict[str, list[HistoricalCase]] = {}
    candidates: list[_CandidatePrediction] = []

    for snapshot_id, by_horizon in targets:
        total_reference = by_horizon[normalized_horizons[0]][None]
        cutoff = total_reference.cutoff_at
        while resolved_pointer < len(resolved) and resolved[resolved_pointer].resolved_at < cutoff:
            historical = resolved[resolved_pointer]
            history_by_domain.setdefault(historical.domain, []).append(historical)
            resolved_pointer += 1

        features = total_reference.features
        country = total_reference.tags[0] if total_reference.tags else "all"
        for horizon in normalized_horizons:
            bucket = by_horizon[horizon]
            total_case = bucket[None]
            if total_case.features != features:
                raise ValueError(f"snapshot {snapshot_id} has inconsistent features")
            outcomes = {
                name: bool(bucket[name].outcome) for name in normalized_mechanisms
            }
            if sum(outcomes.values()) != int(bool(total_case.outcome)):
                raise ValueError(
                    f"snapshot {snapshot_id} horizon {horizon} has invalid labels"
                )
            if not total_case.outcome:
                continue

            domains = {
                name: f"{domain_prefix}_{name}_{horizon}d"
                for name in normalized_mechanisms
            }
            if any(
                len(history_by_domain.get(domain, ())) < minimum_training_cases
                for domain in domains.values()
            ):
                continue
            histories = {
                name: (
                    history_by_domain[domain][-max_history:]
                    if max_history is not None
                    else history_by_domain[domain]
                )
                for name, domain in domains.items()
            }
            representative_sets = {
                name: select_event_representatives(
                    histories[name],
                    horizon_days=horizon,
                    target_cutoff=cutoff,
                    domain=domains[name],
                )
                for name in normalized_mechanisms
            }
            event_sets = {
                name: {
                    exit_event_id(case, horizon_days=horizon)
                    for case in representatives
                }
                for name, representatives in representative_sets.items()
            }
            first_events = event_sets[normalized_mechanisms[0]]
            if any(events != first_events for events in event_sets.values()):
                raise ValueError("mechanism histories contain different exit events")
            if len(first_events) < minimum_exit_events:
                continue
            baseline_raw = {
                name: sum(case.outcome for case in representative_sets[name]) + 1.0
                for name in normalized_mechanisms
            }
            baseline_shares = _normalize(baseline_raw, normalized_mechanisms)

            model_shares: dict[str, dict[str, float]] = {}
            for model_name, model in models.items():
                raw: dict[str, float] = {}
                for name in normalized_mechanisms:
                    result = model.fit_predict(
                        histories[name],
                        target_features=features,
                        target_cutoff=cutoff,
                        domain=domains[name],
                    )
                    raw[name] = result.probability
                model_shares[model_name] = _normalize(raw, normalized_mechanisms)

            true_mechanism = next(
                name for name in normalized_mechanisms if outcomes[name]
            )
            candidates.append(
                _CandidatePrediction(
                    event_id=exit_event_id(total_case, horizon_days=horizon),
                    snapshot_id=snapshot_id,
                    country=country,
                    cutoff_at=cutoff,
                    exit_at=total_case.resolved_at,
                    horizon_days=horizon,
                    days_to_exit=(total_case.resolved_at - cutoff).total_seconds()
                    / 86400.0,
                    true_mechanism=true_mechanism,
                    baseline_shares=baseline_shares,
                    model_shares=model_shares,
                )
            )

    selected = _select_fixed_lead_records(candidates)
    if not selected:
        raise ValueError("no event-level path predictions met the training requirements")
    baseline = _event_macro_metrics(
        selected,
        mechanisms=normalized_mechanisms,
        model_name=None,
    )
    model_reports = tuple(
        _model_audit(
            selected,
            mechanisms=normalized_mechanisms,
            model_name=name,
            baseline=baseline,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed + index,
        )
        for index, name in enumerate(model_names)
    )

    horizon_reports = []
    for horizon in normalized_horizons:
        horizon_records = tuple(
            record for record in selected if record.horizon_days == horizon
        )
        horizon_baseline = _event_macro_metrics(
            horizon_records,
            mechanisms=normalized_mechanisms,
            model_name=None,
        )
        horizon_reports.append(
            HorizonPathAudit(
                horizon_days=horizon,
                records=len(horizon_records),
                unique_events=len({item.event_id for item in horizon_records}),
                baseline=horizon_baseline,
                models=tuple(
                    _model_audit(
                        horizon_records,
                        mechanisms=normalized_mechanisms,
                        model_name=name,
                        baseline=horizon_baseline,
                        bootstrap_replicates=bootstrap_replicates,
                        bootstrap_seed=bootstrap_seed + 100 + index + horizon,
                    )
                    for index, name in enumerate(model_names)
                ),
            )
        )

    public_predictions = tuple(
        SelectedPathPrediction(
            event_id=item.event_id,
            snapshot_id=item.snapshot_id,
            country=item.country,
            cutoff_at=item.cutoff_at.isoformat(),
            exit_at=item.exit_at.isoformat(),
            horizon_days=item.horizon_days,
            days_to_exit=item.days_to_exit,
            true_mechanism=item.true_mechanism,
            baseline_shares=item.baseline_shares,
            model_shares=item.model_shares,
        )
        for item in selected
    )
    return EventPathAuditReport(
        holdout_start=holdout_start.isoformat(),
        horizons=normalized_horizons,
        mechanisms=normalized_mechanisms,
        model_names=model_names,
        candidate_exit_predictions=len(candidates),
        selected_event_horizon_predictions=len(selected),
        unique_exit_events=len({item.event_id for item in selected}),
        countries=len({item.country for item in selected}),
        minimum_training_cases=minimum_training_cases,
        minimum_exit_events=minimum_exit_events,
        selection_rule=(
            "one prediction per exit event and horizon, choosing the snapshot "
            "closest to the horizon boundary"
        ),
        bootstrap_replicates=bootstrap_replicates,
        baseline=baseline,
        models=model_reports,
        horizons_detail=tuple(horizon_reports),
        predictions=public_predictions,
    )
