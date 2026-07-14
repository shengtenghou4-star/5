from __future__ import annotations

from datetime import datetime
from typing import Iterable

from .conditioned_paths import BinaryForecaster, is_realized_exit_case, mechanism_horizon
from .engine import ForecastResult
from .models import FeatureValue, HistoricalCase


def exit_event_id(case: HistoricalCase, *, horizon_days: int) -> str:
    """Return one stable ID for all snapshots of the same observed leader exit."""
    suffix = f":{horizon_days}d"
    prefix = case.case_id
    marker = suffix + ":"
    if marker in prefix:
        prefix = prefix.rsplit(marker, 1)[0]
    elif prefix.endswith(suffix):
        prefix = prefix[: -len(suffix)]
    # ParlGov snapshot IDs end in the forecast cutoff. Removing it leaves the
    # country and leader spell identity. The actual resolution timestamp keeps
    # separate spells apart if the same leader later returns to office.
    leader_spell = prefix.rsplit(":", 1)[0] if ":" in prefix else prefix
    return f"{leader_spell}:{case.resolved_at.isoformat()}"


def select_event_representatives(
    history: Iterable[HistoricalCase],
    *,
    horizon_days: int,
    target_cutoff: datetime,
    domain: str | None,
) -> tuple[HistoricalCase, ...]:
    """Keep one earliest-in-window snapshot for each resolved exit event."""
    selected: dict[str, HistoricalCase] = {}
    for case in history:
        if case.resolved_at >= target_cutoff:
            continue
        if domain is not None and case.domain != domain:
            continue
        if not is_realized_exit_case(case, horizon_days=horizon_days):
            continue
        event_id = exit_event_id(case, horizon_days=horizon_days)
        current = selected.get(event_id)
        if current is None:
            selected[event_id] = case
            continue
        lead = (case.resolved_at - case.cutoff_at).total_seconds()
        current_lead = (current.resolved_at - current.cutoff_at).total_seconds()
        if (lead, -case.cutoff_at.timestamp(), case.case_id) > (
            current_lead,
            -current.cutoff_at.timestamp(),
            current.case_id,
        ):
            selected[event_id] = case
    return tuple(
        sorted(selected.values(), key=lambda item: (item.resolved_at, item.case_id))
    )


class EventBalancedPathForecaster:
    """Estimate path shares from one representative snapshot per exit event.

    Repeated monthly snapshots of one political transition must not count as
    independent historical transitions. Each resolved event contributes once,
    using the earliest observed snapshot that already lies inside the requested
    forecast horizon.
    """

    def __init__(
        self,
        base_model: BinaryForecaster,
        *,
        domain_prefix: str = "government_leader_exit",
    ) -> None:
        normalized = domain_prefix.strip().strip("_")
        if not normalized:
            raise ValueError("domain_prefix must be non-empty")
        self.base_model = base_model
        self.domain_prefix = normalized

    def fit_predict(
        self,
        history: Iterable[HistoricalCase],
        *,
        target_features: dict[str, FeatureValue],
        target_cutoff: datetime,
        domain: str | None = None,
    ) -> ForecastResult:
        horizon = mechanism_horizon(domain, domain_prefix=self.domain_prefix)
        representatives = select_event_representatives(
            history,
            horizon_days=horizon,
            target_cutoff=target_cutoff,
            domain=domain,
        )
        return self.base_model.fit_predict(
            representatives,
            target_features=target_features,
            target_cutoff=target_cutoff,
            domain=domain,
        )
