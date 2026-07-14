from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Iterable

from fencha.datasets.parlgov import CabinetRecord, build_leader_exit_cases
from fencha.models import HistoricalCase

DEFAULT_HORIZONS: tuple[int, ...] = (30, 90, 180, 365)
M3_BUILDER_VERSION = "parlgov-leader-survival-v1"


def normalize_horizons(horizons: Iterable[int]) -> tuple[int, ...]:
    """Return sorted, unique, positive forecast horizons."""
    values = tuple(sorted(set(horizons)))
    if not values:
        raise ValueError("at least one horizon is required")
    if any(value <= 0 for value in values):
        raise ValueError("horizons must be positive")
    return values


def build_leader_survival_cases(
    cabinets: Iterable[CabinetRecord],
    *,
    as_of: date,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    earliest_cutoff: date | None = None,
) -> list[HistoricalCase]:
    """Build one time-safe binary case per leader, cutoff, and horizon.

    The existing ParlGov builder remains the single source of truth for leader
    spells and structural features. This M3 adapter expands every monthly
    snapshot into several coherent forecasting horizons while preserving the
    original right-censoring rules.
    """
    frozen_cabinets = tuple(cabinets)
    normalized = normalize_horizons(horizons)
    result: list[HistoricalCase] = []

    for horizon_days in normalized:
        horizon_cases = build_leader_exit_cases(
            frozen_cabinets,
            as_of=as_of,
            horizon_days=horizon_days,
            earliest_cutoff=earliest_cutoff,
        )
        for case in horizon_cases:
            question = case.question.replace(
                "within 180 days?", f"within {horizon_days} days?"
            )
            result.append(
                replace(
                    case,
                    case_id=f"{case.case_id}:{horizon_days}d",
                    domain=f"government_leader_exit_{horizon_days}d",
                    question=question,
                    tags=(
                        case.tags[0],
                        "parlgov",
                        M3_BUILDER_VERSION,
                        f"horizon:{horizon_days}",
                    ),
                )
            )

    result.sort(key=lambda item: (item.cutoff_at, item.case_id))
    return result
