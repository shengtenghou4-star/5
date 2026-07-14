from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from typing import Iterable

from fencha.datasets.parlgov import CabinetRecord
from fencha.datasets.parlgov_survival import (
    DEFAULT_HORIZONS,
    build_leader_survival_cases,
    normalize_horizons,
)
from fencha.models import HistoricalCase

POST_ELECTION_TRANSITION = "post_election_transition"
OTHER_RECORDED_TRANSITION = "other_recorded_transition"
DEFAULT_MECHANISMS: tuple[str, ...] = (
    POST_ELECTION_TRANSITION,
    OTHER_RECORDED_TRANSITION,
)
M3_COMPETING_BUILDER_VERSION = "parlgov-leader-competing-v2"


@dataclass(frozen=True, slots=True)
class LeaderSpell:
    country_code: str
    country_name: str
    leader_name: str
    start_date: date
    exit_date: date | None
    transition_mechanism: str | None
    transition_election_date: date | None


def classify_transition_mechanism(
    *,
    exit_date: date,
    next_cabinet: CabinetRecord | None,
    election_window_days: int = 120,
) -> tuple[str, date | None]:
    """Classify an observable transition channel without claiming a hidden cause."""
    if election_window_days < 0:
        raise ValueError("election_window_days must be non-negative")
    election_date = next_cabinet.election_date if next_cabinet else None
    if election_date is not None:
        lag = (exit_date - election_date).days
        if 0 <= lag <= election_window_days:
            return POST_ELECTION_TRANSITION, election_date
    return OTHER_RECORDED_TRANSITION, election_date


def build_leader_spells(
    cabinets: Iterable[CabinetRecord],
    *,
    as_of: date,
    election_window_days: int = 120,
) -> tuple[LeaderSpell, ...]:
    if election_window_days < 0:
        raise ValueError("election_window_days must be non-negative")
    by_country: dict[str, list[CabinetRecord]] = {}
    for cabinet in cabinets:
        if cabinet.start_date <= as_of:
            by_country.setdefault(cabinet.country_code, []).append(cabinet)

    spells: list[LeaderSpell] = []
    for country_code, country_cabinets in sorted(by_country.items()):
        country_cabinets.sort(key=lambda item: (item.start_date, item.cabinet_id))
        groups: list[list[CabinetRecord]] = []
        for cabinet in country_cabinets:
            if (
                groups
                and groups[-1][-1].leader_name.casefold().strip()
                == cabinet.leader_name.casefold().strip()
            ):
                groups[-1].append(cabinet)
            else:
                groups.append([cabinet])

        for index, group in enumerate(groups):
            next_cabinet = groups[index + 1][0] if index + 1 < len(groups) else None
            exit_date = next_cabinet.start_date if next_cabinet else None
            mechanism: str | None = None
            election_date: date | None = None
            if exit_date is not None:
                mechanism, election_date = classify_transition_mechanism(
                    exit_date=exit_date,
                    next_cabinet=next_cabinet,
                    election_window_days=election_window_days,
                )
            spells.append(
                LeaderSpell(
                    country_code=country_code,
                    country_name=group[0].country_name,
                    leader_name=group[0].leader_name,
                    start_date=group[0].start_date,
                    exit_date=exit_date,
                    transition_mechanism=mechanism,
                    transition_election_date=election_date,
                )
            )
    return tuple(spells)


def _case_identity(case: HistoricalCase) -> tuple[str, str, date, int]:
    country_code = case.tags[0]
    prefix = f"parlgov:{country_code}:"
    if not case.case_id.startswith(prefix):
        raise ValueError(f"unsupported ParlGov case id: {case.case_id}")
    leader_name, cutoff_text, horizon_text = case.case_id[len(prefix) :].rsplit(":", 2)
    if not horizon_text.endswith("d"):
        raise ValueError(f"unsupported horizon suffix: {case.case_id}")
    return country_code, leader_name, date.fromisoformat(cutoff_text), int(horizon_text[:-1])


def _matching_spell(
    spells: tuple[LeaderSpell, ...],
    *,
    country_code: str,
    leader_name: str,
    cutoff: date,
    as_of: date,
) -> LeaderSpell:
    active_end_default = as_of + timedelta(days=1)
    matches = [
        spell
        for spell in spells
        if spell.country_code == country_code
        and spell.leader_name.casefold().strip() == leader_name.casefold().strip()
        and spell.start_date <= cutoff < (spell.exit_date or active_end_default)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one leader spell for {country_code}/{leader_name}/{cutoff}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _build_mechanism_cases(
    total_cases: Iterable[HistoricalCase],
    *,
    spells: tuple[LeaderSpell, ...],
    as_of: date,
) -> list[HistoricalCase]:
    result: list[HistoricalCase] = []
    descriptions = {
        POST_ELECTION_TRANSITION: "a transition following a recorded election",
        OTHER_RECORDED_TRANSITION: "another recorded transition channel",
    }
    for total_case in total_cases:
        country_code, leader_name, cutoff, horizon = _case_identity(total_case)
        spell = _matching_spell(
            spells,
            country_code=country_code,
            leader_name=leader_name,
            cutoff=cutoff,
            as_of=as_of,
        )
        for mechanism in DEFAULT_MECHANISMS:
            result.append(
                replace(
                    total_case,
                    case_id=f"{total_case.case_id}:{mechanism}",
                    domain=f"government_leader_exit_{mechanism}_{horizon}d",
                    question=(
                        f"Will {leader_name}, head of government of {spell.country_name}, "
                        f"leave office through {descriptions[mechanism]} within "
                        f"{horizon} days?"
                    ),
                    outcome=(
                        total_case.outcome
                        and spell.transition_mechanism == mechanism
                    ),
                    tags=(
                        country_code,
                        "parlgov",
                        M3_COMPETING_BUILDER_VERSION,
                        f"horizon:{horizon}",
                        f"mechanism:{mechanism}",
                        f"total_exit:{int(total_case.outcome)}",
                    ),
                )
            )
    result.sort(key=lambda item: (item.cutoff_at, item.case_id))
    return result


def build_leader_competing_risk_cases(
    cabinets: Iterable[CabinetRecord],
    *,
    as_of: date,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    earliest_cutoff: date | None = None,
    election_window_days: int = 120,
) -> list[HistoricalCase]:
    """Build mutually exclusive cause-specific labels alongside the total exit task."""
    frozen_cabinets = tuple(cabinets)
    normalized_horizons = normalize_horizons(horizons)
    total_cases = build_leader_survival_cases(
        frozen_cabinets,
        as_of=as_of,
        horizons=normalized_horizons,
        earliest_cutoff=earliest_cutoff,
    )
    spells = build_leader_spells(
        frozen_cabinets,
        as_of=as_of,
        election_window_days=election_window_days,
    )
    return _build_mechanism_cases(total_cases, spells=spells, as_of=as_of)


def build_m3_competing_dataset(
    cabinets: Iterable[CabinetRecord],
    *,
    as_of: date,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    earliest_cutoff: date | None = None,
    election_window_days: int = 120,
) -> list[HistoricalCase]:
    """Return the total survival labels and all cause-specific labels together."""
    frozen_cabinets = tuple(cabinets)
    normalized_horizons = normalize_horizons(horizons)
    total_cases = build_leader_survival_cases(
        frozen_cabinets,
        as_of=as_of,
        horizons=normalized_horizons,
        earliest_cutoff=earliest_cutoff,
    )
    spells = build_leader_spells(
        frozen_cabinets,
        as_of=as_of,
        election_window_days=election_window_days,
    )
    cases = list(total_cases)
    cases.extend(_build_mechanism_cases(total_cases, spells=spells, as_of=as_of))
    cases.sort(key=lambda item: (item.cutoff_at, item.case_id))
    return cases
