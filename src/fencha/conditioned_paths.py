from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, Protocol

from .engine import ForecastResult
from .models import FeatureValue, HistoricalCase


class BinaryForecaster(Protocol):
    def fit_predict(
        self,
        history: Iterable[HistoricalCase],
        *,
        target_features: dict[str, FeatureValue],
        target_cutoff: datetime,
        domain: str | None = None,
    ) -> ForecastResult: ...


def mechanism_horizon(domain: str | None, *, domain_prefix: str) -> int:
    if domain is None:
        raise ValueError("exit-conditioned path forecasts require an explicit domain")
    prefix = f"{domain_prefix}_"
    if not domain.startswith(prefix) or not domain.endswith("d"):
        raise ValueError(f"unsupported mechanism domain: {domain}")
    body = domain[len(prefix) : -1]
    if "_" not in body:
        raise ValueError(f"total-exit domain is not a mechanism domain: {domain}")
    _mechanism, horizon_text = body.rsplit("_", 1)
    try:
        horizon = int(horizon_text)
    except ValueError as exc:
        raise ValueError(f"unsupported mechanism domain: {domain}") from exc
    if horizon <= 0:
        raise ValueError("forecast horizon must be positive")
    return horizon


def _tagged_total_exit(case: HistoricalCase) -> bool | None:
    values = [tag.split(":", 1)[1] for tag in case.tags if tag.startswith("total_exit:")]
    if not values:
        return None
    if len(values) != 1 or values[0] not in {"0", "1"}:
        raise ValueError(f"invalid total_exit tag on case {case.case_id}")
    total_exit = values[0] == "1"
    if case.outcome and not total_exit:
        raise ValueError(
            f"positive mechanism case cannot have total_exit:0: {case.case_id}"
        )
    return total_exit


def is_realized_exit_case(case: HistoricalCase, *, horizon_days: int) -> bool:
    """Identify mechanism labels whose total-exit event occurred in the horizon.

    New competing-risk datasets carry an explicit outcome-only ``total_exit`` tag.
    That tag correctly distinguishes an exit exactly on the horizon boundary from
    a no-exit case that also resolves on the boundary. Older datasets fall back to
    the conservative resolution-time rule for reproducibility.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    tagged = _tagged_total_exit(case)
    if tagged is not None:
        return tagged
    if case.outcome:
        return True
    horizon_end = case.cutoff_at + timedelta(days=horizon_days)
    return case.resolved_at < horizon_end


class ExitConditionedPathForecaster:
    """Train path probabilities only on observations where an exit occurred.

    The wrapped binary model still predicts one mechanism versus the other
    mechanisms. Non-exit observations are removed after they have resolved, so
    the model estimates ``P(path | exit within horizon)`` rather than forcing the
    extremely rare total-exit base rate into every path classifier.
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
        frozen = tuple(history)
        exits = tuple(
            case
            for case in frozen
            if case.resolved_at < target_cutoff
            and (domain is None or case.domain == domain)
            and is_realized_exit_case(case, horizon_days=horizon)
        )
        return self.base_model.fit_predict(
            exits,
            target_features=target_features,
            target_cutoff=target_cutoff,
            domain=domain,
        )
