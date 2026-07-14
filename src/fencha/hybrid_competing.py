from __future__ import annotations

from datetime import datetime
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


class HybridCompetingForecaster:
    """Use separate models for total exit risk and conditional path signals.

    Total domains have the form ``<prefix>_<horizon>d``. Mechanism domains add
    one mechanism token before the horizon. The dispatcher does not inspect
    outcomes or dates; it only routes the already time-filtered domain-specific
    history supplied by the benchmark.
    """

    def __init__(
        self,
        *,
        total_model: BinaryForecaster,
        mechanism_model: BinaryForecaster,
        domain_prefix: str = "government_leader_exit",
    ) -> None:
        normalized = domain_prefix.strip().strip("_")
        if not normalized:
            raise ValueError("domain_prefix must be non-empty")
        self.total_model = total_model
        self.mechanism_model = mechanism_model
        self.domain_prefix = normalized

    def _is_total_domain(self, domain: str | None) -> bool:
        if domain is None:
            raise ValueError("hybrid competing forecasts require an explicit domain")
        prefix = f"{self.domain_prefix}_"
        if not domain.startswith(prefix) or not domain.endswith("d"):
            raise ValueError(f"unsupported competing-risk domain: {domain}")
        body = domain[len(prefix) : -1]
        try:
            horizon = int(body)
        except ValueError:
            return False
        if horizon <= 0:
            raise ValueError("forecast horizon must be positive")
        return True

    def fit_predict(
        self,
        history: Iterable[HistoricalCase],
        *,
        target_features: dict[str, FeatureValue],
        target_cutoff: datetime,
        domain: str | None = None,
    ) -> ForecastResult:
        model = self.total_model if self._is_total_domain(domain) else self.mechanism_model
        return model.fit_predict(
            history,
            target_features=target_features,
            target_cutoff=target_cutoff,
            domain=domain,
        )
