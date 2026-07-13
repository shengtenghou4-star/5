from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

FeatureKind = Literal["numeric", "categorical", "boolean"]


def ensure_aware(value: datetime) -> datetime:
    """Return a timezone-aware datetime, treating naive values as UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


@dataclass(frozen=True, slots=True)
class FeatureValue:
    """A feature value together with the moment it became observable."""

    value: float | str | bool
    observed_at: datetime
    kind: FeatureKind

    def __post_init__(self) -> None:
        object.__setattr__(self, "observed_at", ensure_aware(self.observed_at))
        if self.kind == "numeric" and isinstance(self.value, bool):
            raise TypeError("boolean values cannot be numeric features")
        if self.kind == "numeric" and not isinstance(self.value, (int, float)):
            raise TypeError("numeric features require int or float values")
        if self.kind == "categorical" and not isinstance(self.value, str):
            raise TypeError("categorical features require string values")
        if self.kind == "boolean" and not isinstance(self.value, bool):
            raise TypeError("boolean features require bool values")


@dataclass(frozen=True, slots=True)
class HistoricalCase:
    """A resolved binary forecasting case frozen at its prediction cutoff."""

    case_id: str
    domain: str
    question: str
    cutoff_at: datetime
    resolved_at: datetime
    outcome: bool
    features: dict[str, FeatureValue]
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        object.__setattr__(self, "cutoff_at", ensure_aware(self.cutoff_at))
        object.__setattr__(self, "resolved_at", ensure_aware(self.resolved_at))
        if self.resolved_at <= self.cutoff_at:
            raise ValueError("resolved_at must be later than cutoff_at")
        leaked = [
            name
            for name, feature in self.features.items()
            if feature.observed_at > self.cutoff_at
        ]
        if leaked:
            names = ", ".join(sorted(leaked))
            raise ValueError(f"future information leakage in features: {names}")

    def plain_features(self) -> dict[str, Any]:
        return {name: feature.value for name, feature in self.features.items()}
