from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from .engine import ForecastResult
from .models import FeatureValue, HistoricalCase, ensure_aware


@dataclass(frozen=True, slots=True)
class HierarchicalRiskConfig:
    use_country: bool = True
    use_tenure: bool = True
    use_context: bool = False
    tenure_bucket_days: int = 365
    country_strength: float = 100.0
    tenure_strength: float = 100.0
    context_strength: float = 100.0
    global_alpha: float = 1.0
    global_beta: float = 1.0

    def __post_init__(self) -> None:
        if self.tenure_bucket_days <= 0:
            raise ValueError("tenure_bucket_days must be positive")
        for name in (
            "country_strength",
            "tenure_strength",
            "context_strength",
            "global_alpha",
            "global_beta",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


class HierarchicalRiskForecaster:
    """Transparent empirical-Bayes risk model for rare binary events.

    The model starts from a smoothed global rate and optionally updates through
    increasingly specific, nested groups. Each local rate is shrunk toward its
    parent rate, so sparse countries or tenure cells cannot create extreme
    probabilities from a handful of outcomes.
    """

    CONTEXT_FEATURES = (
        "caretaker",
        "minority_government",
        "cabinet_type",
    )

    def __init__(self, config: HierarchicalRiskConfig | None = None) -> None:
        self.config = config or HierarchicalRiskConfig()

    @staticmethod
    def _value(
        features: dict[str, FeatureValue],
        name: str,
    ) -> object | None:
        feature = features.get(name)
        return feature.value if feature is not None else None

    def _tenure_bucket(self, features: dict[str, FeatureValue]) -> int | None:
        value = self._value(features, "tenure_days")
        if value is None:
            return None
        return max(0, int(float(value)) // self.config.tenure_bucket_days)

    @staticmethod
    def _posterior(
        cases: list[HistoricalCase],
        *,
        parent_probability: float,
        strength: float,
    ) -> float:
        if not cases:
            return parent_probability
        positives = sum(case.outcome for case in cases)
        return (positives + strength * parent_probability) / (
            len(cases) + strength
        )

    @classmethod
    def _context_key(
        cls,
        features: dict[str, FeatureValue],
    ) -> tuple[tuple[str, object], ...]:
        return tuple(
            (name, features[name].value)
            for name in cls.CONTEXT_FEATURES
            if name in features
        )

    @staticmethod
    def _matches_context(
        case: HistoricalCase,
        key: tuple[tuple[str, object], ...],
    ) -> bool:
        return all(
            name in case.features and case.features[name].value == value
            for name, value in key
        )

    def fit_predict(
        self,
        history: Iterable[HistoricalCase],
        *,
        target_features: dict[str, FeatureValue],
        target_cutoff: datetime,
        domain: str | None = None,
    ) -> ForecastResult:
        target_cutoff = ensure_aware(target_cutoff)
        leaked = [
            name
            for name, feature in target_features.items()
            if feature.observed_at > target_cutoff
        ]
        if leaked:
            raise ValueError(
                "target contains future information: " + ", ".join(sorted(leaked))
            )

        eligible = [
            case
            for case in history
            if case.resolved_at < target_cutoff
            and (domain is None or case.domain == domain)
        ]
        if not eligible:
            return ForecastResult(
                probability=0.5,
                prior_probability=0.5,
                effective_sample_size=0.0,
                neighbors=(),
                feature_coverage=0.0,
            )

        config = self.config
        positives = sum(case.outcome for case in eligible)
        global_probability = (positives + config.global_alpha) / (
            len(eligible) + config.global_alpha + config.global_beta
        )
        probability = global_probability
        current_pool = eligible
        used_levels = 0
        possible_levels = int(config.use_country) + int(config.use_tenure) + int(
            config.use_context
        )
        effective_sample_size = float(len(eligible))

        country = self._value(target_features, "country_code")
        if config.use_country and country is not None:
            country_pool = [
                case
                for case in current_pool
                if self._value(case.features, "country_code") == country
            ]
            if country_pool:
                probability = self._posterior(
                    country_pool,
                    parent_probability=probability,
                    strength=config.country_strength,
                )
                current_pool = country_pool
                effective_sample_size = float(len(country_pool))
                used_levels += 1

        target_tenure_bucket = self._tenure_bucket(target_features)
        if config.use_tenure and target_tenure_bucket is not None:
            tenure_pool = [
                case
                for case in current_pool
                if self._tenure_bucket(case.features) == target_tenure_bucket
            ]
            if tenure_pool:
                probability = self._posterior(
                    tenure_pool,
                    parent_probability=probability,
                    strength=config.tenure_strength,
                )
                current_pool = tenure_pool
                effective_sample_size = float(len(tenure_pool))
                used_levels += 1

        context_key = self._context_key(target_features)
        if config.use_context and context_key:
            context_pool = [
                case for case in current_pool if self._matches_context(case, context_key)
            ]
            if context_pool:
                probability = self._posterior(
                    context_pool,
                    parent_probability=probability,
                    strength=config.context_strength,
                )
                effective_sample_size = float(len(context_pool))
                used_levels += 1

        coverage = used_levels / possible_levels if possible_levels else 1.0
        return ForecastResult(
            probability=min(0.999, max(0.001, probability)),
            prior_probability=global_probability,
            effective_sample_size=effective_sample_size,
            neighbors=(),
            feature_coverage=coverage,
        )
