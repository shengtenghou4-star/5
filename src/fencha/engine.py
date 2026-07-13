from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from math import exp
from statistics import fmean
from typing import Iterable

from .models import FeatureValue, HistoricalCase, ensure_aware


@dataclass(frozen=True, slots=True)
class Neighbor:
    case_id: str
    similarity: float
    outcome: bool
    cutoff_at: datetime


@dataclass(frozen=True, slots=True)
class ForecastResult:
    probability: float
    prior_probability: float
    effective_sample_size: float
    neighbors: tuple[Neighbor, ...]
    feature_coverage: float


class AnalogForecaster:
    """Weighted historical-analogy forecaster for binary outcomes.

    The model deliberately stays simple in v0.1. Its value is that every
    probability is inspectable and time-safe. More complex learners can later
    be evaluated against this transparent baseline.
    """

    def __init__(
        self,
        *,
        feature_weights: dict[str, float] | None = None,
        prior_strength: float = 4.0,
        top_k: int = 20,
        minimum_similarity: float = 0.05,
    ) -> None:
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        self.feature_weights = feature_weights or {}
        self.prior_strength = prior_strength
        self.top_k = top_k
        self.minimum_similarity = minimum_similarity

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
            if case.resolved_at < target_cutoff and (domain is None or case.domain == domain)
        ]
        if not eligible:
            return ForecastResult(
                probability=0.5,
                prior_probability=0.5,
                effective_sample_size=0.0,
                neighbors=(),
                feature_coverage=0.0,
            )

        prior_probability = fmean(1.0 if case.outcome else 0.0 for case in eligible)
        scales = self._numeric_scales(eligible, target_features)
        scored: list[tuple[float, float, HistoricalCase]] = []
        for case in eligible:
            similarity, coverage = self._similarity(
                case.features, target_features, scales
            )
            if similarity >= self.minimum_similarity:
                scored.append((similarity, coverage, case))

        scored.sort(key=lambda item: (item[0], item[2].resolved_at), reverse=True)
        selected = scored[: self.top_k]
        if not selected:
            return ForecastResult(
                probability=prior_probability,
                prior_probability=prior_probability,
                effective_sample_size=0.0,
                neighbors=(),
                feature_coverage=0.0,
            )

        weighted_yes = sum(
            similarity * (1.0 if case.outcome else 0.0)
            for similarity, _, case in selected
        )
        total_weight = sum(similarity for similarity, _, _ in selected)
        posterior = (
            prior_probability * self.prior_strength + weighted_yes
        ) / (self.prior_strength + total_weight)
        coverages = [coverage for _, coverage, _ in selected]

        neighbors = tuple(
            Neighbor(
                case_id=case.case_id,
                similarity=round(similarity, 6),
                outcome=case.outcome,
                cutoff_at=case.cutoff_at,
            )
            for similarity, _, case in selected
        )
        return ForecastResult(
            probability=min(0.999, max(0.001, posterior)),
            prior_probability=prior_probability,
            effective_sample_size=total_weight,
            neighbors=neighbors,
            feature_coverage=fmean(coverages),
        )

    def _numeric_scales(
        self,
        history: list[HistoricalCase],
        target: dict[str, FeatureValue],
    ) -> dict[str, float]:
        scales: dict[str, float] = {}
        for name, target_feature in target.items():
            if target_feature.kind != "numeric":
                continue
            values = [
                float(case.features[name].value)
                for case in history
                if name in case.features and case.features[name].kind == "numeric"
            ]
            values.append(float(target_feature.value))
            span = max(values) - min(values) if len(values) > 1 else 0.0
            scales[name] = max(span, 1.0)
        return scales

    def _similarity(
        self,
        left: dict[str, FeatureValue],
        right: dict[str, FeatureValue],
        scales: dict[str, float],
    ) -> tuple[float, float]:
        shared = sorted(set(left) & set(right))
        if not shared:
            return 0.0, 0.0

        weighted_similarity = 0.0
        used_weight = 0.0
        possible_weight = sum(
            self.feature_weights.get(name, 1.0) for name in right
        )
        for name in shared:
            a = left[name]
            b = right[name]
            if a.kind != b.kind:
                continue
            weight = self.feature_weights.get(name, 1.0)
            if weight <= 0:
                continue
            if a.kind == "numeric":
                distance = abs(float(a.value) - float(b.value)) / scales.get(name, 1.0)
                score = exp(-3.0 * distance)
            else:
                score = 1.0 if a.value == b.value else 0.0
            weighted_similarity += weight * score
            used_weight += weight

        if used_weight == 0:
            return 0.0, 0.0
        coverage = used_weight / possible_weight if possible_weight else 0.0
        return (weighted_similarity / used_weight) * coverage, coverage
