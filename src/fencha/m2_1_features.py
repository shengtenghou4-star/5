from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from math import log1p
from statistics import median, quantiles

from .models import FeatureValue, HistoricalCase

M21_FEATURE_VERSION = "m2.1-time-safe-news-volume-v1"

_VOLUME_FEATURES: dict[str, tuple[str, str]] = {
    "gdelt_events_per_sample_30d": (
        "gdelt_log_events_per_sample_30d",
        "gdelt_anomaly_log_events_per_sample_30d",
    ),
    "gdelt_articles_per_sample_30d": (
        "gdelt_log_articles_per_sample_30d",
        "gdelt_anomaly_log_articles_per_sample_30d",
    ),
    "gdelt_events_per_sample_90d": (
        "gdelt_log_events_per_sample_90d",
        "gdelt_anomaly_log_events_per_sample_90d",
    ),
    "gdelt_articles_per_sample_90d": (
        "gdelt_log_articles_per_sample_90d",
        "gdelt_anomaly_log_articles_per_sample_90d",
    ),
}


def _robust_scale(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    q1, _, q3 = quantiles(values, n=4, method="inclusive")
    iqr = q3 - q1
    if iqr > 1e-12:
        return iqr
    span = max(values) - min(values)
    return span if span > 1e-12 else 1.0


def add_time_safe_volume_features(
    cases: list[HistoricalCase],
    *,
    minimum_history: int = 6,
    anomaly_clip: float = 5.0,
) -> list[HistoricalCase]:
    """Add log-volume and within-country anomaly features without future leakage.

    Each anomaly compares the current log-volume with earlier cutoffs from the
    same country. Cases sharing one cutoff are transformed as a batch, so they
    cannot leak into one another. Outcomes and resolution dates are never used.
    """
    if minimum_history <= 0:
        raise ValueError("minimum_history must be positive")
    if anomaly_clip <= 0:
        raise ValueError("anomaly_clip must be positive")

    by_country: dict[str, list[HistoricalCase]] = defaultdict(list)
    for case in cases:
        country = case.tags[0] if case.tags else "all"
        by_country[country].append(case)

    transformed: list[HistoricalCase] = []
    for country_cases in by_country.values():
        country_cases.sort(key=lambda case: (case.cutoff_at, case.case_id))
        histories: dict[str, list[float]] = defaultdict(list)
        cursor = 0
        while cursor < len(country_cases):
            cutoff = country_cases[cursor].cutoff_at
            end = cursor
            while end < len(country_cases) and country_cases[end].cutoff_at == cutoff:
                end += 1
            group = country_cases[cursor:end]
            observed_logs: list[dict[str, float]] = []

            for case in group:
                features = dict(case.features)
                current_logs: dict[str, float] = {}
                for raw_name, (log_name, anomaly_name) in _VOLUME_FEATURES.items():
                    raw = features.get(raw_name)
                    if raw is None or raw.kind != "numeric":
                        continue
                    logged = log1p(max(0.0, float(raw.value)))
                    current_logs[log_name] = logged
                    features[log_name] = FeatureValue(
                        logged,
                        case.cutoff_at,
                        "numeric",
                    )
                    history = histories[log_name]
                    if len(history) >= minimum_history:
                        anomaly = (logged - median(history)) / _robust_scale(history)
                        anomaly = max(-anomaly_clip, min(anomaly_clip, anomaly))
                        features[anomaly_name] = FeatureValue(
                            anomaly,
                            case.cutoff_at,
                            "numeric",
                        )

                tags = case.tags
                if M21_FEATURE_VERSION not in tags:
                    tags = tags + (M21_FEATURE_VERSION,)
                transformed.append(replace(case, features=features, tags=tags))
                observed_logs.append(current_logs)

            # Update histories only after every case at this cutoff was scored.
            for current_logs in observed_logs:
                for log_name, value in current_logs.items():
                    histories[log_name].append(value)
            cursor = end

    transformed.sort(key=lambda case: (case.cutoff_at, case.case_id))
    return transformed
