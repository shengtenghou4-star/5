from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .models import FeatureValue, HistoricalCase


def case_from_dict(payload: dict[str, Any]) -> HistoricalCase:
    """Reconstruct a validated HistoricalCase from its serialized form."""
    raw_features = payload.get("features")
    if not isinstance(raw_features, dict):
        raise TypeError("case features must be an object")
    features: dict[str, FeatureValue] = {}
    for name, item in raw_features.items():
        if not isinstance(item, dict):
            raise TypeError(f"feature {name!r} must be an object")
        features[str(name)] = FeatureValue(
            value=item["value"],
            kind=item["kind"],
            observed_at=datetime.fromisoformat(str(item["observed_at"])),
        )
    return HistoricalCase(
        case_id=str(payload["case_id"]),
        domain=str(payload["domain"]),
        question=str(payload["question"]),
        cutoff_at=datetime.fromisoformat(str(payload["cutoff_at"])),
        resolved_at=datetime.fromisoformat(str(payload["resolved_at"])),
        outcome=bool(payload["outcome"]),
        features=features,
        tags=tuple(str(value) for value in payload.get("tags", [])),
    )


def read_jsonl(path: str | Path) -> list[HistoricalCase]:
    """Read and validate an append-friendly JSONL historical case table."""
    cases: list[HistoricalCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise TypeError("case row must be an object")
                cases.append(case_from_dict(payload))
            except Exception as exc:
                raise ValueError(f"invalid historical case at line {line_number}") from exc
    return cases
