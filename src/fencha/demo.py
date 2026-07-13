from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sqlite3

from .engine import AnalogForecaster
from .ledger import ForecastLedger
from .models import FeatureValue, HistoricalCase
from .scoring import walk_forward_backtest

UTC = timezone.utc


def dt(year: int, month: int = 1, day: int = 1) -> datetime:
    return datetime(year, month, day, tzinfo=UTC)


def feature(value: float | str | bool, at: datetime) -> FeatureValue:
    if isinstance(value, bool):
        kind = "boolean"
    elif isinstance(value, (int, float)):
        kind = "numeric"
    else:
        kind = "categorical"
    return FeatureValue(value=value, observed_at=at, kind=kind)


def demo_cases() -> list[HistoricalCase]:
    """Illustrative cases for exercising the engine; not a research dataset."""
    raw = [
        ("w01", 2010, 0.72, "tight", True, True),
        ("w02", 2011, 0.45, "loose", False, False),
        ("w03", 2012, 0.64, "tight", True, True),
        ("w04", 2013, 0.31, "loose", False, False),
        ("w05", 2014, 0.58, "mixed", True, True),
        ("w06", 2015, 0.52, "mixed", False, False),
        ("w07", 2016, 0.77, "tight", True, True),
        ("w08", 2017, 0.40, "loose", True, False),
        ("w09", 2018, 0.69, "tight", False, True),
        ("w10", 2019, 0.55, "mixed", True, True),
    ]
    cases: list[HistoricalCase] = []
    for case_id, year, pressure, coalition, trigger, outcome in raw:
        cutoff = dt(year, 3, 1)
        cases.append(
            HistoricalCase(
                case_id=case_id,
                domain="world_demo",
                question="Will the tracked event occur within twelve months?",
                cutoff_at=cutoff,
                resolved_at=dt(year + 1, 3, 2),
                outcome=outcome,
                features={
                    "pressure_index": feature(pressure, cutoff),
                    "coalition_structure": feature(coalition, cutoff),
                    "recent_trigger": feature(trigger, cutoff),
                },
            )
        )
    return cases


def run_demo(database_path: str | Path = "fencha.db") -> str:
    cases = demo_cases()
    model = AnalogForecaster(
        feature_weights={
            "pressure_index": 2.0,
            "coalition_structure": 1.0,
            "recent_trigger": 1.5,
        },
        prior_strength=3.0,
        top_k=6,
    )
    report = walk_forward_backtest(cases, model, minimum_training_cases=3)

    cutoff = dt(2022, 3, 1)
    target = {
        "pressure_index": feature(0.66, cutoff),
        "coalition_structure": feature("tight", cutoff),
        "recent_trigger": feature(True, cutoff),
    }
    prediction = model.fit_predict(
        cases,
        target_features=target,
        target_cutoff=cutoff,
        domain="world_demo",
    )

    ledger = ForecastLedger(database_path)
    question_id = "demo-world-2022"
    try:
        ledger.create_question(
            question_id=question_id,
            domain="world_demo",
            question="Will the demonstration event occur within twelve months?",
            cutoff_at=cutoff,
            resolution_rule="Resolve YES if the defined event occurs before 2023-03-01 UTC.",
        )
    except sqlite3.IntegrityError:
        # Re-running the demo must never overwrite history; a duplicate question is fine.
        pass
    ledger.append_revision(
        question_id=question_id,
        probability=prediction.probability,
        rationale={
            "prior_probability": prediction.prior_probability,
            "effective_sample_size": prediction.effective_sample_size,
            "feature_coverage": prediction.feature_coverage,
            "neighbors": [
                {
                    "case_id": neighbor.case_id,
                    "similarity": neighbor.similarity,
                    "outcome": neighbor.outcome,
                }
                for neighbor in prediction.neighbors
            ],
        },
    )

    return (
        f"walk-forward points={len(report.points)} "
        f"brier={report.brier_score:.4f} "
        f"log_loss={report.log_loss:.4f} "
        f"calibration_error={report.calibration_error:.4f}\n"
        f"new forecast={prediction.probability:.3f} "
        f"prior={prediction.prior_probability:.3f} "
        f"neighbors={len(prediction.neighbors)} "
        f"ledger={Path(database_path)}"
    )
