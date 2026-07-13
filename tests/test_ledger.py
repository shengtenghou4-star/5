from datetime import datetime, timezone

import pytest

from fencha.ledger import ForecastLedger


def test_ledger_preserves_revisions(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path / "ledger.db")
    ledger.create_question(
        question_id="q1",
        domain="career",
        question="Will the offer arrive?",
        cutoff_at=datetime(2027, 1, 1, tzinfo=timezone.utc),
        resolution_rule="YES if a signed offer is received by the cutoff.",
    )
    ledger.append_revision(question_id="q1", probability=0.4, rationale={"base": 0.4})
    ledger.append_revision(
        question_id="q1", probability=0.65, rationale={"signal": "interview"}
    )
    history = ledger.history("q1")
    assert [row["probability"] for row in history] == [0.4, 0.65]


def test_ledger_rejects_invalid_probability(tmp_path) -> None:
    ledger = ForecastLedger(tmp_path / "ledger.db")
    with pytest.raises(ValueError):
        ledger.append_revision(question_id="missing", probability=1.0, rationale={})
