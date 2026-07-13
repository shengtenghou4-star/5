from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass(frozen=True, slots=True)
class ForecastRevision:
    revision_id: str
    question_id: str
    created_at: str
    probability: float
    rationale: dict[str, Any]


class ForecastLedger:
    """Append-only SQLite ledger for questions, revisions and resolutions."""

    def __init__(self, path: str | Path = "fencha.db") -> None:
        self.path = Path(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS questions (
                    question_id TEXT PRIMARY KEY,
                    domain TEXT NOT NULL,
                    question TEXT NOT NULL,
                    cutoff_at TEXT NOT NULL,
                    resolution_rule TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS revisions (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    revision_id TEXT UNIQUE NOT NULL,
                    question_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    probability REAL NOT NULL CHECK(probability > 0 AND probability < 1),
                    rationale_json TEXT NOT NULL,
                    FOREIGN KEY(question_id) REFERENCES questions(question_id)
                );
                CREATE TABLE IF NOT EXISTS resolutions (
                    question_id TEXT PRIMARY KEY,
                    resolved_at TEXT NOT NULL,
                    outcome INTEGER NOT NULL CHECK(outcome IN (0, 1)),
                    evidence TEXT NOT NULL,
                    FOREIGN KEY(question_id) REFERENCES questions(question_id)
                );
                """
            )

    def create_question(
        self,
        *,
        question_id: str,
        domain: str,
        question: str,
        cutoff_at: datetime,
        resolution_rule: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO questions
                (question_id, domain, question, cutoff_at, resolution_rule, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    domain,
                    question,
                    cutoff_at.isoformat(),
                    resolution_rule,
                    now,
                ),
            )

    def append_revision(
        self,
        *,
        question_id: str,
        probability: float,
        rationale: dict[str, Any],
    ) -> ForecastRevision:
        if not 0.0 < probability < 1.0:
            raise ValueError("probability must be strictly between 0 and 1")
        revision = ForecastRevision(
            revision_id=str(uuid4()),
            question_id=question_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            probability=probability,
            rationale=rationale,
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO revisions
                (revision_id, question_id, created_at, probability, rationale_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    revision.revision_id,
                    revision.question_id,
                    revision.created_at,
                    revision.probability,
                    json.dumps(revision.rationale, ensure_ascii=False, sort_keys=True),
                ),
            )
        return revision

    def resolve(
        self,
        *,
        question_id: str,
        outcome: bool,
        evidence: str,
        resolved_at: datetime | None = None,
    ) -> None:
        resolved_at = resolved_at or datetime.now(timezone.utc)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO resolutions (question_id, resolved_at, outcome, evidence)
                VALUES (?, ?, ?, ?)
                """,
                (question_id, resolved_at.isoformat(), int(outcome), evidence),
            )

    def history(self, question_id: str) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT revision_id, question_id, created_at, probability, rationale_json
                FROM revisions WHERE question_id = ? ORDER BY sequence ASC
                """,
                (question_id,),
            ).fetchall()
        return [
            {
                "revision_id": row["revision_id"],
                "question_id": row["question_id"],
                "created_at": row["created_at"],
                "probability": row["probability"],
                "rationale": json.loads(row["rationale_json"]),
            }
            for row in rows
        ]
