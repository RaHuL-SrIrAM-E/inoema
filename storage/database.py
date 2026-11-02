"""SQLite persistence for INOEMA. Every row written here is SYNTHETIC /
MOCKED -- nothing in this project ever persists data from a real financial
system.

Deliberately primitive-typed: every function takes and returns plain
values (str/int/float/bool/dict), never dataclasses from other modules.
This keeps storage dependency-free (no imports from engine/agent/
execution/collector), so nothing else in the project can create an import
cycle through it.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone

MOCKED_LABEL = "SYNTHETIC_MOCKED_INOEMA"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS mock_batches (
    batch_id TEXT PRIMARY KEY,
    batch_size INTEGER NOT NULL,
    release_schedule_json TEXT NOT NULL,
    n_workers INTEGER NOT NULL,
    reconciliation_close REAL NOT NULL,
    initial_queue_depth INTEGER NOT NULL,
    submitted_at TEXT NOT NULL,
    status TEXT NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 1,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES mock_batches(batch_id),
    risk_score REAL NOT NULL,
    confidence REAL NOT NULL,
    expected_impact_fraction REAL NOT NULL,
    expected_impact_count REAL NOT NULL,
    risk_level TEXT NOT NULL,
    decision TEXT NOT NULL,
    n_trials INTEGER NOT NULL,
    master_seed INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 1,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_decisions (
    decision_id TEXT PRIMARY KEY,
    batch_id TEXT NOT NULL REFERENCES mock_batches(batch_id),
    decision TEXT NOT NULL,
    approved_for_execution INTEGER NOT NULL,
    rationale TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS actual_outcomes (
    batch_id TEXT PRIMARY KEY REFERENCES mock_batches(batch_id),
    missed_fraction REAL NOT NULL,
    n_completed_by_close INTEGER NOT NULL,
    reality_seed INTEGER NOT NULL,
    day_slowdown_multiplier REAL NOT NULL,
    created_at TEXT NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 1,
    label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS outcome_comparisons (
    batch_id TEXT PRIMARY KEY REFERENCES mock_batches(batch_id),
    predicted_risk_score REAL NOT NULL,
    predicted_expected_impact_fraction REAL NOT NULL,
    actual_missed_fraction REAL NOT NULL,
    actual_shortfall INTEGER NOT NULL,
    prediction_error REAL NOT NULL,
    brier_component REAL NOT NULL,
    reality_seed INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    synthetic INTEGER NOT NULL DEFAULT 1,
    label TEXT NOT NULL
);
"""


def init_db(db_path: str) -> sqlite3.Connection:
    """Open (creating if needed) the INOEMA database and ensure its schema
    exists. Safe to call repeatedly: every statement is CREATE TABLE IF NOT
    EXISTS, so re-running this against an existing file is a no-op beyond
    opening the connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def insert_mock_batch(
    conn: sqlite3.Connection,
    batch_id: str,
    batch_size: int,
    release_schedule: list[tuple[float, int]] | tuple[tuple[float, int], ...],
    n_workers: int,
    reconciliation_close: float,
    initial_queue_depth: int,
    submitted_at: str,
    status: str,
) -> None:
    conn.execute(
        """INSERT OR IGNORE INTO mock_batches
           (batch_id, batch_size, release_schedule_json, n_workers,
            reconciliation_close, initial_queue_depth, submitted_at, status,
            synthetic, label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
        (
            batch_id,
            batch_size,
            json.dumps([list(pair) for pair in release_schedule]),
            n_workers,
            reconciliation_close,
            initial_queue_depth,
            submitted_at,
            status,
            MOCKED_LABEL,
        ),
    )


def insert_prediction(
    conn: sqlite3.Connection,
    batch_id: str,
    risk_score: float,
    confidence: float,
    expected_impact_fraction: float,
    expected_impact_count: float,
    risk_level: str,
    decision: str,
    n_trials: int,
    master_seed: int,
) -> str:
    prediction_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO predictions
           (prediction_id, batch_id, risk_score, confidence,
            expected_impact_fraction, expected_impact_count, risk_level,
            decision, n_trials, master_seed, created_at, synthetic, label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
        (
            prediction_id,
            batch_id,
            risk_score,
            confidence,
            expected_impact_fraction,
            expected_impact_count,
            risk_level,
            decision,
            n_trials,
            master_seed,
            _now(),
            MOCKED_LABEL,
        ),
    )
    return prediction_id


def insert_agent_decision(
    conn: sqlite3.Connection,
    batch_id: str,
    decision: str,
    approved_for_execution: bool,
    rationale: str,
) -> str:
    decision_id = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO agent_decisions
           (decision_id, batch_id, decision, approved_for_execution, rationale, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (decision_id, batch_id, decision, int(approved_for_execution), rationale, _now()),
    )
    return decision_id


def has_outcome(conn: sqlite3.Connection, batch_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM outcome_comparisons WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    return row is not None


def insert_outcome(
    conn: sqlite3.Connection,
    batch_id: str,
    missed_fraction: float,
    n_completed_by_close: int,
    reality_seed: int,
    day_slowdown_multiplier: float,
    predicted_risk_score: float,
    predicted_expected_impact_fraction: float,
    actual_shortfall: bool,
    prediction_error: float,
    brier_component: float,
) -> None:
    """Idempotent: INSERT OR IGNORE keyed on batch_id, so collecting the
    same batch's outcome twice never creates duplicate rows."""
    now = _now()
    conn.execute(
        """INSERT OR IGNORE INTO actual_outcomes
           (batch_id, missed_fraction, n_completed_by_close, reality_seed,
            day_slowdown_multiplier, created_at, synthetic, label)
           VALUES (?, ?, ?, ?, ?, ?, 1, ?)""",
        (
            batch_id,
            missed_fraction,
            n_completed_by_close,
            reality_seed,
            day_slowdown_multiplier,
            now,
            MOCKED_LABEL,
        ),
    )
    conn.execute(
        """INSERT OR IGNORE INTO outcome_comparisons
           (batch_id, predicted_risk_score, predicted_expected_impact_fraction,
            actual_missed_fraction, actual_shortfall, prediction_error,
            brier_component, reality_seed, created_at, synthetic, label)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)""",
        (
            batch_id,
            predicted_risk_score,
            predicted_expected_impact_fraction,
            missed_fraction,
            int(actual_shortfall),
            prediction_error,
            brier_component,
            reality_seed,
            now,
            MOCKED_LABEL,
        ),
    )


def fetch_outcome_comparison(conn: sqlite3.Connection, batch_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM outcome_comparisons WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    return dict(row) if row is not None else None


def fetch_mock_batch(conn: sqlite3.Connection, batch_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM mock_batches WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    return dict(row) if row is not None else None


def fetch_prediction(conn: sqlite3.Connection, batch_id: str) -> dict | None:
    row = conn.execute(
        "SELECT * FROM predictions WHERE batch_id = ?", (batch_id,)
    ).fetchone()
    return dict(row) if row is not None else None
