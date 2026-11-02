"""Independent outcome collector.

Retrieves the mocked execution record from execution.mock_paypal, runs (or
retrieves an already-stored) independent reality outcome, and compares it
against the original prediction. Deliberately does NOT import
agent.demo_agent and does NOT accept an agent-reported outcome as input --
the only sources of truth for "what actually happened" are
execution.mock_paypal's own records and collector.reality_simulator.

All data is SYNTHETIC / MOCKED. Never fabricates accuracy metrics: the
Brier component and prediction error here are plain arithmetic over
genuine simulation output, not invented numbers.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import Optional

from collector.reality_simulator import RealityConfig, simulate_reality
from execution.mock_paypal import get_batch
from storage.database import MOCKED_LABEL, fetch_outcome_comparison, insert_mock_batch, insert_outcome

LABEL = MOCKED_LABEL


@dataclass(frozen=True)
class OutcomeComparison:
    """Prediction vs. actual, for one mocked batch. Synthetic data only."""

    batch_id: str
    predicted_risk_score: float
    predicted_expected_impact_fraction: float
    actual_missed_fraction: float
    actual_shortfall: bool
    prediction_error: float
    brier_component: float
    reality_seed: int
    synthetic: bool = field(default=True)
    label: str = field(default=LABEL)


def collect_outcome(
    batch_id: str,
    predicted_risk_score: float,
    predicted_expected_impact_fraction: float,
    shortfall_threshold: float,
    reality_seed: int,
    conn: sqlite3.Connection,
    reality_config: Optional[RealityConfig] = None,
) -> OutcomeComparison:
    """Collect (or re-fetch, if already collected) the actual outcome for
    a mocked batch and compare it against the original prediction.

    Idempotent: if this batch_id has already been collected, the stored
    comparison is returned as-is and the reality simulator is not run
    again -- "what actually happened" is recorded once, not regenerated
    on every call.
    """
    existing = fetch_outcome_comparison(conn, batch_id)
    if existing is not None:
        return OutcomeComparison(
            batch_id=existing["batch_id"],
            predicted_risk_score=existing["predicted_risk_score"],
            predicted_expected_impact_fraction=existing["predicted_expected_impact_fraction"],
            actual_missed_fraction=existing["actual_missed_fraction"],
            actual_shortfall=bool(existing["actual_shortfall"]),
            prediction_error=existing["prediction_error"],
            brier_component=existing["brier_component"],
            reality_seed=existing["reality_seed"],
        )

    batch = get_batch(batch_id)
    if batch is None:
        raise ValueError(f"No mock execution record found for batch_id={batch_id!r}")

    # Ensure the batch's own row exists before writing outcome rows that
    # reference it by foreign key -- the collector retrieves this record
    # from the execution service itself, so it shouldn't depend on some
    # other caller (e.g. pipeline.run_pipeline) having already persisted
    # it. Idempotent (INSERT OR IGNORE), so this is a no-op if it has.
    insert_mock_batch(
        conn,
        batch_id=batch.batch_id,
        batch_size=batch.batch_size,
        release_schedule=batch.release_schedule,
        n_workers=batch.n_workers,
        reconciliation_close=batch.reconciliation_close,
        initial_queue_depth=batch.initial_queue_depth,
        submitted_at=batch.submitted_at,
        status=batch.status,
    )

    reality_outcome = simulate_reality(
        batch_size=batch.batch_size,
        release_schedule=batch.release_schedule,
        n_workers=batch.n_workers,
        reconciliation_close=batch.reconciliation_close,
        initial_queue_depth=batch.initial_queue_depth,
        reality_seed=reality_seed,
        reality_config=reality_config,
    )

    actual_shortfall = reality_outcome.missed_fraction > shortfall_threshold
    prediction_error = reality_outcome.missed_fraction - predicted_expected_impact_fraction
    brier_component = (predicted_risk_score - (1.0 if actual_shortfall else 0.0)) ** 2

    comparison = OutcomeComparison(
        batch_id=batch_id,
        predicted_risk_score=predicted_risk_score,
        predicted_expected_impact_fraction=predicted_expected_impact_fraction,
        actual_missed_fraction=reality_outcome.missed_fraction,
        actual_shortfall=actual_shortfall,
        prediction_error=prediction_error,
        brier_component=brier_component,
        reality_seed=reality_seed,
    )

    insert_outcome(
        conn,
        batch_id=batch_id,
        missed_fraction=reality_outcome.missed_fraction,
        n_completed_by_close=reality_outcome.n_completed_by_close,
        reality_seed=reality_seed,
        day_slowdown_multiplier=reality_outcome.day_slowdown_multiplier,
        predicted_risk_score=predicted_risk_score,
        predicted_expected_impact_fraction=predicted_expected_impact_fraction,
        actual_shortfall=actual_shortfall,
        prediction_error=prediction_error,
        brier_component=brier_component,
    )
    return comparison
