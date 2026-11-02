"""Phase 3 integration: the only place that wires
AgentRunResult -> approved mock execution -> independent reality simulator
-> collector -> SQLite together.

This is deliberately the one module allowed to depend on agent.demo_agent
AND on execution/collector/storage at once -- agent.demo_agent itself
never imports collector or reality_simulator, and collector.collector
never imports agent.demo_agent. Keeping that wiring here, instead of in
either of those modules, is what keeps them independently testable.
"""

from __future__ import annotations

from typing import Optional

from agent.demo_agent import AgentRunResult, INITIAL_QUEUE_DEPTH, N_WORKERS, RECONCILIATION_CLOSE
from collector.collector import OutcomeComparison, collect_outcome
from collector.reality_simulator import RealityConfig
from execution.mock_paypal import submit_payout_batch
from storage.database import init_db, insert_agent_decision, insert_mock_batch, insert_prediction


def run_pipeline(
    agent_run_result: AgentRunResult,
    reality_seed: int,
    db_path: str,
    reality_config: Optional[RealityConfig] = None,
) -> OutcomeComparison:
    """Take a completed agent run, and -- only if its final decision was
    approved for execution -- submit it to the mock PayPal service,
    independently simulate what actually happened, compare the two, and
    persist everything. Raises if the action was never approved: an
    unapproved agent action must never reach execution."""
    final_step = agent_run_result.trace[-1]
    if not final_step.agent_decision.approved_for_execution:
        raise RuntimeError(
            "Agent run was not approved for execution; refusing to submit "
            "to the mock PayPal service."
        )

    action = final_step.action
    batch_size = sum(count for _, count in action.release_schedule)

    batch = submit_payout_batch(
        action=action,
        batch_size=batch_size,
        n_workers=N_WORKERS,
        reconciliation_close=RECONCILIATION_CLOSE,
        initial_queue_depth=INITIAL_QUEUE_DEPTH,
    )

    conn = init_db(db_path)
    try:
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
        insert_prediction(
            conn,
            batch_id=batch.batch_id,
            risk_score=final_step.verdict.risk_score,
            confidence=final_step.verdict.confidence,
            expected_impact_fraction=final_step.verdict.expected_impact_fraction,
            expected_impact_count=final_step.verdict.expected_impact_count,
            risk_level=final_step.verdict.risk_level.value,
            decision=final_step.verdict.decision.value,
            n_trials=final_step.mc_result.n_trials,
            master_seed=final_step.mc_result.master_seed,
        )
        insert_agent_decision(
            conn,
            batch_id=batch.batch_id,
            decision=final_step.agent_decision.decision.value,
            approved_for_execution=final_step.agent_decision.approved_for_execution,
            rationale=final_step.agent_decision.rationale,
        )
        conn.commit()

        comparison = collect_outcome(
            batch_id=batch.batch_id,
            predicted_risk_score=final_step.verdict.risk_score,
            predicted_expected_impact_fraction=final_step.verdict.expected_impact_fraction,
            shortfall_threshold=final_step.mc_result.shortfall_threshold,
            reality_seed=reality_seed,
            conn=conn,
            reality_config=reality_config,
        )
        conn.commit()
        return comparison
    finally:
        conn.close()
