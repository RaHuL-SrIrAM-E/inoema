import pytest

from agent.demo_agent import AgentAction
from collector.collector import collect_outcome
from execution.mock_paypal import submit_payout_batch
from storage.database import init_db


def _submit_test_batch():
    action = AgentAction(label="t", release_schedule=((0.0, 50),), rationale="t")
    return submit_payout_batch(
        action, batch_size=50, n_workers=4, reconciliation_close=60.0, initial_queue_depth=10
    )


def test_collector_retrieves_outcome_without_agent_involvement(tmp_path):
    # No agent.demo_agent object touches this test at all -- the collector
    # only ever sees a batch_id and a few plain predicted numbers.
    batch = _submit_test_batch()
    conn = init_db(str(tmp_path / "collector.db"))
    comparison = collect_outcome(
        batch_id=batch.batch_id,
        predicted_risk_score=0.02,
        predicted_expected_impact_fraction=0.01,
        shortfall_threshold=0.05,
        reality_seed=3,
        conn=conn,
    )
    assert comparison.batch_id == batch.batch_id
    assert isinstance(comparison.actual_missed_fraction, float)
    conn.close()


def test_collector_unknown_batch_raises(tmp_path):
    conn = init_db(str(tmp_path / "collector_unknown.db"))
    with pytest.raises(ValueError):
        collect_outcome(
            batch_id="does-not-exist",
            predicted_risk_score=0.0,
            predicted_expected_impact_fraction=0.0,
            shortfall_threshold=0.05,
            reality_seed=1,
            conn=conn,
        )
    conn.close()


def test_duplicate_collection_does_not_create_duplicate_outcomes(tmp_path):
    batch = _submit_test_batch()
    conn = init_db(str(tmp_path / "collector_dup.db"))
    first = collect_outcome(
        batch_id=batch.batch_id,
        predicted_risk_score=0.02,
        predicted_expected_impact_fraction=0.01,
        shortfall_threshold=0.05,
        reality_seed=5,
        conn=conn,
    )
    # Even a different reality_seed on the second call must not regenerate
    # or duplicate the already-recorded outcome.
    second = collect_outcome(
        batch_id=batch.batch_id,
        predicted_risk_score=0.02,
        predicted_expected_impact_fraction=0.01,
        shortfall_threshold=0.05,
        reality_seed=999,
        conn=conn,
    )
    assert first == second
    rows = conn.execute(
        "SELECT COUNT(*) FROM outcome_comparisons WHERE batch_id = ?", (batch.batch_id,)
    ).fetchone()[0]
    assert rows == 1
    conn.close()
