import pytest

from agent.demo_agent import AgentConfig, run_agent
from collector.collector import collect_outcome
from pipeline import run_pipeline
from storage.database import init_db


def test_unapproved_agent_action_cannot_execute(tmp_path):
    result = run_agent(AgentConfig(master_seed=0, n_trials=5), batch_size=2000)  # too few trials -> ABSTAIN
    assert result.final_decision.approved_for_execution is False

    db_path = str(tmp_path / "pipeline.db")
    with pytest.raises(RuntimeError):
        run_pipeline(result, reality_seed=1, db_path=db_path)

    conn = init_db(db_path)
    count = conn.execute("SELECT COUNT(*) FROM mock_batches").fetchone()[0]
    conn.close()
    assert count == 0


def test_approved_pipeline_persists_predictions_and_outcomes(tmp_path):
    result = run_agent(AgentConfig(master_seed=0, n_trials=150), batch_size=2000)
    assert result.final_decision.approved_for_execution is True

    db_path = str(tmp_path / "pipeline.db")
    comparison = run_pipeline(result, reality_seed=11, db_path=db_path)

    conn = init_db(db_path)
    batch_row = conn.execute("SELECT * FROM mock_batches").fetchone()
    prediction_row = conn.execute("SELECT * FROM predictions").fetchone()
    decision_row = conn.execute("SELECT * FROM agent_decisions").fetchone()
    outcome_row = conn.execute("SELECT * FROM actual_outcomes").fetchone()
    comparison_row = conn.execute("SELECT * FROM outcome_comparisons").fetchone()
    conn.close()

    assert batch_row is not None
    assert prediction_row is not None
    assert decision_row is not None
    assert outcome_row is not None
    assert comparison_row is not None
    assert comparison_row["batch_id"] == batch_row["batch_id"]
    assert comparison.batch_id == batch_row["batch_id"]
    assert prediction_row["decision"] == "PROCEED"
    assert decision_row["approved_for_execution"] == 1


def test_pipeline_duplicate_collection_is_idempotent(tmp_path):
    result = run_agent(AgentConfig(master_seed=0, n_trials=150), batch_size=2000)
    db_path = str(tmp_path / "pipeline.db")

    first = run_pipeline(result, reality_seed=11, db_path=db_path)

    conn = init_db(db_path)
    second = collect_outcome(
        batch_id=first.batch_id,
        predicted_risk_score=first.predicted_risk_score,
        predicted_expected_impact_fraction=first.predicted_expected_impact_fraction,
        shortfall_threshold=0.05,
        reality_seed=999,
        conn=conn,
    )
    rows = conn.execute(
        "SELECT COUNT(*) FROM outcome_comparisons WHERE batch_id = ?", (first.batch_id,)
    ).fetchone()[0]
    conn.close()

    assert second == first
    assert rows == 1
