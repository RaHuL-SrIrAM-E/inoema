from agent.demo_agent import AgentConfig, run_agent
from engine.rules import Decision


def test_risky_initial_proposal_leads_to_adjustment():
    result = run_agent(AgentConfig(master_seed=0, n_trials=150), batch_size=2000)
    first_step = result.trace[0]
    assert first_step.action.label == "initial_full_batch_release"
    assert first_step.verdict.decision == Decision.ADJUST


def test_agent_performs_a_second_simulation_after_adjustment():
    result = run_agent(AgentConfig(master_seed=0, n_trials=150), batch_size=2000)
    assert len(result.trace) == 2
    assert result.trace[1].action.label == "adjusted_split_release"


def test_revised_safe_proposal_leads_to_approval():
    result = run_agent(AgentConfig(master_seed=0, n_trials=150), batch_size=2000)
    second_step = result.trace[1]
    assert second_step.verdict.decision == Decision.PROCEED
    assert result.final_decision.decision == Decision.PROCEED
    assert result.final_decision.approved_for_execution is True


def test_abstention_prevents_approval():
    # Too few trials for engine.rules.evaluate to trust the estimate.
    result = run_agent(AgentConfig(master_seed=0, n_trials=5), batch_size=2000)
    assert result.final_decision.decision == Decision.ABSTAIN
    assert result.final_decision.approved_for_execution is False


def test_still_risky_revised_proposal_does_not_get_approved():
    # A batch large enough to overwhelm even the split proposal.
    result = run_agent(AgentConfig(master_seed=0, n_trials=150), batch_size=20000)
    assert len(result.trace) == 2
    assert result.trace[1].verdict.decision == Decision.ADJUST
    assert result.final_decision.approved_for_execution is False


def test_execution_trace_contains_both_simulation_calls():
    result = run_agent(AgentConfig(master_seed=0, n_trials=150), batch_size=2000)
    assert [step.step_index for step in result.trace] == [0, 1]
    assert result.trace[0].mc_result is not result.trace[1].mc_result
    assert result.trace[0].mc_result.master_seed != result.trace[1].mc_result.master_seed


def test_agent_run_is_reproducible():
    config = AgentConfig(master_seed=42, n_trials=150)
    result_a = run_agent(config, batch_size=2000)
    result_b = run_agent(config, batch_size=2000)

    assert len(result_a.trace) == len(result_b.trace)
    for step_a, step_b in zip(result_a.trace, result_b.trace):
        assert step_a.mc_result.missed_fractions == step_b.mc_result.missed_fractions
        assert step_a.verdict.risk_score == step_b.verdict.risk_score
        assert step_a.agent_decision.approved_for_execution == step_b.agent_decision.approved_for_execution

    assert result_a.final_decision.decision == result_b.final_decision.decision
    assert result_a.final_decision.approved_for_execution == result_b.final_decision.approved_for_execution
