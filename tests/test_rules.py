from engine.montecarlo import run_monte_carlo
from engine.rules import Decision, RiskLevel, evaluate, MIN_TRIALS_FOR_CONFIDENCE
from tests.conftest import make_config


def test_evaluate_proceeds_on_safe_scenario():
    config = make_config(release_schedule=[(0.0, 2000)])
    mc_result = run_monte_carlo(config, n_trials=150, master_seed=1)
    verdict = evaluate(mc_result, config)
    assert verdict.decision == Decision.PROCEED
    assert verdict.risk_level == RiskLevel.LOW
    assert not verdict.reasons


def test_evaluate_flags_adjust_on_risky_scenario():
    config = make_config(release_schedule=[(140.0, 2000)])
    mc_result = run_monte_carlo(config, n_trials=150, master_seed=1)
    verdict = evaluate(mc_result, config)
    assert verdict.decision == Decision.ADJUST
    assert verdict.risk_level in (RiskLevel.MEDIUM, RiskLevel.HIGH)


def test_evaluate_abstains_on_too_few_trials():
    config = make_config()
    mc_result = run_monte_carlo(config, n_trials=MIN_TRIALS_FOR_CONFIDENCE - 1, master_seed=1)
    verdict = evaluate(mc_result, config)
    assert verdict.decision == Decision.ABSTAIN
    assert verdict.risk_level == RiskLevel.UNKNOWN
    assert verdict.reasons


def test_evaluate_abstains_on_invalid_config():
    config = make_config(release_schedule=[(90.0, 1999)])  # does not sum to batch_size
    mc_result = run_monte_carlo(make_config(), n_trials=150, master_seed=1)
    verdict = evaluate(mc_result, config)
    assert verdict.decision == Decision.ABSTAIN
    assert any("release_schedule" in reason for reason in verdict.reasons)


def test_risk_confidence_and_impact_are_independent_fields():
    config = make_config(release_schedule=[(140.0, 2000)])
    mc_result = run_monte_carlo(config, n_trials=150, master_seed=1)
    verdict = evaluate(mc_result, config)
    # They must be distinct attributes that can vary independently -- not
    # a single blended score.
    assert isinstance(verdict.risk_score, float)
    assert isinstance(verdict.confidence, float)
    assert isinstance(verdict.expected_impact_fraction, float)
    assert isinstance(verdict.expected_impact_count, float)
    assert verdict.expected_impact_count == verdict.expected_impact_fraction * config.batch_size
