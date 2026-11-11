"""Behavioural tests for the payment-recovery simulation (engine/payments, engine/objectives).

Exact-timeline tests build the latent world by hand (LatentOutcomes) so every
expected number is derivable from the model rules, independent of any RNG.
"""
import math
from dataclasses import replace
from datetime import datetime, timezone

import numpy as np
import pytest

from engine.montecarlo import wilson_interval
from engine.objectives import (HardConstraints, ObjectiveWeights, PolicyEvaluation, evaluate_policy, paired_difference,
                               rank_policies, score_trial, static_constraint_violations)
from engine.payments.distributions import AmountDistribution, AmountKind
from engine.payments.failure_model import (CATEGORIES, DEFAULT_CLASSIFICATION, MAX_RETRY_DRAWS, FailureModel,
                                           FailureRules, LatentOutcomes, RecoveryClass)
from engine.payments.metrics import aggregate_trials, compute_trial_metrics, mean_ci
from engine.payments.models import (CostModel, EventType as E, FailureCategory as FC, FinalStatus,
                                    PaymentMethodCategory as PM, PaymentStatus, RecoveryStatus, Transaction)
from engine.payments.recovery_policy import (ActionKind, AlternateFundingAction, RecoveryPolicy, TerminalFailureHandling,
                                             buyer_action_eligible, exponential_backoff, immediate_retry,
                                             intelligent_recovery, is_retry_eligible, next_action, no_recovery,
                                             resolve_policy)
from engine.payments.scenarios import PaymentScenario, build_scenario
from engine.payments.simulator import (compare_policies_monte_carlo, run_payment_monte_carlo, simulate_trial,
                                       trace_transaction, trial_seeds)
from engine.payments.transaction_generator import (Workload, WorkloadConfig, generate_workload,
                                                   instantiate_transactions, prepare_trial)

RULES = FailureRules()


# ---------------------------------------------------------------- helpers
def one_txn(category, *, clear=0.0, retry_u=0.0, buyer_resp=0.0, buyer_time=0.5, post=0.0, dup=0.0,
            failed=True, offset=10.0):
    """A single-transaction world. u-values: success iff u < probability, so 0.0 => 'lucky', 0.999 => 'unlucky'."""
    cfg = WorkloadConfig(n_transactions=1)
    wl = Workload(cfg, np.array([100.0]), np.array([0]), np.array([offset]))
    lat = LatentOutcomes(np.array([failed]), np.array([CATEGORIES.index(category)]), np.array([clear]),
                         np.array([buyer_resp]), np.array([buyer_time]), np.array([post]), np.array([dup]),
                         np.full((1, MAX_RETRY_DRAWS), retry_u))
    return PaymentScenario("t", cfg), wl, lat


def run(category, policy, **kw):
    sc, wl, lat = one_txn(category, **kw)
    r = simulate_trial(sc, policy, wl, lat, record_events=True)
    return r.transactions[0], [(round(e.time_minutes, 3), e.type) for e in r.log.events[0]], r.log.events[0]


# ---------------------------------------------------------------- generation
def test_generation_is_seed_deterministic_and_seed_sensitive():
    cfg = WorkloadConfig(n_transactions=500)
    a = generate_workload(cfg, np.random.SeedSequence(1))
    b = generate_workload(cfg, np.random.SeedSequence(1))
    c = generate_workload(cfg, np.random.SeedSequence(2))
    assert np.array_equal(a.amounts, b.amounts) and np.array_equal(a.offsets, b.offsets)
    assert not np.array_equal(a.amounts, c.amounts)


def test_workload_properties():
    cfg = WorkloadConfig(n_transactions=20_000, amount=AmountDistribution(low=5.0, high=300.0))
    w = generate_workload(cfg, np.random.SeedSequence(0))
    assert len(w.amounts) == 20_000
    assert w.amounts.min() >= 5.0 and w.amounts.max() <= 300.0
    assert np.allclose(w.amounts, np.round(w.amounts, 2))
    assert np.all(np.diff(w.offsets) >= 0) and w.offsets.min() >= 0 and w.offsets.max() <= cfg.horizon_minutes
    share_card = np.mean(w.method_idx == 0)  # METHODS[0] is CARD, mix 0.55
    assert abs(share_card - 0.55) < 0.015


def test_amount_distribution_kinds():
    rng = np.random.default_rng(0)
    assert set(AmountDistribution(kind=AmountKind.FIXED, fixed_value=12.5).sample(rng, 5)) == {12.5}
    u = AmountDistribution(kind=AmountKind.UNIFORM, low=10, high=20).sample(rng, 1000)
    assert u.min() >= 10 and u.max() <= 20
    assert AmountDistribution(median=-1).validate()


@pytest.mark.parametrize("kw", [dict(n_transactions=0), dict(n_transactions=10**7), dict(horizon_minutes=0),
                                dict(currency="DOLLARS"), dict(method_mix={PM.CARD: 0.5})])
def test_invalid_workload_rejected(kw):
    with pytest.raises(ValueError):
        generate_workload(WorkloadConfig(**kw), np.random.SeedSequence(0))


def test_instantiated_transactions_match_latent_world():
    sc = build_scenario("baseline", 3000)
    wl, lat = prepare_trial(sc.workload, sc.failure_model, np.random.SeedSequence(5))
    txns = instantiate_transactions(wl, lat)
    failed = [t for t in txns if t.initial_status == PaymentStatus.FAILED]
    assert len(failed) == int(lat.initial_failed.sum())
    assert all(t.failure_category is not None for t in failed)
    assert all(t.failure_category is None for t in txns if t.initial_status == PaymentStatus.APPROVED)
    assert 0.05 < len(failed) / 3000 < 0.09  # assumed mix: ~7.5% initial failure
    assert txns[0].timestamp.tzinfo is not None and txns[0].transaction_id == "txn-000000"


def test_transaction_validation():
    kw = dict(transaction_id="x", merchant_id="m", currency="USD", timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
              payment_method_category=PM.CARD, offset_minutes=0.0)
    with pytest.raises(ValueError):
        Transaction(amount=0, **kw)
    with pytest.raises(ValueError):
        Transaction(amount=5, **{**kw, "currency": "US"})


# ---------------------------------------------------------------- failure classification
def test_default_classification_and_policy_visible_rules():
    assert RULES.class_of(FC.TRANSIENT_NETWORK_ERROR) == RecoveryClass.AUTO_RETRY
    assert RULES.class_of(FC.TEMPORARY_PROCESSING_ERROR) == RecoveryClass.AUTO_RETRY
    for c in (FC.INSTRUMENT_DECLINED, FC.EXPIRED_PAYMENT_METHOD, FC.PAYER_ACTION_REQUIRED, FC.ORDER_NOT_APPROVED):
        assert RULES.class_of(c) == RecoveryClass.BUYER_ACTION
    assert RULES.class_of(FC.VALIDATION_ERROR) == RecoveryClass.TERMINAL
    assert RULES.class_of(FC.TERMINAL_PAYMENT_FAILURE) == RecoveryClass.TERMINAL
    assert set(DEFAULT_CLASSIFICATION) == set(FC)  # every category classified; not all recoverable
    assert FailureRules(classification={FC.VALIDATION_ERROR: RecoveryClass.TERMINAL}).validate()


def test_failure_model_defaults_valid_and_scaling():
    m = FailureModel()
    assert m.validate() == []
    s = m.scaled(initial_failure_multiplier=2.0, category_weight_multipliers={FC.TRANSIENT_NETWORK_ERROR: 3.0})
    assert s.validate() == []
    assert s.initial_failure_prob[PM.CARD] == pytest.approx(0.16)
    row, base = s.category_mix[PM.CARD], m.category_mix[PM.CARD]
    assert sum(row.values()) == pytest.approx(1.0)
    assert row[FC.TRANSIENT_NETWORK_ERROR] / base[FC.TRANSIENT_NETWORK_ERROR] > 1.0
    bad = replace(m, category_mix={**m.category_mix, PM.CARD: {FC.INSTRUMENT_DECLINED: 0.5}})
    assert bad.validate()


# ---------------------------------------------------------------- policy validation / backoff / eligibility
def test_policy_validation():
    assert immediate_retry().validate() == intelligent_recovery().validate() == exponential_backoff().validate() == []
    assert RecoveryPolicy(retry_limit=21).validate()
    assert RecoveryPolicy(retry_limit=2, retry_intervals=()).validate()
    assert RecoveryPolicy(exponential_backoff=True, backoff_multiplier=0.5).validate()
    assert RecoveryPolicy(maximum_recovery_window=0).validate()
    assert RecoveryPolicy(alternate_funding_action=AlternateFundingAction.REQUEST_ALTERNATE).validate()
    assert RecoveryPolicy(classification_aware=False, request_buyer_action=True).validate()
    with pytest.raises(ValueError):
        resolve_policy("exponential_backoff", retry_limit=99)
    with pytest.raises(ValueError):
        resolve_policy("nope")


def test_exponential_backoff_schedule_and_cap():
    p = exponential_backoff()
    assert [p.delay_for_retry(k) for k in range(5)] == [5, 10, 20, 40, 80]
    capped = replace(p, backoff_max_interval_minutes=30.0)
    assert [capped.delay_for_retry(k) for k in range(5)] == [5, 10, 20, 30, 30]
    fixed = RecoveryPolicy(retry_intervals=(1.0, 4.0))
    assert [fixed.delay_for_retry(k) for k in range(4)] == [1, 4, 4, 4]


def test_retry_eligibility_blind_vs_aware():
    for cat in FC:
        assert is_retry_eligible(immediate_retry(), RULES, cat)  # blind retries everything
        assert is_retry_eligible(intelligent_recovery(), RULES, cat) == (cat in (FC.TRANSIENT_NETWORK_ERROR,
                                                                                 FC.TEMPORARY_PROCESSING_ERROR))
        assert not is_retry_eligible(no_recovery(), RULES, cat)


def test_buyer_action_eligibility():
    p = intelligent_recovery()
    assert buyer_action_eligible(p, RULES, FC.INSTRUMENT_DECLINED)
    assert not buyer_action_eligible(p, RULES, FC.TERMINAL_PAYMENT_FAILURE)
    assert not buyer_action_eligible(p, RULES, FC.TRANSIENT_NETWORK_ERROR)
    no_alt = replace(p, alternate_funding_action=AlternateFundingAction.NONE)
    assert not buyer_action_eligible(no_alt, RULES, FC.INSTRUMENT_DECLINED)
    assert not buyer_action_eligible(no_alt, RULES, FC.EXPIRED_PAYMENT_METHOD)
    assert buyer_action_eligible(no_alt, RULES, FC.ORDER_NOT_APPROVED)  # approval doesn't need a new funding source
    assert not buyer_action_eligible(exponential_backoff(), RULES, FC.ORDER_NOT_APPROVED)


def test_next_action_decisions():
    p = intelligent_recovery()
    a = next_action(p, RULES, FC.TRANSIENT_NETWORK_ERROR, 0, 0.0, False)
    assert (a.kind, a.delay_minutes) == (ActionKind.RETRY, 5.0)
    assert next_action(p, RULES, FC.TRANSIENT_NETWORK_ERROR, 2, 35.0, False).delay_minutes == 20.0
    assert next_action(p, RULES, FC.TRANSIENT_NETWORK_ERROR, 5, 155.0, False).reason == "retry_limit_reached"
    assert next_action(p, RULES, FC.VALIDATION_ERROR, 0, 0.0, False).reason == "terminal_failure"
    assert next_action(p, RULES, FC.INSTRUMENT_DECLINED, 0, 0.0, False).kind == ActionKind.REQUEST_BUYER_ACTION
    assert next_action(p, RULES, FC.INSTRUMENT_DECLINED, 0, 0.0, True).reason == "buyer_action_already_used"
    short = replace(p, maximum_recovery_window=12.0)
    assert next_action(short, RULES, FC.TRANSIENT_NETWORK_ERROR, 1, 5.0, False).reason == "recovery_window_exceeded"
    blind = immediate_retry()
    assert next_action(blind, RULES, FC.TERMINAL_PAYMENT_FAILURE, 0, 0.0, False).kind == ActionKind.RETRY
    assert next_action(blind, RULES, FC.TERMINAL_PAYMENT_FAILURE, 3, 3.0, False).reason == "retry_limit_reached"


# ---------------------------------------------------------------- lifecycle: transient, terminal, buyer
def test_transient_recovers_with_backoff_but_not_immediate_retry():
    # Condition clears 7 minutes after first failure. Backoff retries at +5 (too early) and +15 (works).
    t, seq, _ = run(FC.TRANSIENT_NETWORK_ERROR, exponential_backoff(), clear=7.0)
    assert (t.recovery_status, t.final_status, t.retry_count) == (RecoveryStatus.RECOVERED, FinalStatus.COMPLETED, 2)
    assert t.total_recovery_time == pytest.approx(15.0) and t.attempt_count == 3
    assert t.processing_cost == pytest.approx(3 * 0.03) and t.recovery_cost == pytest.approx(2 * 0.03)
    types = [e for _, e in seq]
    assert types[:5] == [E.TRANSACTION_CREATED, E.PAYMENT_ATTEMPTED, E.PAYMENT_FAILED, E.FAILURE_CLASSIFIED,
                         E.RECOVERY_SCHEDULED]
    assert types.count(E.RETRY_ATTEMPTED) == 2 and types[-2:] == [E.PAYMENT_APPROVED, E.PAYMENT_RECOVERED]
    # Immediate retry (1,1,1 minute) is exhausted at +3 min, before the condition clears.
    t2, seq2, ev2 = run(FC.TRANSIENT_NETWORK_ERROR, immediate_retry(), clear=7.0)
    assert t2.recovery_status == RecoveryStatus.UNRECOVERED and t2.retry_count == 3
    assert t2.total_recovery_time == pytest.approx(3.0)
    assert ev2[-1].type == E.PAYMENT_TERMINATED and ev2[-1].detail["reason"] == "retry_limit_reached"
    assert t2.unnecessary_retries == 0  # retrying a transient failure is never "unnecessary"


def test_unlucky_retries_exhaust_limit_even_after_clear():
    t, _, ev = run(FC.TEMPORARY_PROCESSING_ERROR, exponential_backoff(), clear=0.0, retry_u=0.999)
    assert t.retry_count == 5 and t.recovery_status == RecoveryStatus.UNRECOVERED
    assert t.total_recovery_time == pytest.approx(5 + 10 + 20 + 40 + 80)


def test_window_cuts_retries_short():
    pol = replace(exponential_backoff(), maximum_recovery_window=30.0)
    t, _, ev = run(FC.TEMPORARY_PROCESSING_ERROR, pol, clear=1000.0)
    assert t.retry_count == 2  # retries at +5, +15; next would be +35 > 30
    assert ev[-1].detail["reason"] == "recovery_window_exceeded"


def test_terminal_failure_blind_policy_wastes_retries():
    t, _, _ = run(FC.TERMINAL_PAYMENT_FAILURE, exponential_backoff(), retry_u=0.0)
    assert t.retry_count == 5 and t.terminal_retries == 5 and t.unnecessary_retries == 5
    assert t.recovery_status == RecoveryStatus.UNRECOVERED  # success prob is 0 for terminal failures even if u==0


def test_terminal_failure_intelligent_policy_makes_no_retries_and_escalates():
    t, seq, ev = run(FC.VALIDATION_ERROR, intelligent_recovery())
    assert t.retry_count == 0 and t.terminal_retries == 0 and t.attempt_count == 1
    assert t.recovery_status == RecoveryStatus.ESCALATED_FOR_REVIEW and t.final_status == FinalStatus.FAILED
    assert t.processing_cost == pytest.approx(0.03 + 0.25)
    assert ev[-1].type == E.PAYMENT_TERMINATED and ev[-1].detail == {"reason": "terminal_failure", "escalated": True}
    # abandon-handling variant: no review cost
    t2, _, _ = run(FC.VALIDATION_ERROR, replace(intelligent_recovery(), terminal_failure_handling=TerminalFailureHandling.ABANDON))
    assert t2.recovery_status == RecoveryStatus.UNRECOVERED and t2.processing_cost == pytest.approx(0.03)


def test_buyer_action_success_path_timing_and_cost():
    # response time = -120 * ln(1 - 0.5) minutes for INSTRUMENT_DECLINED
    expected = -120.0 * math.log(0.5)
    t, _, ev = run(FC.INSTRUMENT_DECLINED, intelligent_recovery(), buyer_resp=0.0, buyer_time=0.5, post=0.0)
    assert t.recovery_status == RecoveryStatus.RECOVERED and t.retry_count == 0
    assert (t.buyer_actions_requested, t.buyer_actions_completed, t.attempt_count) == (1, 1, 2)
    assert t.total_recovery_time == pytest.approx(expected)
    assert t.recovery_cost == pytest.approx(0.02 + 0.03)
    types = [e.type for e in ev]
    assert E.BUYER_ACTION_REQUESTED in types and E.BUYER_ACTION_COMPLETED in types and E.RETRY_ATTEMPTED not in types


def test_buyer_never_responds_or_too_slow_times_out():
    t, _, ev = run(FC.INSTRUMENT_DECLINED, intelligent_recovery(), buyer_resp=0.99)  # 0.99 >= 0.55 -> no response
    assert t.recovery_status == RecoveryStatus.UNRECOVERED and t.buyer_actions_completed == 0
    assert t.total_recovery_time == pytest.approx(1440.0)
    assert ev[-1].detail["reason"] == "buyer_action_timeout"
    t2, _, ev2 = run(FC.INSTRUMENT_DECLINED, intelligent_recovery(), buyer_resp=0.0, buyer_time=0.9999999)
    assert ev2[-1].detail["reason"] == "buyer_action_timeout" and t2.buyer_actions_completed == 0
    t3, _, ev3 = run(FC.INSTRUMENT_DECLINED, replace(intelligent_recovery(), buyer_action_timeout=10.0), buyer_resp=0.0, buyer_time=0.5)
    assert ev3[-1].detail["reason"] == "buyer_action_timeout" and t3.total_recovery_time == pytest.approx(10.0)


def test_post_buyer_attempt_can_fail():
    t, _, ev = run(FC.PAYER_ACTION_REQUIRED, intelligent_recovery(), buyer_resp=0.0, post=0.999)
    assert t.buyer_actions_completed == 1 and t.recovery_status == RecoveryStatus.UNRECOVERED
    assert ev[-1].detail["reason"] == "post_buyer_action_attempt_failed"


def test_no_alternate_funding_means_declined_is_abandoned_without_contact():
    pol = replace(intelligent_recovery(), alternate_funding_action=AlternateFundingAction.NONE)
    t, _, ev = run(FC.INSTRUMENT_DECLINED, pol, buyer_resp=0.0)
    assert t.buyer_actions_requested == 0 and t.retry_count == 0
    assert ev[-1].detail["reason"] == "not_recoverable_by_policy"


def test_blind_retry_of_buyer_action_failures_is_unnecessary_and_fails():
    t, _, _ = run(FC.ORDER_NOT_APPROVED, exponential_backoff(), retry_u=0.0)
    assert t.retry_count == 5 and t.buyer_class_retries == 5 and t.recovery_status == RecoveryStatus.UNRECOVERED


def test_approved_first_time_needs_no_recovery():
    t, _, ev = run(FC.INSTRUMENT_DECLINED, intelligent_recovery(), failed=False)
    assert t.initial_status == PaymentStatus.APPROVED and t.recovery_status == RecoveryStatus.NOT_NEEDED
    assert t.total_recovery_time is None and t.processing_cost == pytest.approx(0.03)
    assert [e.type for e in ev] == [E.TRANSACTION_CREATED, E.PAYMENT_ATTEMPTED, E.PAYMENT_APPROVED]


def test_idempotency_duplicate_capture_risk():
    pol = replace(exponential_backoff(), idempotent_retries=False)
    t, _, _ = run(FC.TRANSIENT_NETWORK_ERROR, pol, clear=0.0, dup=0.0)
    assert t.duplicate_capture_risk is True
    t2, _, _ = run(FC.TRANSIENT_NETWORK_ERROR, exponential_backoff(), clear=0.0, dup=0.0)
    assert t2.duplicate_capture_risk is False


# ---------------------------------------------------------------- reproducibility / tracing
def _key(res):
    return [(m.recovered, m.recovered_revenue, m.total_retries, round(m.total_processing_cost, 9)) for m in res.trial_metrics]


def test_reproducible_with_fixed_seed_and_sensitive_to_seed():
    sc = build_scenario("baseline", 3000)
    a = run_payment_monte_carlo(sc, exponential_backoff(), 4, 7)
    b = run_payment_monte_carlo(sc, exponential_backoff(), 4, 7)
    c = run_payment_monte_carlo(sc, exponential_backoff(), 4, 8)
    assert _key(a) == _key(b) and a.aggregate.to_dict() == b.aggregate.to_dict()
    assert _key(a) != _key(c)


def test_policies_share_identical_workload_per_trial():
    sc = build_scenario("baseline", 3000)
    res = compare_policies_monte_carlo(sc, [immediate_retry(), exponential_backoff(), intelligent_recovery()], 3, 1)
    for t in range(3):
        ms = [r.trial_metrics[t] for r in res.values()]
        assert len({(m.total_transactions, m.initially_failed, m.total_revenue, m.initially_failed_revenue) for m in ms}) == 1
    with pytest.raises(ValueError):
        compare_policies_monte_carlo(sc, [immediate_retry(), immediate_retry()], 2, 1)


def test_single_transaction_trace_matches_full_run():
    sc = build_scenario("baseline", 2000)
    pol = intelligent_recovery()
    seed = trial_seeds(11, 2)[1]
    wl, lat = prepare_trial(sc.workload, sc.failure_model, seed)
    full = simulate_trial(sc, pol, wl, lat).transactions
    checked = 0
    for i in np.flatnonzero(lat.initial_failed)[:15]:
        txn, events = trace_transaction(sc, pol, 11, 1, int(i))
        assert txn.to_dict() == full[i].to_dict()
        assert events[0].type == E.TRANSACTION_CREATED and events[-1].type in (E.PAYMENT_RECOVERED, E.PAYMENT_TERMINATED)
        checked += 1
    assert checked == 15
    with pytest.raises(ValueError):
        trace_transaction(sc, pol, 11, 1, 99_999)


def test_invalid_policy_rejected_by_simulator():
    sc, wl, lat = one_txn(FC.TRANSIENT_NETWORK_ERROR)
    with pytest.raises(ValueError):
        simulate_trial(sc, RecoveryPolicy(retry_limit=50), wl, lat)


# ---------------------------------------------------------------- metrics, Monte Carlo aggregation, CIs
def _txn(i, amount, status, rec, retries=0, rt=None, cost=0.03, unnec=0, term=0, buyer_cls=0):
    t = Transaction(f"t{i}", "m", amount, "USD", datetime(2026, 1, 1, tzinfo=timezone.utc), PM.CARD, 0.0,
                    initial_status=status, recovery_status=rec, retry_count=retries, total_recovery_time=rt,
                    processing_cost=cost, recovery_cost=cost - 0.03, attempt_count=1 + retries,
                    unnecessary_retries=unnec, terminal_retries=term, buyer_class_retries=buyer_cls)
    t.final_status = FinalStatus.COMPLETED if rec in (RecoveryStatus.NOT_NEEDED, RecoveryStatus.RECOVERED) else FinalStatus.FAILED
    return t


def test_trial_metrics_exact_values_and_denominators():
    A, F = PaymentStatus.APPROVED, PaymentStatus.FAILED
    N, R, U = RecoveryStatus.NOT_NEEDED, RecoveryStatus.RECOVERED, RecoveryStatus.UNRECOVERED
    txns = [_txn(0, 10, A, N), _txn(1, 20, A, N), _txn(2, 30, A, N), _txn(3, 40, A, N), _txn(4, 50, A, N), _txn(5, 60, A, N),
            _txn(6, 100, F, R, retries=2, rt=30.0, cost=0.09),
            _txn(7, 200, F, R, retries=1, rt=10.0, cost=0.06),
            _txn(8, 300, F, U, retries=3, rt=3.0, cost=0.12, unnec=3, term=3),
            _txn(9, 400, F, U, retries=0, rt=0.0, cost=0.03)]
    m = compute_trial_metrics(txns)
    assert (m.total_transactions, m.successful_initial, m.initially_failed, m.recovered, m.final_failed) == (10, 6, 4, 2, 2)
    assert m.recovery_rate == 0.5 and m.completion_rate == 0.8  # 2/4 and (6+2)/10
    assert m.recovered_revenue == 300 and m.initially_failed_revenue == 1000 and m.total_revenue == 1210
    assert m.avg_recovery_time == 20.0  # (30+10)/2, recovered only
    assert m.total_retries == 6 and m.avg_retries_per_transaction == 0.6 and m.avg_retries_per_failed == 1.5
    assert (m.unnecessary_retries, m.terminal_retries) == (3, 3) and m.unnecessary_retry_fraction == 0.5
    assert m.total_processing_cost == pytest.approx(6 * 0.03 + 0.09 + 0.06 + 0.12 + 0.03)
    assert m.cost_per_successful_transaction == pytest.approx(m.total_processing_cost / 8)
    assert m.max_retry_count == 3 and m.total_attempts == 10 + 6


def test_metrics_zero_denominators_are_safe():
    m = compute_trial_metrics([_txn(0, 10, PaymentStatus.APPROVED, RecoveryStatus.NOT_NEEDED)])
    assert m.recovery_rate == 0.0 and m.avg_recovery_time == 0.0 and m.unnecessary_retry_fraction == 0.0


def test_mean_ci_known_values():
    r = mean_ci([1.0, 2.0, 3.0])
    half = 1.959963984540054 * 1.0 / math.sqrt(3)
    assert r["mean"] == 2.0 and r["std"] == pytest.approx(1.0)
    assert r["ci95"] == pytest.approx([2.0 - half, 2.0 + half])
    assert mean_ci([5.0])["ci95"] is None


def test_aggregation_pools_counts_and_uses_wilson():
    sc = build_scenario("baseline", 2000)
    res = run_payment_monte_carlo(sc, exponential_backoff(), 6, 3)
    tm = res.trial_metrics
    pooled = res.aggregate.pooled["recovery_rate"]
    assert pooled["numerator"] == sum(t.recovered for t in tm) and pooled["denominator"] == sum(t.initially_failed for t in tm)
    assert pooled["ci95_wilson"] == pytest.approx(list(wilson_interval(pooled["numerator"], pooled["denominator"])))
    lo, hi = pooled["ci95_wilson"]
    assert lo < pooled["value"] < hi
    assert res.aggregate.means["recovered_revenue"]["mean"] == pytest.approx(np.mean([t.recovered_revenue for t in tm]))
    assert res.aggregate.pooled["avg_recovery_time_minutes"]["value"] == pytest.approx(
        sum(t.recovery_time_sum for t in tm) / sum(t.recovered for t in tm))
    assert res.simulated and "NOT_PAYPAL" in res.label


def test_ci_narrows_with_more_transactions():
    w = lambda n: run_payment_monte_carlo(build_scenario("baseline", n), exponential_backoff(), 4, 2).aggregate.pooled["recovery_rate"]["ci95_wilson"]
    small, big = w(1000), w(10000)
    assert (big[1] - big[0]) < (small[1] - small[0])


# ---------------------------------------------------------------- objectives
def test_score_trial_formula():
    m = compute_trial_metrics([
        _txn(0, 100, PaymentStatus.FAILED, RecoveryStatus.RECOVERED, retries=1, rt=720.0, cost=0.33, unnec=0),
        _txn(1, 100, PaymentStatus.FAILED, RecoveryStatus.UNRECOVERED, retries=1, rt=5.0, cost=0.33, unnec=1, term=1)])
    w = ObjectiveWeights(revenue=1.0, cost=0.2, latency=0.1, unnecessary_attempts=0.2, cost_reference=1.0, latency_reference_minutes=1440.0)
    s = score_trial(m, w)
    assert s.revenue_term == 0.5                       # 100 / 200
    assert s.cost_term == pytest.approx(0.30)          # (0.60 recovery cost / 2 failed) / 1.0
    assert s.latency_term == 0.5                       # 720 / 1440
    assert s.unnecessary_term == 0.5                   # 1 / 2 retries
    assert s.score == pytest.approx(0.5 - 0.2 * 0.30 - 0.1 * 0.5 - 0.2 * 0.5)


def test_weights_validation():
    assert ObjectiveWeights(revenue=0).validate()
    assert ObjectiveWeights(cost=-1).validate()
    assert ObjectiveWeights(latency_reference_minutes=0).validate()


def test_static_constraints_per_baseline():
    hc = HardConstraints()
    assert static_constraint_violations(intelligent_recovery(), RULES, hc) == []
    blind = static_constraint_violations(exponential_backoff(), RULES, hc)
    assert any("terminal failure" in v for v in blind) and any("buyer action" in v for v in blind)
    assert static_constraint_violations(no_recovery(), RULES, hc) == []
    assert static_constraint_violations(exponential_backoff(), RULES, HardConstraints(None, None, False, False, False)) == []
    assert static_constraint_violations(replace(intelligent_recovery(), retry_limit=8), RULES, hc)  # > max_retries 5
    assert static_constraint_violations(replace(intelligent_recovery(), maximum_recovery_window=9999.0), RULES, hc)
    assert static_constraint_violations(replace(intelligent_recovery(), idempotent_retries=False), RULES, hc)


def test_hard_constraints_dominate_score_in_ranking():
    sc = build_scenario("baseline", 3000)
    res = compare_policies_monte_carlo(sc, [immediate_retry(), exponential_backoff(), intelligent_recovery()], 4, 9)
    w = ObjectiveWeights()
    ranked = rank_policies([evaluate_policy(r, sc.rules, w, HardConstraints()) for r in res.values()])
    assert ranked[0].policy_name == "intelligent_recovery" and ranked[0].feasible
    assert [e.feasible for e in ranked] == [True, False, False]
    # observed counts also flag blind retries of terminal failures
    assert any("observed" in v for v in ranked[1].violations)
    # a feasible policy outranks an infeasible one even with a higher score
    fake = [PolicyEvaluation("bad", False, ("x",), 0.9, None, {}), PolicyEvaluation("ok", True, (), 0.1, None, {})]
    assert [e.policy_name for e in rank_policies(fake)] == ["ok", "bad"]


def test_ranking_without_constraints_matches_score_order_and_is_stable():
    sc = build_scenario("baseline", 3000)
    res = compare_policies_monte_carlo(sc, [immediate_retry(), exponential_backoff(), intelligent_recovery()], 4, 9)
    free = HardConstraints(None, None, False, False, False)
    ev = [evaluate_policy(r, sc.rules, ObjectiveWeights(), free) for r in res.values()]
    assert all(e.feasible for e in ev)
    r1 = [e.policy_name for e in rank_policies(ev)]
    assert r1 == [e.policy_name for e in rank_policies(list(reversed(ev)))]
    assert r1 == ["intelligent_recovery", "exponential_backoff", "immediate_retry"]


def test_weights_change_the_winner():
    # Make latency extremely costly: the fast-but-weak immediate retry should beat slow intelligent recovery.
    sc = build_scenario("baseline", 3000)
    res = compare_policies_monte_carlo(sc, [immediate_retry(), intelligent_recovery()], 4, 9)
    free = HardConstraints(None, None, False, False, False)
    w = ObjectiveWeights(revenue=0.05, cost=0.0, latency=5.0, unnecessary_attempts=0.0, latency_reference_minutes=60.0)
    ranked = rank_policies([evaluate_policy(r, sc.rules, w, free) for r in res.values()])
    assert ranked[0].policy_name == "immediate_retry"


def test_paired_difference():
    d = paired_difference([3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0])
    assert d["mean_difference"] == 2.0 and d["ci_excludes_zero"] is True
    assert paired_difference([1.0, -1.0, 1.0, -1.0], [0.0] * 4)["ci_excludes_zero"] is False
    with pytest.raises(ValueError):
        paired_difference([1.0], [1.0, 2.0])


# ---------------------------------------------------------------- acceptance: >=10k txns, 3 policies, meaningful differences
def test_acceptance_10k_transactions_three_policies_meaningful_differences():
    sc = build_scenario("baseline", 10_000)
    pols = [immediate_retry(), exponential_backoff(), intelligent_recovery()]
    res = compare_policies_monte_carlo(sc, pols, 5, 42)
    assert all(m.total_transactions == 10_000 for r in res.values() for m in r.trial_metrics)
    rr = {n: r.aggregate.pooled["recovery_rate"] for n, r in res.items()}
    assert rr["immediate_retry"]["value"] < rr["exponential_backoff"]["value"] < rr["intelligent_recovery"]["value"]
    # confidence intervals are disjoint: differences are not sampling noise
    assert rr["immediate_retry"]["ci95_wilson"][1] < rr["exponential_backoff"]["ci95_wilson"][0]
    assert rr["exponential_backoff"]["ci95_wilson"][1] < rr["intelligent_recovery"]["ci95_wilson"][0]
    unn = {n: r.aggregate.means["unnecessary_retries"]["mean"] for n, r in res.items()}
    assert unn["intelligent_recovery"] == 0 and unn["immediate_retry"] > 100 and unn["exponential_backoff"] > 100
    lat = {n: r.aggregate.pooled["avg_recovery_time_minutes"]["value"] for n, r in res.items()}
    assert lat["immediate_retry"] < lat["exponential_backoff"] < lat["intelligent_recovery"]  # real trade-off
