"""Discrete-event payment-lifecycle simulation and Monte Carlo driver.

Each transaction is a SimPy process on a shared minute clock. Transactions
are independent (no shared capacity is modelled), which is what allows a
single transaction to be re-simulated alone for tracing and get exactly the
same result as in the full run.

SIMULATED merchant-side recovery behaviour under ASSUMED distributions.
Not PayPal. See docs/PAYMENT_SIMULATION_MODEL.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from engine.payments.event_scheduler import EventLog, EventScheduler
from engine.payments.failure_model import CATEGORIES, FailureModel, FailureRules, LatentOutcomes, RecoveryClass
from engine.payments.metrics import AggregateMetrics, TrialMetrics, aggregate_trials, compute_trial_metrics
from engine.payments.models import (ASSUMPTION_LABEL, SIM_LABEL, CostModel, EventType as E, FailureCategory,
                                    FinalStatus, PaymentStatus, RecoveryStatus, Transaction)
from engine.payments.recovery_policy import ActionKind, RecoveryPolicy, TerminalFailureHandling, next_action
from engine.payments.scenarios import PaymentScenario
from engine.payments.transaction_generator import Workload, instantiate_transactions, prepare_trial


@dataclass
class _Ctx:
    policy: RecoveryPolicy
    rules: FailureRules
    model: FailureModel
    costs: CostModel
    latent: LatentOutcomes
    log: EventLog


def _lifecycle(env, idx: int, txn: Transaction, ctx: _Ctx):
    pol, lat, log = ctx.policy, ctx.latent, ctx.log
    yield env.timeout(txn.offset_minutes)
    log.record(idx, env.now, E.TRANSACTION_CREATED, amount=txn.amount,
               method=txn.payment_method_category.value)
    log.record(idx, env.now, E.PAYMENT_ATTEMPTED, attempt=1)
    txn.attempt_count = 1
    txn.processing_cost += ctx.costs.attempt_cost

    if not lat.initial_failed[idx]:
        log.record(idx, env.now, E.PAYMENT_APPROVED, attempt=1)
        txn.resolved_at_minutes = env.now
        return  # defaults already describe an approved-first-time transaction

    cat: FailureCategory = txn.failure_category
    cls = ctx.rules.class_of(cat)
    t_fail = env.now
    txn.final_status = FinalStatus.FAILED
    txn.recovery_status = RecoveryStatus.UNRECOVERED

    def fail_events():
        log.record(idx, env.now, E.PAYMENT_FAILED, category=cat.value)
        log.record(idx, env.now, E.FAILURE_CLASSIFIED, category=cat.value, recovery_class=cls.value)

    def finish(recovered: bool, reason: str = "", escalate: bool = False):
        txn.total_recovery_time = env.now - t_fail
        txn.resolved_at_minutes = env.now
        if recovered:
            txn.final_status, txn.recovery_status = FinalStatus.COMPLETED, RecoveryStatus.RECOVERED
            log.record(idx, env.now, E.PAYMENT_RECOVERED, retries=txn.retry_count,
                       total_recovery_minutes=round(txn.total_recovery_time, 4))
        else:
            if escalate:
                txn.recovery_status = RecoveryStatus.ESCALATED_FOR_REVIEW
                txn.processing_cost += ctx.costs.manual_review_cost
                txn.recovery_cost += ctx.costs.manual_review_cost
            log.record(idx, env.now, E.PAYMENT_TERMINATED, reason=reason, escalated=escalate)

    def charge(amount: float):
        txn.processing_cost += amount
        txn.recovery_cost += amount

    fail_events()
    buyer_used = False
    while True:
        elapsed = env.now - t_fail
        act = next_action(pol, ctx.rules, cat, txn.retry_count, elapsed, buyer_used)

        if act.kind == ActionKind.TERMINATE:
            escalate = (act.reason == "terminal_failure"
                        and pol.terminal_failure_handling == TerminalFailureHandling.FLAG_FOR_REVIEW)
            finish(False, act.reason, escalate)
            return

        if act.kind == ActionKind.RETRY:
            log.record(idx, env.now, E.RECOVERY_SCHEDULED, action="RETRY", delay_minutes=act.delay_minutes,
                       retry_number=txn.retry_count + 1)
            yield env.timeout(act.delay_minutes)
            k = txn.retry_count
            txn.retry_count += 1
            txn.attempt_count += 1
            charge(ctx.costs.attempt_cost)
            if cls != RecoveryClass.AUTO_RETRY:
                txn.unnecessary_retries += 1
                if cls == RecoveryClass.TERMINAL:
                    txn.terminal_retries += 1
                else:
                    txn.buyer_class_retries += 1
            if (not pol.idempotent_retries and cat == FailureCategory.TRANSIENT_NETWORK_ERROR
                    and lat.duplicate_u[idx] < ctx.model.duplicate_capture_prob_without_idempotency):
                txn.duplicate_capture_risk = True
            log.record(idx, env.now, E.RETRY_ATTEMPTED, retry_number=txn.retry_count,
                       idempotent=pol.idempotent_retries)
            since_failure = env.now - t_fail
            ok = (since_failure >= lat.clear_minutes[idx]
                  and lat.retry_u[idx, k] < ctx.model.retry_success_after_clear[cat])
            if ok:
                log.record(idx, env.now, E.PAYMENT_APPROVED, attempt=txn.attempt_count)
                finish(True)
                return
            fail_events()
            continue

        # REQUEST_BUYER_ACTION
        buyer_used = True
        txn.buyer_actions_requested += 1
        charge(ctx.costs.buyer_notification_cost)
        log.record(idx, env.now, E.BUYER_ACTION_REQUESTED, category=cat.value,
                   timeout_minutes=pol.buyer_action_timeout)
        remaining = min(pol.buyer_action_timeout, pol.maximum_recovery_window - elapsed)
        responds = lat.buyer_responds_u[idx] < ctx.model.buyer_response_prob[cat]
        resp_time = float(-ctx.model.buyer_response_mean_minutes[cat] * np.log1p(-lat.buyer_time_u[idx]))
        if responds and resp_time <= remaining:
            yield env.timeout(resp_time)
            txn.buyer_actions_completed += 1
            log.record(idx, env.now, E.BUYER_ACTION_COMPLETED, category=cat.value)
            log.record(idx, env.now, E.PAYMENT_ATTEMPTED, attempt=txn.attempt_count + 1, after_buyer_action=True)
            txn.attempt_count += 1
            charge(ctx.costs.attempt_cost)
            if lat.post_buyer_u[idx] < ctx.model.post_buyer_success_prob[cat]:
                log.record(idx, env.now, E.PAYMENT_APPROVED, attempt=txn.attempt_count)
                finish(True)
            else:
                fail_events()
                finish(False, "post_buyer_action_attempt_failed")
            return
        yield env.timeout(remaining)
        finish(False, "buyer_action_timeout" if remaining == pol.buyer_action_timeout else "recovery_window_exceeded")
        return


@dataclass
class TrialRun:
    transactions: list[Transaction]
    log: EventLog


def simulate_trial(scenario: PaymentScenario, policy: RecoveryPolicy, workload: Workload,
                   latent: LatentOutcomes, record_events: bool = False,
                   only_index: Optional[int] = None) -> TrialRun:
    problems = policy.validate()
    if problems:
        raise ValueError("Invalid RecoveryPolicy: " + "; ".join(problems))
    txns = instantiate_transactions(workload, latent)
    ctx = _Ctx(policy, scenario.rules, scenario.failure_model, scenario.costs, latent, EventLog(record_events))
    sched = EventScheduler()
    indices = range(len(txns)) if only_index is None else [only_index]
    for i in indices:
        sched.spawn(_lifecycle, i, txns[i], ctx)
    sched.run()
    if only_index is not None:
        txns = [txns[only_index]]
    return TrialRun(txns, ctx.log)


@dataclass(frozen=True)
class PaymentMCResult:
    policy: RecoveryPolicy
    scenario_name: str
    n_trials: int
    master_seed: int
    trial_metrics: tuple[TrialMetrics, ...]
    aggregate: AggregateMetrics
    simulated: bool = True
    label: str = SIM_LABEL
    assumptions_label: str = ASSUMPTION_LABEL


def trial_seeds(master_seed: int, n_trials: int) -> list[np.random.SeedSequence]:
    """Same spawn pattern as engine.montecarlo.run_monte_carlo."""
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")
    return np.random.SeedSequence(master_seed).spawn(n_trials)


def compare_policies_monte_carlo(scenario: PaymentScenario, policies: list[RecoveryPolicy],
                                 n_trials: int, master_seed: int) -> dict[str, PaymentMCResult]:
    """Evaluate every policy on IDENTICAL workloads and latent draws per trial (common random numbers)."""
    problems = scenario.validate()
    if problems:
        raise ValueError("Invalid scenario: " + "; ".join(problems))
    names = [p.name for p in policies]
    if len(set(names)) != len(names):
        raise ValueError("policy names must be unique")
    per_policy: dict[str, list[TrialMetrics]] = {n: [] for n in names}
    for seed in trial_seeds(master_seed, n_trials):
        workload, latent = prepare_trial(scenario.workload, scenario.failure_model, seed)
        for pol in policies:
            run = simulate_trial(scenario, pol, workload, latent)
            per_policy[pol.name].append(compute_trial_metrics(run.transactions))
    return {p.name: PaymentMCResult(p, scenario.name, n_trials, master_seed, tuple(per_policy[p.name]),
                                    aggregate_trials(per_policy[p.name])) for p in policies}


def run_payment_monte_carlo(scenario: PaymentScenario, policy: RecoveryPolicy, n_trials: int,
                            master_seed: int) -> PaymentMCResult:
    return compare_policies_monte_carlo(scenario, [policy], n_trials, master_seed)[policy.name]


def trace_transaction(scenario: PaymentScenario, policy: RecoveryPolicy, master_seed: int, trial_index: int,
                      transaction_index: int) -> tuple[Transaction, list]:
    """Re-simulate ONE transaction with full event logging; identical to its outcome in the full run."""
    seeds = trial_seeds(master_seed, trial_index + 1)
    workload, latent = prepare_trial(scenario.workload, scenario.failure_model, seeds[trial_index])
    if not 0 <= transaction_index < scenario.workload.n_transactions:
        raise ValueError("transaction_index out of range")
    run = simulate_trial(scenario, policy, workload, latent, record_events=True, only_index=transaction_index)
    return run.transactions[0], run.log.events.get(transaction_index, [])
