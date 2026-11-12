"""Deterministic policy executor: (payment state, classified failure event) -> RETRY | WAIT | REQUIRE_BUYER_ACTION | STOP.

Pure function of its inputs: no clock, no randomness, no I/O. Every decision carries the rule that fired, a rationale, and the
provenance (source optimization run, policy, policy hash, plan id/version) so a trace is auditable end to end.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from engine.payments.failure_model import FailureRules, RecoveryClass
from engine.payments.models import FailureCategory
from engine.payments.recovery_policy import ActionKind, TerminalFailureHandling, next_action

from execution_bridge.plan import ExecutionPlan

SIMULATED_STATE = "SIMULATED_STATE"
REAL_SANDBOX_READONLY = "REAL_SANDBOX_READONLY"


class Action(str, Enum):
    RETRY = "RETRY"
    WAIT = "WAIT"
    REQUIRE_BUYER_ACTION = "REQUIRE_BUYER_ACTION"
    STOP = "STOP"


@dataclass(frozen=True)
class PaymentState:
    payment_id: str
    order_status: str = "CREATED"  # CREATED | PAYER_ACTION_REQUIRED | APPROVED | COMPLETED | VOIDED
    retries_made: int = 0
    elapsed_minutes: float = 0.0
    buyer_action_used: bool = False
    awaiting_buyer_since: Optional[float] = None  # elapsed-minutes mark when the buyer was asked to act
    idempotency_key: Optional[str] = None
    capture_id: Optional[str] = None
    source: str = SIMULATED_STATE


@dataclass(frozen=True)
class FailureEvent:
    category: FailureCategory
    note: str = ""


@dataclass(frozen=True)
class Decision:
    action: Action
    rule: str
    rationale: str
    delay_minutes: float = 0.0
    flag_for_review: bool = False


@dataclass(frozen=True)
class TraceStep:
    index: int
    payment_id: str
    state_source: str
    order_status: str
    retries_made: int
    elapsed_minutes: float
    event: Optional[str]
    action: str
    delay_minutes: float
    rule: str
    rationale: str
    flag_for_review: bool
    plan_id: str
    plan_version: str
    source_run_id: str
    policy_name: str
    policy_hash: str


class PolicyExecutor:
    def __init__(self, plan: ExecutionPlan, rules: Optional[FailureRules] = None):
        self.plan, self.rules, self.policy = plan, rules or FailureRules(), plan.engine_policy()
        self.trace: list[TraceStep] = []

    def decide(self, state: PaymentState, event: Optional[FailureEvent] = None) -> Decision:
        """Pure decision; use `step` to also record the trace."""
        p, plan = self.policy, self.plan
        # 1. Safety first: a finished payment is never retried or re-captured, whatever the event says.
        if state.order_status == "COMPLETED" or state.capture_id:
            return Decision(Action.STOP, "NEVER_RECAPTURE", "payment already completed/captured; no retry, no second capture")
        if state.order_status == "VOIDED":
            return Decision(Action.STOP, "ORDER_VOIDED", "order is voided; nothing recoverable")
        if event is None:
            if state.order_status in ("CREATED", "PAYER_ACTION_REQUIRED"):
                return Decision(Action.WAIT, "AWAITING_BUYER_APPROVAL", "order is not APPROVED; capture is not possible until the buyer approves")
            return Decision(Action.STOP, "NO_FAILURE_TO_RECOVER", "no failure event: nothing for the recovery policy to do (capture is a separate explicit step)")
        cat = event.category
        cls = self.rules.class_of(cat)
        flag = p.terminal_failure_handling == TerminalFailureHandling.FLAG_FOR_REVIEW and cls == RecoveryClass.TERMINAL
        # 2. A buyer action is already pending: wait for it, or stop once its timeout passes.
        if state.awaiting_buyer_since is not None:
            waited = state.elapsed_minutes - state.awaiting_buyer_since
            if waited < plan.buyer_action_timeout_minutes:
                return Decision(Action.WAIT, "BUYER_ACTION_PENDING", f"waiting for the buyer ({waited:.0f} of {plan.buyer_action_timeout_minutes:.0f} min used)")
            return Decision(Action.STOP, "BUYER_ACTION_TIMEOUT", f"buyer did not act within {plan.buyer_action_timeout_minutes:.0f} min")
        # 3. Terminal failures are never retried when the objective forbids it (defence in depth: also checked at compile time).
        if cls == RecoveryClass.TERMINAL and plan.constraints.get("forbid_terminal_retries"):
            return Decision(Action.STOP, "TERMINAL_FAILURE_NO_RETRY", f"{cat.value} is terminal; retries are forbidden", flag_for_review=flag)
        a = next_action(p, self.rules, cat, state.retries_made, state.elapsed_minutes, state.buyer_action_used)
        if a.kind == ActionKind.RETRY:
            if state.retries_made >= min(plan.retry_limit, plan.constraints.get("max_retries", plan.retry_limit)):  # belt and braces
                return Decision(Action.STOP, "RETRY_BUDGET_EXHAUSTED", f"{state.retries_made} retries already made (limit {plan.retry_limit})")
            if plan.constraints.get("require_idempotency") and not state.idempotency_key:
                return Decision(Action.STOP, "RETRY_BLOCKED_NO_IDEMPOTENCY_KEY", "retry requires a stable idempotency key (PayPal-Request-Id); none present")
            return Decision(Action.RETRY, "SCHEDULED_RETRY", f"{cat.value} is retryable; retry {state.retries_made + 1} of {plan.retry_limit} after {a.delay_minutes:g} min, same idempotency key",
                            delay_minutes=a.delay_minutes)
        if a.kind == ActionKind.REQUEST_BUYER_ACTION:
            return Decision(Action.REQUIRE_BUYER_ACTION, "BUYER_ACTION_REQUIRED", f"{cat.value} needs the buyer (policy requests buyer action; timeout {plan.buyer_action_timeout_minutes:.0f} min)")
        reason = {"retry_limit_reached": "RETRY_BUDGET_EXHAUSTED", "terminal_failure": "TERMINAL_FAILURE_NO_RETRY", "recovery_window_exceeded": "RECOVERY_WINDOW_EXCEEDED",
                  "buyer_action_already_used": "BUYER_ACTION_ALREADY_USED", "not_recoverable_by_policy": "NOT_RECOVERABLE_BY_POLICY"}.get(a.reason, a.reason.upper())
        return Decision(Action.STOP, reason, f"{cat.value}: {a.reason.replace('_', ' ')}", flag_for_review=flag)

    def step(self, state: PaymentState, event: Optional[FailureEvent] = None) -> Decision:
        d = self.decide(state, event)
        pl = self.plan
        self.trace.append(TraceStep(len(self.trace) + 1, state.payment_id, state.source, state.order_status, state.retries_made, state.elapsed_minutes,
                                    event.category.value if event else None, d.action.value, d.delay_minutes, d.rule, d.rationale, d.flag_for_review,
                                    pl.plan_id, pl.plan_version, pl.source_run_id, pl.policy_name, pl.policy_hash))
        return d
