"""Recovery policies: deterministic, pure decision logic (no randomness here).

A policy answers one question after every failed attempt: given the failure
category, how many retries have been made and how long recovery has been
running, what do we do next -- retry (after what delay), ask the buyer to
act, or stop? `next_action` is a pure function so it can be tested exhaustively.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from engine.payments.failure_model import (FUNDING_SOURCE_CATEGORIES, MAX_RETRY_DRAWS, FailureRules,
                                           RecoveryClass)
from engine.payments.models import FailureCategory


class AlternateFundingAction(str, Enum):
    NONE = "NONE"  # never ask for a different/updated funding source
    REQUEST_ALTERNATE = "REQUEST_ALTERNATE"  # ask the buyer for another funding source / updated instrument


class TerminalFailureHandling(str, Enum):
    ABANDON = "ABANDON"  # stop silently
    FLAG_FOR_REVIEW = "FLAG_FOR_REVIEW"  # stop, and escalate terminal-class failures to a human (costs money)


class ActionKind(str, Enum):
    RETRY = "RETRY"
    REQUEST_BUYER_ACTION = "REQUEST_BUYER_ACTION"
    TERMINATE = "TERMINATE"


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    delay_minutes: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class RecoveryPolicy:
    name: str = "custom"
    retry_limit: int = 3  # max automatic retries per transaction
    retry_intervals: tuple[float, ...] = (5.0,)  # minutes; used when exponential_backoff is False
    exponential_backoff: bool = False
    backoff_base_minutes: float = 5.0  # delay before retry #1 when exponential
    backoff_multiplier: float = 2.0
    backoff_max_interval_minutes: float = 240.0
    maximum_recovery_window: float = 1440.0  # minutes from first failure
    buyer_action_timeout: float = 1440.0  # minutes the buyer has to act
    request_buyer_action: bool = False
    alternate_funding_action: AlternateFundingAction = AlternateFundingAction.NONE
    terminal_failure_handling: TerminalFailureHandling = TerminalFailureHandling.ABANDON
    classification_aware: bool = False  # False = blind: treats every failure as retryable
    idempotent_retries: bool = True  # retries reuse an idempotency key (PayPal-Request-Id)

    def validate(self) -> list[str]:
        p: list[str] = []
        if not 0 <= self.retry_limit <= MAX_RETRY_DRAWS:
            p.append(f"retry_limit must be in [0, {MAX_RETRY_DRAWS}]")
        if any(i < 0 for i in self.retry_intervals):
            p.append("retry_intervals must be >= 0")
        if not self.exponential_backoff and self.retry_limit > 0 and not self.retry_intervals:
            p.append("retry_intervals required when retry_limit > 0 and exponential_backoff is False")
        if self.exponential_backoff and (self.backoff_base_minutes <= 0 or self.backoff_multiplier < 1
                                         or self.backoff_max_interval_minutes < self.backoff_base_minutes):
            p.append("exponential backoff needs base > 0, multiplier >= 1, max_interval >= base")
        if self.maximum_recovery_window <= 0:
            p.append("maximum_recovery_window must be > 0")
        if self.buyer_action_timeout <= 0:
            p.append("buyer_action_timeout must be > 0")
        if self.alternate_funding_action != AlternateFundingAction.NONE and not self.request_buyer_action:
            p.append("alternate_funding_action requires request_buyer_action")
        if self.request_buyer_action and not self.classification_aware:
            p.append("request_buyer_action requires classification_aware (a blind policy cannot choose it)")
        return p

    def delay_for_retry(self, retry_index: int) -> float:
        """Delay (minutes) before automatic retry number retry_index+1 (0-based index)."""
        if self.exponential_backoff:
            return min(self.backoff_base_minutes * self.backoff_multiplier ** retry_index,
                       self.backoff_max_interval_minutes)
        return self.retry_intervals[min(retry_index, len(self.retry_intervals) - 1)]


def is_retry_eligible(policy: RecoveryPolicy, rules: FailureRules, category: FailureCategory) -> bool:
    """Would this policy ever automatically retry a failure of this category (ignoring limits/window)?"""
    if policy.retry_limit <= 0:
        return False
    if not policy.classification_aware:
        return True  # blind: retries everything
    return rules.class_of(category) == RecoveryClass.AUTO_RETRY


def buyer_action_eligible(policy: RecoveryPolicy, rules: FailureRules, category: FailureCategory) -> bool:
    if not (policy.classification_aware and policy.request_buyer_action):
        return False
    if rules.class_of(category) != RecoveryClass.BUYER_ACTION:
        return False
    if category in FUNDING_SOURCE_CATEGORIES:
        return policy.alternate_funding_action == AlternateFundingAction.REQUEST_ALTERNATE
    return True


def next_action(policy: RecoveryPolicy, rules: FailureRules, category: FailureCategory,
                retries_made: int, elapsed_minutes: float, buyer_action_used: bool) -> Action:
    """Decide what to do after a failed attempt. Pure and deterministic."""
    if buyer_action_eligible(policy, rules, category):
        if buyer_action_used:
            return Action(ActionKind.TERMINATE, reason="buyer_action_already_used")
        if elapsed_minutes >= policy.maximum_recovery_window:
            return Action(ActionKind.TERMINATE, reason="recovery_window_exceeded")
        return Action(ActionKind.REQUEST_BUYER_ACTION, 0.0, "buyer_action_required")

    if policy.classification_aware and rules.class_of(category) == RecoveryClass.TERMINAL:
        return Action(ActionKind.TERMINATE, reason="terminal_failure")

    if is_retry_eligible(policy, rules, category):
        if retries_made >= policy.retry_limit:
            return Action(ActionKind.TERMINATE, reason="retry_limit_reached")
        delay = policy.delay_for_retry(retries_made)
        if elapsed_minutes + delay > policy.maximum_recovery_window:
            return Action(ActionKind.TERMINATE, reason="recovery_window_exceeded")
        return Action(ActionKind.RETRY, delay, "scheduled_retry")

    return Action(ActionKind.TERMINATE, reason="not_recoverable_by_policy")


# --- baseline strategies ----------------------------------------------------

def immediate_retry() -> RecoveryPolicy:
    """A. Blind: retry every failure up to 3 times, 1 minute apart. No classification, no buyer contact."""
    return RecoveryPolicy(name="immediate_retry", retry_limit=3, retry_intervals=(1.0,),
                          maximum_recovery_window=60.0)


def exponential_backoff() -> RecoveryPolicy:
    """B. Blind: retry every failure up to 5 times with 5,10,20,40,80 minute gaps. No classification."""
    return RecoveryPolicy(name="exponential_backoff", retry_limit=5, exponential_backoff=True,
                          backoff_base_minutes=5.0, backoff_multiplier=2.0, backoff_max_interval_minutes=240.0,
                          maximum_recovery_window=1440.0)


def intelligent_recovery() -> RecoveryPolicy:
    """C. Classification-aware: back off only on transient failures, ask the buyer to act on
    buyer-actionable failures (incl. alternate funding source), never retry terminal failures."""
    return RecoveryPolicy(name="intelligent_recovery", retry_limit=5, exponential_backoff=True,
                          backoff_base_minutes=5.0, backoff_multiplier=2.0, backoff_max_interval_minutes=240.0,
                          maximum_recovery_window=4320.0, buyer_action_timeout=1440.0,
                          request_buyer_action=True, alternate_funding_action=AlternateFundingAction.REQUEST_ALTERNATE,
                          terminal_failure_handling=TerminalFailureHandling.FLAG_FOR_REVIEW,
                          classification_aware=True, idempotent_retries=True)


def no_recovery() -> RecoveryPolicy:
    """Reference point: never recover anything."""
    return RecoveryPolicy(name="no_recovery", retry_limit=0, classification_aware=True)


BASELINES = {
    "immediate_retry": immediate_retry,
    "exponential_backoff": exponential_backoff,
    "intelligent_recovery": intelligent_recovery,
    "no_recovery": no_recovery,
}


def resolve_policy(base: str | None, **overrides) -> RecoveryPolicy:
    """Build a policy from an optional baseline plus overrides; raises ValueError if invalid."""
    if base is not None and base not in BASELINES:
        raise ValueError(f"unknown baseline {base!r}; choose from {sorted(BASELINES)}")
    policy = BASELINES[base]() if base else RecoveryPolicy()
    policy = replace(policy, **{k: v for k, v in overrides.items() if v is not None})
    problems = policy.validate()
    if problems:
        raise ValueError("Invalid RecoveryPolicy: " + "; ".join(problems))
    return policy
