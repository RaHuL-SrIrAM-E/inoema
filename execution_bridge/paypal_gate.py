"""Where (and only where) a policy decision may touch the PayPal adapter. Pure refusal logic + one injected callable.

Nothing here imports httpx or builds a client. A real call can only happen if the caller injects the existing, verified
ResumeCapture.run and the state is a REAL, non-completed Sandbox payment that PayPal-side checks (fresh GET, identity, confirm) still guard.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from execution_bridge.executor import Action, Decision, PaymentState

REAL_SANDBOX = "REAL_SANDBOX"  # only a state the operator explicitly marks as a live Sandbox payment qualifies (REAL_SANDBOX_READONLY never does)


@dataclass(frozen=True)
class Support:
    supported: bool
    mechanism: str


@dataclass(frozen=True)
class Application:
    performed: bool
    detail: str
    result: Any = None


def paypal_support(decision: Decision, state: PaymentState) -> Support:
    if state.source != REAL_SANDBOX:
        return Support(False, "simulated or read-only state: no PayPal call is ever made")
    if state.order_status == "COMPLETED" or state.capture_id:
        return Support(False, "payment already completed: never retried or re-captured")
    if decision.action == Action.WAIT:
        return Support(True, "re-check by GET order (read-only) until the buyer approves")
    if decision.action == Action.REQUIRE_BUYER_ACTION:
        return Support(True, "expose the approval URL; buyer notification is NOT performed by INOEMA")
    if decision.action == Action.RETRY:
        if state.order_status != "APPROVED":
            return Support(False, "capture retry needs an APPROVED order")
        if not state.idempotency_key:
            return Support(False, "capture retry needs the stable idempotency key")
        return Support(True, "ResumeCapture.run(..., confirm_capture=True): fresh GET, identity checks, same idempotency key")
    return Support(False, "STOP: nothing is sent to PayPal")


def apply_to_paypal(decision: Decision, state: PaymentState, *, resume: Optional[Callable[..., Any]] = None, confirm: bool = False) -> Application:
    """Invoke the injected resume callable only for a supported RETRY on a REAL, APPROVED, not-completed payment, and only with explicit confirm."""
    sup = paypal_support(decision, state)
    if not sup.supported:
        return Application(False, f"not executed: {sup.mechanism}")
    if decision.action != Action.RETRY:
        return Application(False, f"no write needed: {sup.mechanism}")
    if not confirm or resume is None:
        return Application(False, "not executed: explicit confirmation and an injected resume callable are required")
    return Application(True, "executed via the existing resume-and-capture workflow", resume(confirm_capture=True))
