"""Connect a SELECTED recovery policy to genuine PayPal Sandbox operations.

Safety rules enforced here regardless of what the simulated policy would do:
  * Sandbox only (guaranteed by PayPalConfig/PayPalClient).
  * One execution per (experiment_id, reference_id): re-running never creates a second order or capture.
  * Never capture an order that is not APPROVED; never capture a COMPLETED one.
  * After any ambiguous failure (timeout/5xx/in-progress/already-captured) GET the order BEFORE any further write.
  * A retry re-sends the SAME logical operation with the SAME PayPal-Request-Id.
  * Buyer-action and terminal failures are NEVER automatically retried; the workflow stops (awaiting buyer / terminated).
  * A simulated action is only executed if PayPal can actually perform it (ACTION_SUPPORT); the rest are recorded, not faked.
Every API exchange, state transition and predicted-vs-actual decision is persisted (paypal.persistence).

"Predicted action" = what the SIMULATION's policy logic (engine.payments.recovery_policy.next_action) would do for the
category observed from the real API response. This compares ACTIONS for single observations; it never compares or blends
success RATES.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Optional

from engine.payments.failure_model import FailureRules
from engine.payments.models import FailureCategory
from engine.payments.recovery_policy import (ActionKind, AlternateFundingAction, RecoveryPolicy, TerminalFailureHandling,
                                             next_action)
from paypal.capture import CapturesApi, verify_capture
from paypal.client import PayPalClient
from paypal.error_classifier import Classification, RecoveryClass, classify
from paypal.models import (Order, OrderState, PayPalApiError, VerificationResult, validate_amount, validate_reference_id)
from paypal.orders import OrdersApi
from paypal.persistence import SandboxStore

NAMESPACE = uuid.UUID("6f1d9b0e-4a39-4c53-9d5a-1f6f6a0c2b11")

# --- which SIMULATED recovery actions can really be executed through PayPal ------------------------------------
ACTION_SUPPORT: list[dict[str, Any]] = [
    {"simulated_action": "Automatic retry after a transient failure (network/timeout/HTTP 500/in-progress)", "executable_via_paypal": True,
     "how": "Re-send the same operation with the same PayPal-Request-Id after a GET-order state check",
     "limits": "AUTO_RETRY-class failures only; at most min(policy.retry_limit, hard cap) attempts"},
    {"simulated_action": "Retry delay / exponential backoff / recovery window / retry limit", "executable_via_paypal": True,
     "how": "Scheduled client-side by the adapter (delays scaled to real seconds and capped for Sandbox runs)", "limits": "policy minutes are scaled; see seconds_per_policy_minute"},
    {"simulated_action": "Idempotent retries", "executable_via_paypal": True, "how": "PayPal-Request-Id header derived deterministically per (execution, operation)",
     "limits": "docs: safe for POST retries after timeouts/HTTP 500; capture-order not named explicitly"},
    {"simulated_action": "Request buyer action (re-approve / alternate funding source)", "executable_via_paypal": "PARTIAL",
     "how": "Adapter returns the order's approve/payer-action URL and stops; the buyer acts inside PayPal; resume() re-checks state and captures",
     "limits": "Notifying the buyer is a merchant-side action NOT performed here; 'alternate funding' = buyer choosing another source during approval; not exercised end-to-end"},
    {"simulated_action": "Flag terminal failure for manual review", "executable_via_paypal": False, "how": "Merchant-side bookkeeping only (recorded in sandbox_executions/transitions)", "limits": "no PayPal API involved"},
    {"simulated_action": "Blind automatic retry of buyer-action or terminal failures", "executable_via_paypal": False,
     "how": "REFUSED by the executor (recorded as BLOCKED_RETRY)", "limits": "never executed"},
    {"simulated_action": "Re-capture without first checking order state", "executable_via_paypal": False, "how": "REFUSED", "limits": "never executed"},
]

_POLICY_FIELDS = ("name", "retry_limit", "retry_intervals", "exponential_backoff", "backoff_base_minutes", "backoff_multiplier",
                  "backoff_max_interval_minutes", "maximum_recovery_window", "buyer_action_timeout", "request_buyer_action",
                  "alternate_funding_action", "terminal_failure_handling", "classification_aware", "idempotent_retries")


class PolicyNotApproved(ValueError):
    """The policy is not a selected, validated, feasible policy."""


@dataclass(frozen=True)
class ExecutionPolicy:
    policy_id: str
    name: str
    definition: dict
    engine_policy: RecoveryPolicy
    source: str

    @classmethod
    def from_definition(cls, definition: dict, *, feasible: bool, source: str = "manual") -> "ExecutionPolicy":
        if not feasible:
            raise PolicyNotApproved("policy is not marked feasible under the merchant's hard constraints; refusing to execute it")
        fields = {k: definition[k] for k in _POLICY_FIELDS if k in definition}
        if "retry_intervals" in fields:
            fields["retry_intervals"] = tuple(fields["retry_intervals"])
        if "alternate_funding_action" in fields:
            fields["alternate_funding_action"] = AlternateFundingAction(fields["alternate_funding_action"])
        if "terminal_failure_handling" in fields:
            fields["terminal_failure_handling"] = TerminalFailureHandling(fields["terminal_failure_handling"])
        pol = RecoveryPolicy(**fields)
        problems = pol.validate()
        if problems:
            raise PolicyNotApproved("invalid policy: " + "; ".join(problems))
        canon = json.dumps({k: definition.get(k) for k in _POLICY_FIELDS}, sort_keys=True, default=str)
        return cls(hashlib.sha256(canon.encode()).hexdigest()[:16], pol.name, dict(definition), pol, source)

    @classmethod
    def from_decision_report(cls, report: dict) -> "ExecutionPolicy":
        """Build from an agent DecisionReport.to_dict(): requires a selected policy that is feasible in the final evaluation."""
        sel = report.get("selected_policy")
        if not sel:
            raise PolicyNotApproved("report has no selected policy")
        row = next((r for r in report.get("final_evaluation", {}).get("ranking", []) if r["policy"] == sel["name"]), None)
        if row is None or not row.get("feasible"):
            raise PolicyNotApproved("selected policy is not feasible in the report's final evaluation")
        return cls.from_definition(sel["definition"], feasible=True, source=f"agent_report:{report.get('execution_id')}")


@dataclass(frozen=True)
class PaymentRequest:
    amount: Any
    currency: str
    reference_id: str
    invoice_id: Optional[str] = None

    def __post_init__(self) -> None:
        validate_amount(self.amount, self.currency)
        validate_reference_id(self.reference_id)


class ExecStatus(str, Enum):
    NEW = "NEW"
    ORDER_CREATED = "ORDER_CREATED"
    AWAITING_BUYER = "AWAITING_BUYER"
    COMPLETED = "COMPLETED"
    TERMINATED = "TERMINATED"


@dataclass
class ExecutionResult:
    execution_id: str
    status: ExecStatus
    order_id: Optional[str]
    capture_id: Optional[str]
    approval_url: Optional[str]
    reason: Optional[str]
    verification: Optional[VerificationResult] = None
    last_classification: Optional[dict] = None
    new_api_calls: int = 0
    validation_record_ids: list[str] = field(default_factory=list)


class _Stop(Exception):
    def __init__(self, status: ExecStatus, reason: str):
        self.status, self.reason = status, reason


class RecoveryExecutor:
    def __init__(self, client: PayPalClient, store: SandboxStore, policy: ExecutionPolicy, experiment_id: str, *,
                 rules: Optional[FailureRules] = None, sleep: Callable[[float], None] = time.sleep,
                 seconds_per_policy_minute: float = 1.0, max_wait_s: float = 20.0, hard_max_retries: int = 3):
        self.client, self.store, self.policy, self.experiment_id = client, store, policy, experiment_id
        self.rules, self.sleep = rules or FailureRules(), sleep
        self.sec_per_min, self.max_wait_s, self.hard_max_retries = seconds_per_policy_minute, max_wait_s, hard_max_retries
        if client.sink is None:
            client.sink = store
        store.add_secrets(*client.config.secrets())
        self.orders, self.captures = OrdersApi(client), CapturesApi(client)
        self._records: list[str] = []
        self.last_verification: Optional[VerificationResult] = None

    # ------------------------------------------------------------------ public API
    def execute(self, req: PaymentRequest) -> ExecutionResult:
        amount, cur = validate_amount(req.amount, req.currency)
        ex, created = self.store.get_or_create_execution(self.experiment_id, req.reference_id, self.policy.policy_id, self.policy.name,
                                                         self.policy.definition, str(amount), cur)
        if not created:
            if Decimal(ex["amount"]) != amount or ex["currency"] != cur:
                raise ValueError("reference_id already used with a different amount/currency")
            if ex["policy_id"] != self.policy.policy_id:
                raise ValueError("reference_id already executing under a different policy")
        if ex["status"] in (ExecStatus.COMPLETED.value, ExecStatus.TERMINATED.value):
            return self._result(ex, 0)  # duplicate-execution prevention: no API calls, no new order/capture
        return self._advance(ex, req)

    def resume(self, reference_id: str) -> ExecutionResult:
        """Continue an execution that was waiting for the buyer: re-inspect the real order state, then capture if APPROVED."""
        ex = self.store.get_execution_by_reference(self.experiment_id, reference_id)
        if ex is None:
            raise ValueError(f"no execution for reference_id {reference_id!r}")
        if ex["status"] in (ExecStatus.COMPLETED.value, ExecStatus.TERMINATED.value):
            return self._result(ex, 0)
        return self._advance(ex, PaymentRequest(ex["amount"], ex["currency"], ex["reference_id"]))

    # ------------------------------------------------------------------ workflow
    def _advance(self, ex: dict, req: PaymentRequest) -> ExecutionResult:
        before = len(self.store.api_calls(ex["execution_id"]))
        self._records = []
        meta = {"execution_id": ex["execution_id"]}
        try:
            if not ex["order_id"]:
                order = self._guarded(ex, "CREATE_ORDER", lambda: self.orders.create_order(
                    req.amount, req.currency, req.reference_id, self._rid(ex, "create_order"), invoice_id=req.invoice_id, meta=meta), reconcile=None)
                self._set(ex, ExecStatus.ORDER_CREATED, "ORDER_CREATED", {"order_id": order.id, "order_status": order.raw_status},
                          order_id=order.id, approval_url=order.approval_url)
            order = self._guarded(ex, "GET_ORDER", lambda: self.orders.get_order(self._ex(ex)["order_id"], meta=meta), reconcile=None)
            self._dispatch(ex, order, req)
        except _Stop as s:
            fresh = self._ex(ex)
            if fresh["status"] != s.status.value:
                self._set(ex, s.status, "STOPPED", {"reason": s.reason})
            merged = json.loads(fresh["last_error_json"]) if fresh.get("last_error_json") else {}
            merged["reason"] = s.reason  # keep the stored classification, add the stop reason
            self.store.update_execution(ex["execution_id"], last_error_json=merged)
        return self._result(self._ex(ex), len(self.store.api_calls(ex["execution_id"])) - before)

    def _dispatch(self, ex: dict, order: Order, req: PaymentRequest) -> None:
        st = order.state
        if st == OrderState.COMPLETED:
            self._finish_verified(ex, order, req, first_attempt=False)
        elif st == OrderState.APPROVED:
            self._capture(ex, req)
        elif st in (OrderState.CREATED, OrderState.PAYER_ACTION_REQUIRED, OrderState.SAVED):
            self._await_buyer_approval(ex, order)
        elif st == OrderState.VOIDED:
            self._terminate(ex, "order_voided")
        else:
            self._terminate(ex, f"unrecognised_order_status:{order.raw_status}")

    # -- buyer approval is a STATE, not an error: the capture is not attempted --------------------------------
    def _await_buyer_approval(self, ex: dict, order: Order) -> None:
        cat = FailureCategory.ORDER_NOT_APPROVED
        predicted = next_action(self.policy.engine_policy, self.rules, cat, 0, 0.0, False)
        wants_buyer = predicted.kind == ActionKind.REQUEST_BUYER_ACTION
        self._set(ex, ExecStatus.AWAITING_BUYER, "BUYER_APPROVAL_REQUIRED", {"order_status": order.raw_status, "approval_url": order.approval_url},
                  approval_url=order.approval_url)
        self._record(ex, "GET_ORDER", f"order.status={order.raw_status}", "ORDER_NOT_APPROVED (state-derived; no capture attempted)",
                     predicted.kind.value, "AWAIT_BUYER_APPROVAL", "PENDING_BUYER",
                     predicted_text=f"simulation treats an unapproved order as ORDER_NOT_APPROVED (buyer-action class); policy would {predicted.kind.value}",
                     observed_text="order awaits buyer approval; adapter stopped without capturing and exposed the approval URL (buyer notification NOT sent)",
                     order_id=order.id, extra={"policy_requests_buyer_action": wants_buyer})
        raise _Stop(ExecStatus.AWAITING_BUYER, "buyer_approval_required")

    # -- capture ---------------------------------------------------------------------------------------------
    def _capture(self, ex: dict, req: PaymentRequest) -> None:
        meta = {"execution_id": ex["execution_id"]}

        def reconcile() -> Optional[Order]:
            o = self.orders.get_order(self._ex(ex)["order_id"], meta=meta)
            if o.state == OrderState.COMPLETED:
                return o
            if o.state != OrderState.APPROVED:  # no longer capturable by us (e.g. buyer must act again)
                self._dispatch(ex, o, req)
            return None

        # one buyer round per execution: if we already asked the buyer to act after a capture failure, do not ask again
        buyer_used = any(t["trigger"] == "BUYER_ACTION_REQUESTED" for t in self.store.transitions(ex["execution_id"]))
        done = self._guarded(ex, "CAPTURE_ORDER", lambda: self.orders.capture_order(self._ex(ex)["order_id"], self._rid(ex, "capture_order"), meta=meta),
                             reconcile=reconcile, buyer_used=buyer_used)
        self._finish_verified(ex, done, req, first_attempt=True)

    def _finish_verified(self, ex: dict, order: Order, req: PaymentRequest, first_attempt: bool) -> None:
        meta = {"execution_id": ex["execution_id"]}
        if not order.capture_ids:
            self._terminate(ex, "completed_order_without_capture_id")
        cap_id = order.capture_ids[0]
        capture = self._guarded(ex, "GET_CAPTURE", lambda: self.captures.get_capture(cap_id, meta=meta), reconcile=None)
        vr = self.last_verification = verify_capture(order, capture, expected_amount=req.amount, expected_currency=req.currency,
                                                     expected_reference_id=req.reference_id)
        retried = any(t["trigger"] == "RETRY_SCHEDULED" for t in self.store.transitions(ex["execution_id"]))
        buyer = any(t["trigger"] == "BUYER_APPROVAL_REQUIRED" for t in self.store.transitions(ex["execution_id"]))
        if vr.ok:
            self._set(ex, ExecStatus.COMPLETED, "CAPTURE_VERIFIED", {"capture_id": cap_id, "details": dict(vr.details)}, capture_id=cap_id)
            outcome = "NO_RECOVERY_NEEDED" if (first_attempt and not retried and not buyer) else "CONSISTENT"
            self._record(ex, "GET_CAPTURE", "capture.status=COMPLETED", None, "payment completes", "VERIFIED_COMPLETED", outcome,
                         predicted_text="simulation: a payment either succeeds on the first attempt or is recovered by the policy (single observation; no rate inference)",
                         observed_text=("completed after recovery actions (retry and/or buyer approval)" if outcome == "CONSISTENT" else "completed without any recovery action"),
                         order_id=order.id, capture_id=cap_id)
            return
        self._set(ex, ExecStatus.TERMINATED, "VERIFICATION_FAILED", {"problems": list(vr.problems)}, capture_id=cap_id)
        self._record(ex, "GET_CAPTURE", f"capture.status={capture.get('status')}", None, "payment completes", "NONE_VERIFICATION_FAILED", "DIVERGED",
                     predicted_text="simulation expects a completed capture to match the requested amount", observed_text="; ".join(vr.problems),
                     order_id=order.id, capture_id=cap_id)
        raise _Stop(ExecStatus.TERMINATED, "capture_verification_failed")

    # ------------------------------------------------------------------ guarded call with recovery decisions
    def _guarded(self, ex: dict, op: str, fn: Callable[[], Any], reconcile: Optional[Callable[[], Optional[Order]]], buyer_used: bool = False) -> Any:
        retries, elapsed = 0, 0.0
        while True:
            try:
                return fn()
            except PayPalApiError as e:
                cls = classify(e)
                self._transition(ex, "API_ERROR", {"operation": op, "http_status": e.status, "name": e.name, "issues": list(e.issues),
                                                   "debug_id": e.debug_id, "classification": cls.to_dict()})
                self.store.update_execution(ex["execution_id"], last_error_json={"operation": op, "classification": cls.to_dict(), "debug_id": e.debug_id})
                # ORDER_ALREADY_CAPTURED / ambiguous outcomes: fetch the order BEFORE deciding anything else.
                if op == "CAPTURE_ORDER" and reconcile is not None and (cls.requires_state_check or cls.trigger == "ORDER_ALREADY_CAPTURED"):
                    found = reconcile()
                    if found is not None:
                        self._transition(ex, "RECONCILED_ALREADY_COMPLETED", {"order_id": found.id, "capture_ids": list(found.capture_ids)})
                        return found
                decision = self._decide(ex, op, e, cls, retries, elapsed, buyer_used)
                if decision["actual"] == "RETRY":
                    wait = min(decision["delay_minutes"] * self.sec_per_min, self.max_wait_s)
                    self._transition(ex, "RETRY_SCHEDULED", {"operation": op, "retry_number": retries + 1, "policy_delay_minutes": decision["delay_minutes"],
                                                             "actual_wait_s": wait, "same_request_id": True})
                    self.sleep(wait)
                    retries += 1
                    elapsed += decision["delay_minutes"]
                    continue
                if decision["actual"] == "REQUEST_BUYER_ACTION":
                    o = reconcile_state(self, ex)
                    self._set(ex, ExecStatus.AWAITING_BUYER, "BUYER_ACTION_REQUESTED",
                              {"category": cls.category, "approval_url": o.approval_url if o else None, "buyer_notified": False},
                              approval_url=(o.approval_url if o else None))
                    raise _Stop(ExecStatus.AWAITING_BUYER, f"buyer_action_required:{cls.category}")
                self._set(ex, ExecStatus.TERMINATED, "TERMINATED", {"reason": decision["reason"], "category": cls.category})
                raise _Stop(ExecStatus.TERMINATED, decision["reason"])

    def _decide(self, ex: dict, op: str, e: PayPalApiError, cls: Classification, retries: int, elapsed: float, buyer_used: bool) -> dict:
        """predicted = simulated policy logic; actual = what is permitted/executable against real PayPal."""
        cat_name = cls.category or ("TEMPORARY_PROCESSING_ERROR" if cls.trigger == "PREVIOUS_REQUEST_IN_PROGRESS" else None)
        status_txt = f"HTTP {e.status} {e.name} {list(e.issues)}" if not e.transport else f"transport:{e.name}"
        if cat_name is None:
            actual, reason, predicted_kind, delay = "TERMINATE", f"not_a_payment_failure:{cls.trigger}", None, 0.0
            outcome = "NOT_COMPARABLE"
        else:
            predicted = next_action(self.policy.engine_policy, self.rules, FailureCategory[cat_name], retries, elapsed, buyer_used)
            predicted_kind, delay = predicted.kind.value, predicted.delay_minutes
            actual, reason = "TERMINATE", predicted.reason
            if retries >= self.hard_max_retries and predicted.kind == ActionKind.RETRY:
                actual, reason = "TERMINATE", "executor_hard_retry_cap"
            elif cls.recovery_class in (RecoveryClass.AUTO_RETRY, RecoveryClass.RECONCILE):
                if predicted.kind == ActionKind.RETRY and cls.retry_same_request:
                    actual, reason = "RETRY", "scheduled_retry_same_request_id"
                else:
                    actual = "TERMINATE"
            elif cls.recovery_class == RecoveryClass.BUYER_ACTION:
                if predicted.kind == ActionKind.REQUEST_BUYER_ACTION:
                    actual, reason = "REQUEST_BUYER_ACTION", "buyer_action_required"
                elif predicted.kind == ActionKind.RETRY:
                    actual, reason = "BLOCKED_RETRY", "blocked:automatic_retry_of_buyer_action_failure"
            else:  # TERMINAL
                if predicted.kind == ActionKind.RETRY:
                    actual, reason = "BLOCKED_RETRY", "blocked:automatic_retry_of_terminal_failure"
            if actual == "BLOCKED_RETRY":
                outcome = "DIVERGED"
            elif actual == predicted.kind.value and actual == "REQUEST_BUYER_ACTION":
                outcome = "PENDING_BUYER"
            elif actual == predicted.kind.value:
                outcome = "CONSISTENT"
            else:
                outcome = "DIVERGED"
        self._record(ex, op, status_txt, cat_name, predicted_kind, actual, outcome,
                     predicted_text=(f"policy `{self.policy.name}` would {predicted_kind} (category {cat_name})" if predicted_kind else "no simulated category applies"),
                     observed_text=f"{status_txt}; classified {cls.recovery_class.value} [{cls.basis}]; executor action {actual} ({reason})",
                     order_id=self._ex(ex)["order_id"], extra={"classification": cls.to_dict(), "retries_so_far": retries, "debug_id": e.debug_id})
        return {"actual": actual, "reason": reason, "delay_minutes": delay}

    # ------------------------------------------------------------------ helpers
    def _rid(self, ex: dict, operation: str) -> str:
        """Deterministic idempotency key: the same logical operation always reuses the same PayPal-Request-Id."""
        return str(uuid.uuid5(NAMESPACE, f"{ex['execution_id']}:{operation}"))

    def _ex(self, ex: dict) -> dict:
        return self.store.get_execution(ex["execution_id"])

    def _set(self, ex: dict, status: ExecStatus, trigger: str, detail: dict, **cols: Any) -> None:
        prev = self._ex(ex)["status"]
        self.store.update_execution(ex["execution_id"], status=status.value, **cols)
        self.store.add_transition(ex["execution_id"], prev, status.value, trigger, detail)

    def _transition(self, ex: dict, trigger: str, detail: dict) -> None:
        cur = self._ex(ex)["status"]
        self.store.add_transition(ex["execution_id"], cur, cur, trigger, detail)

    def _terminate(self, ex: dict, reason: str) -> None:
        self._set(ex, ExecStatus.TERMINATED, "TERMINATED", {"reason": reason})
        raise _Stop(ExecStatus.TERMINATED, reason)

    def _record(self, ex: dict, op: str, api_status: str, observed_cat: Optional[str], predicted: Optional[str], actual: str, outcome: str, *,
                predicted_text: str, observed_text: str, order_id: Optional[str] = None, capture_id: Optional[str] = None,
                extra: Optional[dict] = None) -> None:
        rid = self.store.add_validation_record({
            "execution_id": ex["execution_id"], "experiment_id": self.experiment_id, "selected_policy_id": self.policy.policy_id,
            "order_id": order_id, "capture_id": capture_id, "api_operation": op, "actual_api_status": api_status,
            "observed_error_category": observed_cat, "predicted_action": predicted, "actual_action": actual,
            "predicted_vs_observed": {"predicted": predicted_text, "observed": observed_text, "policy": self.policy.name,
                                      "scope": "single Sandbox observation; not a rate; never combined with simulated populations", **(extra or {})},
            "validation_outcome": outcome})
        self._records.append(rid)

    def _result(self, ex: dict, new_calls: int) -> ExecutionResult:
        err = json.loads(ex["last_error_json"]) if ex.get("last_error_json") else None
        return ExecutionResult(ex["execution_id"], ExecStatus(ex["status"]), ex["order_id"], ex["capture_id"], ex["approval_url"],
                               (err or {}).get("reason"), self.last_verification if ex["status"] == ExecStatus.COMPLETED.value else None,
                               (err or {}).get("classification"), new_calls, list(self._records))


def reconcile_state(executor: RecoveryExecutor, ex: dict) -> Optional[Order]:
    """Best-effort GET of the order for the approval URL when handing over to the buyer (a failure here is non-fatal)."""
    try:
        return executor.orders.get_order(executor._ex(ex)["order_id"], meta={"execution_id": ex["execution_id"]})
    except PayPalApiError:
        return None
