"""Explicit, safe "adopt an existing Sandbox order, then capture it once it is buyer-approved" workflow.

Unlike RecoveryExecutor this never creates an order and never retries: every run starts with a fresh GET of the real
order, verifies it is THE order the operator named, and captures only if (a) PayPal says APPROVED and (b) the operator
passed confirm_capture=True. Everything is persisted (api calls, execution row, transitions) via SandboxStore.

Safety rules (each covered by an offline test):
  * order id, reference/invoice id, amount, currency and intent must all match the operator's explicit inputs -> else REFUSED
  * status other than APPROVED -> refuse capture (CREATED/PAYER_ACTION_REQUIRED: buyer has not approved; VOIDED etc: terminal)
  * APPROVED without confirm_capture -> no capture
  * one capture request per run, with a stable PayPal-Request-Id derived from (execution, 'capture_order')
  * ambiguous capture outcome (timeout / 5xx / in-progress / already-captured) -> GET the order; never capture a second time in the same run
  * order already COMPLETED at PayPal -> never capture; only verify the existing capture
  * a locally COMPLETED or TERMINATED execution -> no API calls at all
  * final result is verified from a fresh GET of the order plus a GET of the capture
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from paypal.capture import CapturesApi, verify_capture
from paypal.client import PayPalClient
from paypal.error_classifier import RecoveryClass, classify
from paypal.execution import NAMESPACE, ExecStatus
from paypal.models import (Order, OrderState, PayPalApiError, VerificationResult, validate_amount, validate_order_id,
                           validate_reference_id)
from paypal.orders import OrdersApi
from paypal.persistence import SandboxStore

EXPERIMENT_ID = "manual-resume-capture"
POLICY_ID = "manual-resume"


class ResumeOutcome(str, Enum):
    CAPTURED_AND_VERIFIED = "CAPTURED_AND_VERIFIED"      # we captured it and the fresh GET verification passed
    ALREADY_CAPTURED_VERIFIED = "ALREADY_CAPTURED_VERIFIED"  # PayPal already had a capture; verified, nothing captured by this run
    NOT_APPROVED = "NOT_APPROVED"                        # refused: buyer has not approved yet
    CONFIRMATION_REQUIRED = "CONFIRMATION_REQUIRED"      # APPROVED but --confirm-capture not given: nothing captured
    MISMATCH = "MISMATCH"                                # refused: order is not the one the operator named
    ORDER_TERMINAL = "ORDER_TERMINAL"                    # refused: voided / unrecognised status
    AMBIGUOUS = "AMBIGUOUS"                              # capture outcome unclear; order re-fetched; no further capture this run
    CAPTURE_FAILED = "CAPTURE_FAILED"                    # PayPal rejected the capture; not retried
    VERIFICATION_FAILED = "VERIFICATION_FAILED"          # capture exists but does not verify; not retried
    ALREADY_FINISHED_LOCALLY = "ALREADY_FINISHED_LOCALLY"  # local execution already COMPLETED/TERMINATED: no API calls


@dataclass
class ResumeResult:
    outcome: ResumeOutcome
    execution_id: str
    order_id: str
    order_status: Optional[str] = None
    capture_id: Optional[str] = None
    reason: str = ""
    approval_url: Optional[str] = None
    problems: list[str] = field(default_factory=list)
    verification: Optional[VerificationResult] = None
    capture_requested: bool = False
    new_api_calls: int = 0

    @property
    def ok(self) -> bool:
        return self.outcome in (ResumeOutcome.CAPTURED_AND_VERIFIED, ResumeOutcome.ALREADY_CAPTURED_VERIFIED)


def check_identity(order: Order, *, order_id: str, reference: str, amount: Decimal, currency: str) -> list[str]:
    """Every discrepancy between the real PayPal order and what the operator explicitly named."""
    problems: list[str] = []
    pu = (order.raw.get("purchase_units") or [{}])[0]
    if order.id != order_id:
        problems.append(f"order id {order.id!r} != expected {order_id!r}")
    if order.reference_id != reference:
        problems.append(f"reference_id {order.reference_id!r} != expected {reference!r}")
    if pu.get("invoice_id") != reference:
        problems.append(f"invoice_id {pu.get('invoice_id')!r} != expected {reference!r}")
    if order.amount is None or order.amount != amount:
        problems.append(f"amount {order.amount} != expected {amount}")
    if order.currency != currency:
        problems.append(f"currency {order.currency!r} != expected {currency!r}")
    if order.raw.get("intent") != "CAPTURE":
        problems.append(f"intent {order.raw.get('intent')!r} != 'CAPTURE'")
    if len(pu.get("payments", {}).get("captures", [])) > 1:
        problems.append("order already has more than one capture")
    return problems


class ResumeCapture:
    def __init__(self, client: PayPalClient, store: SandboxStore):
        self.client, self.store = client, store
        if client.sink is None:
            client.sink = store
        store.add_secrets(*client.config.secrets())
        self.orders, self.captures = OrdersApi(client), CapturesApi(client)

    # ------------------------------------------------------------------ public
    def run(self, order_id: str, reference: str, amount: Any, currency: str, *, confirm_capture: bool = False) -> ResumeResult:
        order_id, reference = validate_order_id(order_id), validate_reference_id(reference)
        amt, cur = validate_amount(amount, currency)
        ex, _ = self.store.get_or_create_execution(EXPERIMENT_ID, reference, POLICY_ID, "manual_resume_capture",
                                                   {"workflow": "resume_and_capture", "order_id": order_id}, str(amt), cur)
        eid = ex["execution_id"]
        if ex["order_id"] and ex["order_id"] != order_id:
            raise ValueError(f"reference {reference!r} is already bound to a different order; refusing")
        if Decimal(ex["amount"]) != amt or ex["currency"] != cur:
            raise ValueError("reference already used with a different amount/currency; refusing")
        if ex["status"] in (ExecStatus.COMPLETED.value, ExecStatus.TERMINATED.value):
            return ResumeResult(ResumeOutcome.ALREADY_FINISHED_LOCALLY, eid, order_id, capture_id=ex["capture_id"],
                                reason=f"execution already {ex['status']} locally; no API call made (never re-capture a finished payment)")
        if not ex["order_id"]:
            self.store.update_execution(eid, order_id=order_id)
            self._to(eid, ex["status"], ExecStatus.ORDER_CREATED, "ORDER_ADOPTED", {"order_id": order_id, "note": "existing order; none created"})
        before = len(self.store.api_calls(eid))
        res = self._flow(eid, order_id, reference, amt, cur, confirm_capture)
        res.new_api_calls = len(self.store.api_calls(eid)) - before
        return res

    # ------------------------------------------------------------------ workflow
    def _flow(self, eid: str, order_id: str, reference: str, amt: Decimal, cur: str, confirm: bool) -> ResumeResult:
        meta = {"execution_id": eid}
        try:
            order = self.orders.get_order(order_id, meta=meta)  # FIRST thing: the real order, straight from PayPal
        except PayPalApiError as e:
            self._transition(eid, "API_ERROR", {"operation": "GET_ORDER", "http_status": e.status, "name": e.name, "debug_id": e.debug_id})
            return self._res(eid, order_id, ResumeOutcome.ORDER_TERMINAL if e.status == 404 else ResumeOutcome.AMBIGUOUS,
                             reason=f"could not retrieve the order: HTTP {e.status} {e.name}")
        problems = check_identity(order, order_id=order_id, reference=reference, amount=amt, currency=cur)
        if problems:
            self._transition(eid, "IDENTITY_MISMATCH_REFUSED", {"problems": problems})  # recorded refusal; not a terminal state
            return self._res(eid, order_id, ResumeOutcome.MISMATCH, order, problems=problems,
                             reason="the PayPal order does not match the explicitly supplied inputs; nothing captured")
        st = order.state
        if st == OrderState.COMPLETED:
            return self._verify_final(eid, order_id, reference, amt, cur, captured_by_us=False)
        if st in (OrderState.CREATED, OrderState.PAYER_ACTION_REQUIRED, OrderState.SAVED):
            self._set(eid, ExecStatus.AWAITING_BUYER, "BUYER_APPROVAL_REQUIRED", {"order_status": order.raw_status, "approval_url": order.approval_url},
                      approval_url=order.approval_url)
            return self._res(eid, order_id, ResumeOutcome.NOT_APPROVED, order, approval_url=order.approval_url,
                             reason=f"order status is {order.raw_status}, not APPROVED: the buyer has not approved it, so capture is refused")
        if st != OrderState.APPROVED:
            self._set(eid, ExecStatus.TERMINATED, "ORDER_TERMINAL", {"order_status": order.raw_status})
            return self._res(eid, order_id, ResumeOutcome.ORDER_TERMINAL, order,
                             reason=f"order status {order.raw_status} is not capturable; refusing")
        if not confirm:
            self._transition(eid, "CAPTURE_NOT_CONFIRMED", {"order_status": order.raw_status})
            return self._res(eid, order_id, ResumeOutcome.CONFIRMATION_REQUIRED, order,
                             reason="order is APPROVED and capturable, but --confirm-capture was not given: nothing captured")
        return self._capture(eid, order_id, reference, amt, cur)

    def _capture(self, eid: str, order_id: str, reference: str, amt: Decimal, cur: str) -> ResumeResult:
        meta = {"execution_id": eid}
        rid = str(uuid.uuid5(NAMESPACE, f"{eid}:capture_order"))  # stable: re-sending this logical capture is idempotent
        self._transition(eid, "CAPTURE_REQUESTED", {"request_id": rid, "confirmed": True})
        try:
            done = self.orders.capture_order(order_id, rid, meta=meta)
        except PayPalApiError as e:
            cls = classify(e)
            self._transition(eid, "API_ERROR", {"operation": "CAPTURE_ORDER", "http_status": e.status, "name": e.name, "issues": list(e.issues),
                                                "debug_id": e.debug_id, "classification": cls.to_dict()})
            self.store.update_execution(eid, last_error_json={"operation": "CAPTURE_ORDER", "classification": cls.to_dict(), "debug_id": e.debug_id})
            if cls.requires_state_check or cls.trigger == "ORDER_ALREADY_CAPTURED" or e.transport:
                # outcome unknown: look at the real order before anything else, and do NOT capture again in this run
                try:
                    now = self.orders.get_order(order_id, meta=meta)
                except PayPalApiError as e2:
                    return self._res(eid, order_id, ResumeOutcome.AMBIGUOUS, reason=f"capture outcome unknown and the order could not be re-fetched (HTTP {e2.status} {e2.name}); do not re-run blindly")
                self._transition(eid, "RECONCILE_AFTER_AMBIGUOUS_CAPTURE", {"order_status": now.raw_status})
                if now.state == OrderState.COMPLETED:
                    return self._verify_final(eid, order_id, reference, amt, cur, captured_by_us=False, reconciled=True)
                return self._res(eid, order_id, ResumeOutcome.AMBIGUOUS, now,
                                 reason=f"capture outcome was ambiguous (HTTP {e.status} {e.name}); order is now {now.raw_status}. No second capture was sent; re-run to re-check")
            if cls.recovery_class == RecoveryClass.BUYER_ACTION:
                now = self._safe_get(order_id, meta)
                self._set(eid, ExecStatus.AWAITING_BUYER, "BUYER_ACTION_REQUIRED", {"issues": list(e.issues)})
                return self._res(eid, order_id, ResumeOutcome.CAPTURE_FAILED, now, reason=f"PayPal needs buyer action: HTTP {e.status} {e.name} {list(e.issues)}; not retried")
            self._set(eid, ExecStatus.TERMINATED, "CAPTURE_REJECTED", {"http_status": e.status, "name": e.name, "issues": list(e.issues)})
            return self._res(eid, order_id, ResumeOutcome.CAPTURE_FAILED, reason=f"capture rejected: HTTP {e.status} {e.name} {list(e.issues)}; terminal, not retried")
        self._transition(eid, "CAPTURE_RESPONSE", {"order_status": done.raw_status, "capture_ids": list(done.capture_ids)})
        return self._verify_final(eid, order_id, reference, amt, cur, captured_by_us=True)

    def _verify_final(self, eid: str, order_id: str, reference: str, amt: Decimal, cur: str, *, captured_by_us: bool, reconciled: bool = False) -> ResumeResult:
        """Independent verification from a FRESH order GET and a capture GET."""
        meta = {"execution_id": eid}
        fresh = self.orders.get_order(order_id, meta=meta)
        problems = check_identity(fresh, order_id=order_id, reference=reference, amount=amt, currency=cur)
        if fresh.state != OrderState.COMPLETED or not fresh.capture_ids:
            problems.append(f"fresh order status {fresh.raw_status!r} / captures {len(fresh.capture_ids)}: not a completed single capture")
            self._set(eid, ExecStatus.TERMINATED, "VERIFICATION_FAILED", {"problems": problems})
            return self._res(eid, order_id, ResumeOutcome.VERIFICATION_FAILED, fresh, problems=problems, reason="final verification failed; not retried",
                             capture_requested=captured_by_us)
        capture = self.captures.get_capture(fresh.capture_ids[0], meta=meta)
        vr = verify_capture(fresh, capture, expected_amount=amt, expected_currency=cur, expected_reference_id=reference)
        problems += list(vr.problems)
        if problems:
            self._set(eid, ExecStatus.TERMINATED, "VERIFICATION_FAILED", {"problems": problems}, capture_id=vr.capture_id)
            return self._res(eid, order_id, ResumeOutcome.VERIFICATION_FAILED, fresh, problems=problems, verification=vr, capture_id=vr.capture_id,
                             reason="final verification failed; not retried", capture_requested=captured_by_us)
        self._set(eid, ExecStatus.COMPLETED, "CAPTURE_VERIFIED", {"capture_id": vr.capture_id, "details": dict(vr.details), "captured_by_this_run": captured_by_us,
                                                                    "reconciled_after_ambiguity": reconciled}, capture_id=vr.capture_id)
        out = ResumeOutcome.CAPTURED_AND_VERIFIED if captured_by_us else ResumeOutcome.ALREADY_CAPTURED_VERIFIED
        return self._res(eid, order_id, out, fresh, verification=vr, capture_id=vr.capture_id, capture_requested=captured_by_us,
                         reason="verified from a fresh GET of the order and the capture" + ("" if captured_by_us else " (capture already existed; this run sent no capture request)"))

    # ------------------------------------------------------------------ helpers
    def _safe_get(self, order_id: str, meta: dict) -> Optional[Order]:
        try:
            return self.orders.get_order(order_id, meta=meta)
        except PayPalApiError:
            return None

    def _transition(self, eid: str, trigger: str, detail: dict) -> None:
        cur = self.store.get_execution(eid)["status"]
        self.store.add_transition(eid, cur, cur, trigger, detail)

    def _to(self, eid: str, from_state: Optional[str], to: ExecStatus, trigger: str, detail: dict) -> None:
        self.store.update_execution(eid, status=to.value)
        self.store.add_transition(eid, from_state, to.value, trigger, detail)

    def _set(self, eid: str, to: ExecStatus, trigger: str, detail: dict, **fields: Any) -> None:
        cur = self.store.get_execution(eid)["status"]
        self.store.update_execution(eid, status=to.value, **fields)
        self.store.add_transition(eid, cur, to.value, trigger, detail)

    def _res(self, eid: str, order_id: str, outcome: ResumeOutcome, order: Optional[Order] = None, **kw: Any) -> ResumeResult:
        return ResumeResult(outcome, eid, order_id, order_status=order.raw_status if order else None, **kw)
