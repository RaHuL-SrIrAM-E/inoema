"""Sandbox failure-scenario harness.

PayPal documents negative testing for the Sandbox via the `PayPal-Mock-Response: {"mock_application_codes": "<CODE>"}`
request header (beta; requires a BUSINESS sandbox account with Negative Testing switched ON; Sandbox only). Orders v2
create/get/patch/authorize/capture are listed as supported endpoints, but PayPal does not document which codes work on which
endpoint -- and only DUPLICATE_INVOICE_ID appears in an example. Therefore NOTHING here is assumed to work:
every candidate code is *attempted* against the real Sandbox and only what PayPal actually returns is recorded.
`coverage_report` states, per simulation failure category, whether it was REPRODUCED, ATTEMPTED_NOT_REPRODUCED,
NOT_ATTEMPTED, or is SIMULATION_ONLY. With no recorded Sandbox attempts, every reproducible-in-principle category reads
NOT_ATTEMPTED -- never "verified".
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from paypal.error_classifier import CODE_TABLE, classify
from paypal.models import PayPalApiError
from paypal.orders import OrdersApi
from paypal.persistence import SandboxStore

# Candidate mock codes = codes that appear in PayPal's Orders v2 error documentation (as read 2026-11-11). Header support is UNVERIFIED.
CANDIDATE_MOCK_CODES: list[dict[str, str]] = [
    {"code": "DUPLICATE_INVOICE_ID", "doc_status": "the only value shown in PayPal's header example (capture endpoint)"},
    {"code": "INSTRUMENT_DECLINED", "doc_status": "documented Orders v2 error code; header support unverified"},
    {"code": "TRANSACTION_REFUSED", "doc_status": "documented Orders v2 error code; header support unverified"},
    {"code": "PAYER_ACCOUNT_RESTRICTED", "doc_status": "documented Orders v2 error code; header support unverified"},
    {"code": "PAYER_CANNOT_PAY", "doc_status": "documented Orders v2 error code; header support unverified"},
    {"code": "ORDER_NOT_APPROVED", "doc_status": "documented (summary-sourced); also the likely REAL error for an unapproved capture"},
    {"code": "PAYER_ACTION_REQUIRED", "doc_status": "documented (summary-sourced); header support unverified"},
    {"code": "INTERNAL_SERVER_ERROR", "doc_status": "documented Orders v2 error code; header support unverified"},
]

# Simulation categories that cannot be produced against the Sandbox (by any documented mechanism found) -> validated by simulation only.
SIMULATION_ONLY: dict[str, str] = {
    "TRANSIENT_NETWORK_ERROR": "network faults/timeouts cannot be induced against the Sandbox; no documented mock code",
    "EXPIRED_PAYMENT_METHOD": "no generic documented mock code found; CARD_EXPIRED/REFERENCED_CARD_EXPIRED exist as error codes but header support is unverified",
}
BASELINE_CODE = "(no mock header: real capture of an unapproved order)"


def run_negative_tests(orders: OrdersApi, store: SandboxStore, codes: Optional[list[str]] = None, *, amount: str = "10.00",
                       currency: str = "USD") -> list[dict[str, Any]]:
    """Attempt each candidate against the real Sandbox: fresh order, then capture with the mock header. Records the outcome
    honestly (reproduced = the mocked code appears among the returned issue codes)."""
    results = []
    plan = [None] + [c["code"] for c in CANDIDATE_MOCK_CODES if codes is None or c["code"] in codes]
    for code in plan:
        tag = uuid.uuid4().hex[:10]
        order = orders.create_order(amount, currency, f"negtest-{tag}", str(uuid.uuid4()))
        status = name = debug = None
        issues: tuple[str, ...] = ()
        success = False
        try:
            orders.capture_order(order.id, str(uuid.uuid4()), mock_code=code)
            success = True  # capture unexpectedly succeeded (should not happen for an unapproved order)
        except PayPalApiError as e:
            status, name, issues, debug = e.status, e.name, e.issues, e.debug_id
            cls = classify(e)
        rec = {"mock_code": code or BASELINE_CODE, "target_operation": "CAPTURE_ORDER", "observed_status": status, "observed_name": name,
               "observed_issues": list(issues), "paypal_debug_id": debug,
               "reproduced": bool(code and code in issues) if code else bool(issues),
               "detail": {"order_id": order.id, "capture_succeeded": success, "classified_as": None if success else cls.to_dict(),
                          "baseline": code is None}}
        store.add_negative_test(rec)
        results.append(rec)
    return results


def coverage_report(store: SandboxStore) -> list[dict[str, Any]]:
    """Per simulation failure category: what has actually been observed against the Sandbox."""
    tests = store.negative_tests()
    by_code: dict[str, list[dict]] = {}
    for t in tests:
        by_code.setdefault(t["mock_code"], []).append(t)
    observed_issue_codes = {i for t in tests for i in __import__("json").loads(t["observed_issues"] or "[]")}
    cats: dict[str, dict[str, Any]] = {}
    for code, row in CODE_TABLE.items():
        cat = row[0]
        if cat is None:
            continue
        c = cats.setdefault(cat, {"category": cat, "codes_observed": [], "codes_attempted": [], "status": "NOT_ATTEMPTED", "note": ""})
        if code in observed_issue_codes:
            c["codes_observed"].append(code)
        if code in by_code:
            c["codes_attempted"].append(code)
    for cat, c in cats.items():
        if cat in SIMULATION_ONLY:
            c["status"], c["note"] = "SIMULATION_ONLY", SIMULATION_ONLY[cat]
        elif c["codes_observed"]:
            c["status"] = "REPRODUCED"
        elif c["codes_attempted"]:
            c["status"] = "ATTEMPTED_NOT_REPRODUCED"
    for cat, why in SIMULATION_ONLY.items():
        cats.setdefault(cat, {"category": cat, "codes_observed": [], "codes_attempted": [], "status": "SIMULATION_ONLY", "note": why})
    return sorted(cats.values(), key=lambda c: c["category"])
