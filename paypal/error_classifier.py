"""Classify REAL PayPal API failures into recovery classes.

Output is compatible with the simulation's failure categories (same names) so predicted and observed
behaviour can be compared, but this module does NOT import the simulation.

Every table row carries its evidence `basis`:
  DOCUMENTED            the error code/guidance appears in PayPal's public Orders v2 docs (as read on 2026-11-11;
                        parts were summary-sourced -- see docs/PAYPAL_SANDBOX_INTEGRATION.md)
  MODELLING_ASSUMPTION  our mapping of a documented code to a recovery class (PayPal gives no guidance), or an
                        undocumented-but-conventional behaviour (e.g. 429/502/503/504 are retryable)
  UNKNOWN_CONSERVATIVE  not recognised: treated as terminal, never auto-retried
No error code here is invented: unrecognised codes fall through to UNKNOWN_CONSERVATIVE.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from paypal.models import PayPalApiError


class RecoveryClass(str, Enum):
    AUTO_RETRY = "AUTO_RETRY"  # same request may succeed later (after checking state; same PayPal-Request-Id)
    BUYER_ACTION = "BUYER_ACTION"  # needs the buyer (approve / other funding source / updated instrument)
    TERMINAL = "TERMINAL"  # no automatic recovery
    RECONCILE = "RECONCILE"  # state conflict: fetch the order/capture, do not re-send blindly


@dataclass(frozen=True)
class Classification:
    category: Optional[str]  # simulation-compatible FailureCategory name, or None (auth/config)
    recovery_class: RecoveryClass
    retry_same_request: bool  # a retry (after reconciliation) with the SAME PayPal-Request-Id is permitted
    requires_state_check: bool  # outcome may be ambiguous: GET the order before any further write
    basis: str
    trigger: str  # what matched (issue code / error name / HTTP status / transport)
    note: str = ""

    def to_dict(self) -> dict:
        return {"category": self.category, "recovery_class": self.recovery_class.value, "retry_same_request": self.retry_same_request,
                "requires_state_check": self.requires_state_check, "basis": self.basis, "trigger": self.trigger, "note": self.note}


D, M, U = "DOCUMENTED", "MODELLING_ASSUMPTION", "UNKNOWN_CONSERVATIVE"
BUYER, TERM, AUTO, REC = RecoveryClass.BUYER_ACTION, RecoveryClass.TERMINAL, RecoveryClass.AUTO_RETRY, RecoveryClass.RECONCILE

# issue / error-name code -> (category, class, retry_same_request, state_check, basis, note)
CODE_TABLE: dict[str, tuple[Optional[str], RecoveryClass, bool, bool, str, str]] = {
    "INSTRUMENT_DECLINED": ("INSTRUMENT_DECLINED", BUYER, False, False, M, "code documented; docs give no retry guidance; buyer must use another funding source"),
    "PAYMENT_SOURCE_DECLINED_BY_PROCESSOR": ("INSTRUMENT_DECLINED", BUYER, False, False, D, "docs: try a different payment source by creating a new order"),
    "PAYMENT_SOURCE_CANNOT_BE_USED": ("INSTRUMENT_DECLINED", BUYER, False, False, D, "docs: try a different payment source by creating a new order"),
    "PAYER_ACTION_REQUIRED": ("PAYER_ACTION_REQUIRED", BUYER, False, False, D, "docs (summary-sourced): instruct the buyer to return to PayPal"),
    "ORDER_NOT_APPROVED": ("ORDER_NOT_APPROVED", BUYER, False, False, D, "docs (summary-sourced): redirect the payer to the approve URL"),
    "CARD_EXPIRED": ("EXPIRED_PAYMENT_METHOD", BUYER, False, False, M, "code documented; recovery class is our mapping"),
    "REFERENCED_CARD_EXPIRED": ("EXPIRED_PAYMENT_METHOD", BUYER, False, False, M, "code documented (summary-sourced); recovery class is our mapping"),
    "TOKEN_EXPIRED": ("EXPIRED_PAYMENT_METHOD", BUYER, False, False, M, "code documented; recovery class is our mapping"),
    "TRANSACTION_REFUSED": ("TERMINAL_PAYMENT_FAILURE", TERM, False, False, D, "docs: 'The request was refused.'"),
    "PAYMENT_DENIED": ("TERMINAL_PAYMENT_FAILURE", TERM, False, False, D, "docs: PayPal has declined to process this transaction"),
    "PAYER_ACCOUNT_RESTRICTED": ("TERMINAL_PAYMENT_FAILURE", TERM, False, False, M, "code documented; recovery class is our mapping"),
    "PAYER_CANNOT_PAY": ("TERMINAL_PAYMENT_FAILURE", TERM, False, False, M, "code documented; recovery class is our mapping"),
    "ORDER_EXPIRED": ("TERMINAL_PAYMENT_FAILURE", TERM, False, False, M, "code documented; a new order would be required"),
    "DUPLICATE_INVOICE_ID": ("VALIDATION_ERROR", TERM, False, False, D, "docs: invoice id must be unique; fix the request"),
    "PAYPAL_REQUEST_ID_REQUIRED": ("VALIDATION_ERROR", TERM, False, False, D, "docs: PayPal-Request-Id is required"),
    "MISSING_REQUIRED_PARAMETER": ("VALIDATION_ERROR", TERM, False, False, M, "request must be corrected"),
    "INVALID_PARAMETER_VALUE": ("VALIDATION_ERROR", TERM, False, False, M, "request must be corrected"),
    "INVALID_CURRENCY_CODE": ("VALIDATION_ERROR", TERM, False, False, M, "request must be corrected"),
    "ORDER_ALREADY_CAPTURED": (None, REC, False, True, D, "docs: one capture per order; fetch the order, do NOT capture again"),
    "PREVIOUS_REQUEST_IN_PROGRESS": (None, REC, True, True, D, "docs (summary-sourced): wait and try again; space out requests"),
    "INTERNAL_SERVER_ERROR": ("TEMPORARY_PROCESSING_ERROR", AUTO, True, True, D, "docs: HTTP 500 calls may be retried with the same PayPal-Request-Id"),
    "RATE_LIMIT_REACHED": ("TRANSIENT_NETWORK_ERROR", AUTO, True, True, M, "conventional rate-limit handling; not confirmed in the docs read"),
}

AUTH_NAMES = {"invalid_client", "AUTHENTICATION_FAILURE", "NOT_AUTHORIZED", "invalid_token", "PERMISSION_DENIED"}
VALIDATION_NAMES = {"INVALID_REQUEST", "MALFORMED_REQUEST_JSON", "RESOURCE_NOT_FOUND", "INVALID_RESOURCE_ID"}


def classify(err: PayPalApiError) -> Classification:
    """Classify a failed API exchange. Order: transport -> issue codes -> error name -> HTTP status -> unknown."""
    if err.transport:
        return Classification("TRANSIENT_NETWORK_ERROR", AUTO, True, True, D, "transport:" + err.name,
                              "timeout/connection failure: the request MAY have been processed; check order state before re-sending; docs: network timeouts may be retried with the same PayPal-Request-Id")
    for code in (*err.issues, err.name):
        if code in AUTH_NAMES:
            return Classification(None, TERM, False, False, M, code, "authentication/authorisation problem: fix configuration; not a payment failure")
    for code in (*err.issues, err.name):
        row = CODE_TABLE.get(code)
        if row:
            cat, cls, retry, check, basis, note = row
            return Classification(cat, cls, retry, check, basis, code, note)
    for code in (*err.issues, err.name):
        if code in VALIDATION_NAMES:
            return Classification("VALIDATION_ERROR", TERM, False, False, M, code, "request/resource problem")
    if err.status in (502, 503, 504) or (err.status is not None and err.status >= 500):
        return Classification("TEMPORARY_PROCESSING_ERROR", AUTO, True, True, D if err.status == 500 else M, f"http:{err.status}",
                              "server error: retry after state check with the same PayPal-Request-Id (docs name HTTP 500 only)")
    if err.status == 429:
        return Classification("TRANSIENT_NETWORK_ERROR", AUTO, True, True, M, "http:429", "rate limited")
    return Classification("TERMINAL_PAYMENT_FAILURE", TERM, False, False, U, f"unrecognised:{err.name}/{list(err.issues)}/{err.status}",
                          "unrecognised error: never auto-retried")


def classification_table() -> list[dict]:
    """Rows for documentation/tests: every recognised code and its basis."""
    return [{"code": c, "category": v[0], "recovery_class": v[1].value, "retry_same_request": v[2], "requires_state_check": v[3],
             "basis": v[4], "note": v[5]} for c, v in CODE_TABLE.items()]
