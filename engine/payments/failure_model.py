"""Failure classification rules and the (ASSUMED) stochastic world model.

Two separate things live here:

1. FailureRules -- which recovery class each failure category belongs to
   (auto-retryable / needs buyer action / terminal). This is the
   *classification* a recovery policy may consult. It is configurable.
   The default mapping was checked (2026-11-11) against PayPal's public Orders v2
   docs (developer.paypal.com/api/rest/reference/orders/v2/errors/ and
   /api/rest/requests/) via a summarised fetch; it does not contradict them, but
   the docs are thin and parts were summary-sourced:
     - Timeouts / HTTP 500 INTERNAL_SERVER_ERROR: docs say such POST calls may be
       retried with the same PayPal-Request-Id           -> AUTO_RETRY
       (capture-order is not named explicitly on the idempotency page).
     - PAYER_ACTION_REQUIRED ("instruct the buyer to return to PayPal"),
       ORDER_NOT_APPROVED ("redirect the payer to the approve URL")  -> BUYER_ACTION.
     - INSTRUMENT_DECLINED: no retry guidance is documented; PAYMENT_SOURCE_* codes say
       "try again with a different payment source by creating a new order" -> BUYER_ACTION.
     - Request-fix errors (e.g. DUPLICATE_INVOICE_ID, ORDER_ALREADY_CAPTURED) and
       TRANSACTION_REFUSED / PAYMENT_DENIED                -> TERMINAL for automatic handling.
     - EXPIRED_PAYMENT_METHOD is a MODELLING category, not a PayPal code; nearest
       documented codes are CARD_EXPIRED / REFERENCED_CARD_EXPIRED.
   Numeric HTTP statuses per code were not confirmed. Live Sandbox behaviour is unverified.

2. FailureModel -- the probabilities and time constants that drive the
   simulated world. EVERY number here is an ASSUMPTION chosen to be
   plausible and to make trade-offs visible. None is an empirical payment
   statistic and none describes PayPal.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Mapping

import numpy as np

from engine.payments.distributions import categorical_from_uniform, exponential_from_uniform
from engine.payments.models import FailureCategory as FC, PaymentMethodCategory as PM

METHODS = list(PM)
CATEGORIES = list(FC)
MAX_RETRY_DRAWS = 20  # latent uniforms reserved per transaction for retry attempts


class RecoveryClass(str, Enum):
    AUTO_RETRY = "AUTO_RETRY"  # same request may succeed later without anyone else acting
    BUYER_ACTION = "BUYER_ACTION"  # needs the payer (new funding source / approval / updated instrument)
    TERMINAL = "TERMINAL"  # automatic recovery is not possible (merchant must fix or abandon)


DEFAULT_CLASSIFICATION: dict[FC, RecoveryClass] = {
    FC.TRANSIENT_NETWORK_ERROR: RecoveryClass.AUTO_RETRY,
    FC.TEMPORARY_PROCESSING_ERROR: RecoveryClass.AUTO_RETRY,
    FC.INSTRUMENT_DECLINED: RecoveryClass.BUYER_ACTION,
    FC.EXPIRED_PAYMENT_METHOD: RecoveryClass.BUYER_ACTION,
    FC.PAYER_ACTION_REQUIRED: RecoveryClass.BUYER_ACTION,
    FC.ORDER_NOT_APPROVED: RecoveryClass.BUYER_ACTION,
    FC.VALIDATION_ERROR: RecoveryClass.TERMINAL,
    FC.TERMINAL_PAYMENT_FAILURE: RecoveryClass.TERMINAL,
}

# Categories whose buyer action is "use a different / updated funding source".
FUNDING_SOURCE_CATEGORIES = frozenset({FC.INSTRUMENT_DECLINED, FC.EXPIRED_PAYMENT_METHOD})


@dataclass(frozen=True)
class FailureRules:
    classification: Mapping[FC, RecoveryClass] = field(default_factory=lambda: dict(DEFAULT_CLASSIFICATION))

    def class_of(self, category: FC) -> RecoveryClass:
        return self.classification[category]

    def validate(self) -> list[str]:
        return [f"classification missing for {c.value}" for c in FC if c not in self.classification]


def _mix(**w: float) -> dict[FC, float]:
    return {FC[k.upper()]: v for k, v in w.items()}


@dataclass(frozen=True)
class FailureModel:
    """ASSUMED world parameters. See module docstring."""

    initial_failure_prob: Mapping[PM, float] = field(default_factory=lambda: {
        PM.CARD: 0.08, PM.BANK_ACCOUNT: 0.10, PM.WALLET_BALANCE: 0.04, PM.PAY_LATER: 0.12})
    # P(category | initial failure, method); rows must sum to 1.
    category_mix: Mapping[PM, Mapping[FC, float]] = field(default_factory=lambda: {
        PM.CARD: _mix(transient_network_error=.10, temporary_processing_error=.10, instrument_declined=.33,
                      expired_payment_method=.13, payer_action_required=.08, order_not_approved=.10,
                      validation_error=.06, terminal_payment_failure=.10),
        PM.BANK_ACCOUNT: _mix(transient_network_error=.08, temporary_processing_error=.14, instrument_declined=.30,
                              expired_payment_method=.05, payer_action_required=.12, order_not_approved=.10,
                              validation_error=.08, terminal_payment_failure=.13),
        PM.WALLET_BALANCE: _mix(transient_network_error=.14, temporary_processing_error=.12, instrument_declined=.24,
                                expired_payment_method=.03, payer_action_required=.14, order_not_approved=.18,
                                validation_error=.07, terminal_payment_failure=.08),
        PM.PAY_LATER: _mix(transient_network_error=.07, temporary_processing_error=.08, instrument_declined=.36,
                           expired_payment_method=.02, payer_action_required=.15, order_not_approved=.12,
                           validation_error=.05, terminal_payment_failure=.15),
    })
    # Transient outage persistence: time (minutes) until the underlying condition clears ~ Exponential(mean).
    mean_clear_minutes: Mapping[FC, float] = field(default_factory=lambda: {
        FC.TRANSIENT_NETWORK_ERROR: 3.0, FC.TEMPORARY_PROCESSING_ERROR: 20.0,
        FC.INSTRUMENT_DECLINED: 0.0, FC.EXPIRED_PAYMENT_METHOD: 0.0, FC.PAYER_ACTION_REQUIRED: 0.0,
        FC.ORDER_NOT_APPROVED: 0.0, FC.VALIDATION_ERROR: 0.0, FC.TERMINAL_PAYMENT_FAILURE: 0.0})
    # P(an automatic retry succeeds | condition cleared). Non-transient categories: ~never without buyer action.
    retry_success_after_clear: Mapping[FC, float] = field(default_factory=lambda: {
        FC.TRANSIENT_NETWORK_ERROR: 0.90, FC.TEMPORARY_PROCESSING_ERROR: 0.90,
        FC.INSTRUMENT_DECLINED: 0.03, FC.EXPIRED_PAYMENT_METHOD: 0.0, FC.PAYER_ACTION_REQUIRED: 0.0,
        FC.ORDER_NOT_APPROVED: 0.0, FC.VALIDATION_ERROR: 0.0, FC.TERMINAL_PAYMENT_FAILURE: 0.0})
    buyer_response_prob: Mapping[FC, float] = field(default_factory=lambda: {
        FC.INSTRUMENT_DECLINED: 0.55, FC.EXPIRED_PAYMENT_METHOD: 0.45,
        FC.PAYER_ACTION_REQUIRED: 0.60, FC.ORDER_NOT_APPROVED: 0.35})
    buyer_response_mean_minutes: Mapping[FC, float] = field(default_factory=lambda: {
        FC.INSTRUMENT_DECLINED: 120.0, FC.EXPIRED_PAYMENT_METHOD: 240.0,
        FC.PAYER_ACTION_REQUIRED: 90.0, FC.ORDER_NOT_APPROVED: 180.0})
    post_buyer_success_prob: Mapping[FC, float] = field(default_factory=lambda: {
        FC.INSTRUMENT_DECLINED: 0.85, FC.EXPIRED_PAYMENT_METHOD: 0.90,
        FC.PAYER_ACTION_REQUIRED: 0.92, FC.ORDER_NOT_APPROVED: 0.92})
    # If a retry after a network error is sent WITHOUT an idempotency key, chance the original had in fact
    # been processed (duplicate-capture risk).
    duplicate_capture_prob_without_idempotency: float = 0.10

    def validate(self) -> list[str]:
        p: list[str] = []
        for m in PM:
            if m not in self.initial_failure_prob or not 0 <= self.initial_failure_prob[m] <= 1:
                p.append(f"initial_failure_prob invalid for {m.value}")
            mix = self.category_mix.get(m)
            if mix is None or abs(sum(mix.get(c, 0.0) for c in FC) - 1.0) > 1e-6 or any(v < 0 for v in mix.values()):
                p.append(f"category_mix for {m.value} must be non-negative and sum to 1")
        for name in ("mean_clear_minutes", "retry_success_after_clear"):
            d = getattr(self, name)
            for c in FC:
                if c not in d or d[c] < 0 or (name == "retry_success_after_clear" and d[c] > 1):
                    p.append(f"{name} invalid for {c.value}")
        for name in ("buyer_response_prob", "post_buyer_success_prob"):
            if any(not 0 <= v <= 1 for v in getattr(self, name).values()):
                p.append(f"{name} values must be in [0,1]")
        if any(v <= 0 for v in self.buyer_response_mean_minutes.values()):
            p.append("buyer_response_mean_minutes must be > 0")
        if not 0 <= self.duplicate_capture_prob_without_idempotency <= 1:
            p.append("duplicate_capture_prob_without_idempotency must be in [0,1]")
        return p

    def scaled(self, initial_failure_multiplier: float = 1.0,
               category_weight_multipliers: Mapping[FC, float] | None = None) -> "FailureModel":
        """Return a copy with initial failure rates scaled and/or category mixes re-weighted (renormalised)."""
        ifp = {m: min(1.0, v * initial_failure_multiplier) for m, v in self.initial_failure_prob.items()}
        mult = category_weight_multipliers or {}
        mix = {}
        for m, row in self.category_mix.items():
            w = {c: row.get(c, 0.0) * mult.get(c, 1.0) for c in FC}
            s = sum(w.values())
            mix[m] = {c: v / s for c, v in w.items()}
        return replace(self, initial_failure_prob=ifp, category_mix=mix)


@dataclass(frozen=True)
class LatentOutcomes:
    """Per-transaction pre-drawn randomness ("the world"), independent of any policy.

    Because every policy is evaluated against the SAME latent draws for a
    given trial seed (common random numbers), policy differences are not
    sampling noise: the same transaction faces the same outage duration,
    the same buyer behaviour and the same per-attempt luck.
    """

    initial_failed: np.ndarray  # bool (N,)
    category_idx: np.ndarray  # int (N,), index into CATEGORIES (only meaningful where initial_failed)
    clear_minutes: np.ndarray  # float (N,) time after first failure when the transient condition clears
    buyer_responds_u: np.ndarray
    buyer_time_u: np.ndarray
    post_buyer_u: np.ndarray
    duplicate_u: np.ndarray
    retry_u: np.ndarray  # (N, MAX_RETRY_DRAWS)


def draw_latent(rng: np.random.Generator, method_idx: np.ndarray, model: FailureModel) -> LatentOutcomes:
    n = len(method_idx)
    u = rng.random((n, 7 + MAX_RETRY_DRAWS))
    p_fail = np.array([model.initial_failure_prob[m] for m in METHODS])[method_idx]
    initial_failed = u[:, 0] < p_fail

    category_idx = np.zeros(n, dtype=int)
    for mi, m in enumerate(METHODS):
        sel = method_idx == mi
        if sel.any():
            probs = np.array([model.category_mix[m].get(c, 0.0) for c in CATEGORIES])
            category_idx[sel] = categorical_from_uniform(u[sel, 1], probs)

    mean_clear = np.array([model.mean_clear_minutes[c] for c in CATEGORIES])[category_idx]
    clear = exponential_from_uniform(u[:, 2], mean_clear)
    return LatentOutcomes(initial_failed, category_idx, clear, u[:, 3], u[:, 4], u[:, 5], u[:, 6], u[:, 7:])
