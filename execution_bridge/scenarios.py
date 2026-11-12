"""The fixed SIMULATED payment states used to demonstrate the policy-driven decisions. They are inputs we chose, not outcomes PayPal produced."""
from __future__ import annotations

from engine.payments.models import FailureCategory as FC

from execution_bridge.executor import FailureEvent, PaymentState

KEY = "demo-idempotency-key"
SCENARIOS: list[tuple[str, PaymentState, FailureEvent | None]] = [
    ("Transient network error on the first attempt", PaymentState("sim-1", "APPROVED", 0, 0.0, idempotency_key=KEY), FailureEvent(FC.TRANSIENT_NETWORK_ERROR)),
    ("Transient error, retry budget already spent", PaymentState("sim-2", "APPROVED", 2, 135.0, idempotency_key=KEY), FailureEvent(FC.TEMPORARY_PROCESSING_ERROR)),
    ("Retryable failure but no idempotency key", PaymentState("sim-3", "APPROVED", 0, 0.0), FailureEvent(FC.TRANSIENT_NETWORK_ERROR)),
    ("Instrument declined (buyer-actionable)", PaymentState("sim-4", "APPROVED", 0, 10.0, idempotency_key=KEY), FailureEvent(FC.INSTRUMENT_DECLINED)),
    ("Buyer asked to act; still inside the timeout", PaymentState("sim-5", "PAYER_ACTION_REQUIRED", 0, 600.0, True, 10.0, KEY), FailureEvent(FC.INSTRUMENT_DECLINED)),
    ("Buyer asked to act; timeout exceeded", PaymentState("sim-6", "PAYER_ACTION_REQUIRED", 0, 3000.0, True, 10.0, KEY), FailureEvent(FC.INSTRUMENT_DECLINED)),
    ("Terminal payment failure", PaymentState("sim-7", "APPROVED", 0, 5.0, idempotency_key=KEY), FailureEvent(FC.TERMINAL_PAYMENT_FAILURE)),
    ("Order created, buyer has not approved", PaymentState("sim-8", "CREATED", 0, 0.0, idempotency_key=KEY), None),
    ("Payment already completed and captured", PaymentState("sim-9", "COMPLETED", 0, 0.0, capture_id="SIMULATED-CAPTURE", idempotency_key=KEY), FailureEvent(FC.TRANSIENT_NETWORK_ERROR)),
]
