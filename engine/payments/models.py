"""Core types for the INOEMA payment-recovery simulation.

Everything produced by this package is SIMULATED merchant-side recovery
behaviour driven by ASSUMED distributions. It is not PayPal data and does
not model PayPal's proprietary authorization or risk systems. Outputs must
never be stored alongside, or presented as, PayPal Sandbox observations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

SIM_LABEL = "SIMULATED_MERCHANT_RECOVERY_NOT_PAYPAL"
ASSUMPTION_LABEL = "ASSUMED_DISTRIBUTIONS_NOT_EMPIRICAL"
MODEL_VERSION = "payments-sim-v1"


class PaymentMethodCategory(str, Enum):
    CARD = "CARD"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    WALLET_BALANCE = "WALLET_BALANCE"
    PAY_LATER = "PAY_LATER"


class FailureCategory(str, Enum):
    TRANSIENT_NETWORK_ERROR = "TRANSIENT_NETWORK_ERROR"
    TEMPORARY_PROCESSING_ERROR = "TEMPORARY_PROCESSING_ERROR"
    INSTRUMENT_DECLINED = "INSTRUMENT_DECLINED"
    EXPIRED_PAYMENT_METHOD = "EXPIRED_PAYMENT_METHOD"
    PAYER_ACTION_REQUIRED = "PAYER_ACTION_REQUIRED"
    ORDER_NOT_APPROVED = "ORDER_NOT_APPROVED"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    TERMINAL_PAYMENT_FAILURE = "TERMINAL_PAYMENT_FAILURE"


class PaymentStatus(str, Enum):
    APPROVED = "APPROVED"
    FAILED = "FAILED"


class FinalStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class RecoveryStatus(str, Enum):
    NOT_NEEDED = "NOT_NEEDED"  # approved on the first attempt
    RECOVERED = "RECOVERED"
    UNRECOVERED = "UNRECOVERED"
    ESCALATED_FOR_REVIEW = "ESCALATED_FOR_REVIEW"  # unrecovered + flagged for a human


class EventType(str, Enum):
    TRANSACTION_CREATED = "TRANSACTION_CREATED"
    PAYMENT_ATTEMPTED = "PAYMENT_ATTEMPTED"
    PAYMENT_APPROVED = "PAYMENT_APPROVED"
    PAYMENT_FAILED = "PAYMENT_FAILED"
    FAILURE_CLASSIFIED = "FAILURE_CLASSIFIED"
    RECOVERY_SCHEDULED = "RECOVERY_SCHEDULED"
    RETRY_ATTEMPTED = "RETRY_ATTEMPTED"
    BUYER_ACTION_REQUESTED = "BUYER_ACTION_REQUESTED"
    BUYER_ACTION_COMPLETED = "BUYER_ACTION_COMPLETED"
    PAYMENT_RECOVERED = "PAYMENT_RECOVERED"
    PAYMENT_TERMINATED = "PAYMENT_TERMINATED"


@dataclass(frozen=True)
class CostModel:
    """Operational cost assumptions (currency units). ASSUMPTIONS, not PayPal fees.

    Only merchant-side operational costs are modelled; percentage processing
    fees on successful payments are deliberately excluded.
    """

    attempt_cost: float = 0.03  # per payment attempt (initial or retry)
    buyer_notification_cost: float = 0.02  # per buyer-action request
    manual_review_cost: float = 0.25  # per transaction escalated for review

    def validate(self) -> list[str]:
        return [f"{n} must be >= 0" for n, v in (
            ("attempt_cost", self.attempt_cost),
            ("buyer_notification_cost", self.buyer_notification_cost),
            ("manual_review_cost", self.manual_review_cost)) if v < 0]


@dataclass
class TxnEvent:
    time_minutes: float  # minutes since scenario start
    type: EventType
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"time_minutes": round(self.time_minutes, 4), "type": self.type.value, **self.detail}


@dataclass
class Transaction:
    transaction_id: str
    merchant_id: str
    amount: float
    currency: str
    timestamp: datetime
    payment_method_category: PaymentMethodCategory
    offset_minutes: float  # timestamp as minutes since scenario start (simulation clock)
    failure_category: Optional[FailureCategory] = None  # category of the INITIAL failure
    initial_status: PaymentStatus = PaymentStatus.APPROVED
    recovery_status: RecoveryStatus = RecoveryStatus.NOT_NEEDED
    retry_count: int = 0  # automatic retries only (excludes the post-buyer-action attempt)
    final_status: FinalStatus = FinalStatus.COMPLETED
    total_recovery_time: Optional[float] = None  # minutes, first failure -> resolution; None if never failed
    processing_cost: float = 0.0  # all operational cost incl. the initial attempt
    recovery_cost: float = 0.0  # cost incurred after the initial attempt
    attempt_count: int = 0
    unnecessary_retries: int = 0  # automatic retries on a non-AUTO_RETRY failure class
    terminal_retries: int = 0  # ... of which on TERMINAL-class failures
    buyer_class_retries: int = 0  # ... of which on BUYER_ACTION-class failures
    buyer_actions_requested: int = 0
    buyer_actions_completed: int = 0
    duplicate_capture_risk: bool = False
    resolved_at_minutes: Optional[float] = None

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError("amount must be > 0")
        if len(self.currency) != 3 or not self.currency.isalpha():
            raise ValueError("currency must be a 3-letter code")

    def to_dict(self) -> dict[str, Any]:
        return {
            "transaction_id": self.transaction_id, "merchant_id": self.merchant_id,
            "amount": self.amount, "currency": self.currency, "timestamp": self.timestamp.isoformat(),
            "payment_method_category": self.payment_method_category.value,
            "failure_category": self.failure_category.value if self.failure_category else None,
            "initial_status": self.initial_status.value, "recovery_status": self.recovery_status.value,
            "retry_count": self.retry_count, "final_status": self.final_status.value,
            "total_recovery_time": self.total_recovery_time,
            "processing_cost": round(self.processing_cost, 6), "attempt_count": self.attempt_count,
            "buyer_actions_requested": self.buyer_actions_requested,
            "buyer_actions_completed": self.buyer_actions_completed,
            "duplicate_capture_risk": self.duplicate_capture_risk,
        }
