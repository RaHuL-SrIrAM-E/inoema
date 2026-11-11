"""Types, configuration and Sandbox-only guards for the PayPal integration.

Everything in this package talks to the PayPal SANDBOX only. Evidence and validation records are tagged
`PAYPAL_SANDBOX` and are never mixed with simulated data (`SIMULATED`).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping, Optional
from urllib.parse import urlparse

ENVIRONMENT_LABEL = "PAYPAL_SANDBOX"
SANDBOX_BASE_URL = "https://api-m.sandbox.paypal.com"
SANDBOX_HOSTS = frozenset({"api-m.sandbox.paypal.com", "api.sandbox.paypal.com"})
ALLOWED_CURRENCIES = frozenset({"USD", "EUR", "GBP", "CAD", "AUD"})  # two-decimal currencies only
MAX_SANDBOX_AMOUNT = Decimal("10000.00")
_ID_RE = re.compile(r"^[A-Za-z0-9_-]{5,64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9_.-]{1,127}$")
_MOCK_RE = re.compile(r"^[A-Z][A-Z0-9_]{2,63}$")


class SandboxViolation(RuntimeError):
    """Raised whenever something would leave the PayPal Sandbox (non-sandbox host/mode/env). Never retried."""


class PayPalConfigError(ValueError):
    """Missing or invalid configuration."""


class PayPalApiError(Exception):
    """A failed PayPal API exchange (HTTP error or transport failure). Carries evidence, never secrets."""

    def __init__(self, status: Optional[int], name: str, message: str, *, issues: tuple[str, ...] = (),
                 debug_id: Optional[str] = None, operation: str = "", transport: bool = False,
                 body: Optional[dict] = None):
        super().__init__(f"[{status}] {name}: {message} issues={list(issues)} (paypal-debug-id={debug_id})")
        self.status, self.name, self.message, self.issues = status, name, message, issues
        self.debug_id, self.operation, self.transport, self.body = debug_id, operation, transport, body or {}


@dataclass(frozen=True)
class PayPalConfig:
    client_id: str = field(default="", repr=False)
    client_secret: str = field(default="", repr=False)
    base_url: str = SANDBOX_BASE_URL
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 30.0
    return_url: str = "https://example.com/return"
    cancel_url: str = "https://example.com/cancel"

    def secrets(self) -> list[str]:
        return [s for s in (self.client_id, self.client_secret) if s]

    def validate(self) -> None:
        enforce_sandbox_url(self.base_url)
        if not self.client_id or not self.client_secret:
            raise PayPalConfigError("PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET are required (see .env.example)")

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, **overrides: Any) -> "PayPalConfig":
        env = os.environ if env is None else env
        for var in ("PAYPAL_ENV", "PAYPAL_MODE", "PAYPAL_ENVIRONMENT"):
            if env.get(var, "sandbox").strip().lower() not in ("sandbox", ""):
                raise SandboxViolation(f"{var}={env[var]!r}: this integration is SANDBOX-ONLY; live mode is refused")
        cfg = cls(client_id=env.get("PAYPAL_CLIENT_ID", ""), client_secret=env.get("PAYPAL_CLIENT_SECRET", ""),
                  base_url=env.get("PAYPAL_BASE_URL", SANDBOX_BASE_URL), **overrides)
        cfg.validate()
        return cfg


def enforce_sandbox_url(url: str) -> str:
    """Only https://<sandbox host> (default port, no credentials, no path trickery) is accepted."""
    p = urlparse(url)
    if (p.scheme != "https" or p.hostname not in SANDBOX_HOSTS or p.port not in (None, 443)
            or p.username or p.password or p.path not in ("", "/") or p.query or p.fragment):
        raise SandboxViolation(f"refusing non-sandbox PayPal URL {url!r}; allowed hosts: {sorted(SANDBOX_HOSTS)}")
    return url.rstrip("/")


def validate_order_id(order_id: str) -> str:
    if not isinstance(order_id, str) or not _ID_RE.match(order_id):
        raise ValueError(f"invalid PayPal id {order_id!r}")
    return order_id


def validate_reference_id(ref: str) -> str:
    if not isinstance(ref, str) or not _REF_RE.match(ref):
        raise ValueError(f"invalid reference_id {ref!r} (allowed: letters, digits, '_', '.', '-', max 127)")
    return ref


def validate_mock_code(code: str) -> str:
    if not _MOCK_RE.match(code):
        raise ValueError(f"invalid mock_application_codes value {code!r}")
    return code


def validate_amount(value: Any, currency: str) -> tuple[Decimal, str]:
    """Positive, finite, two-decimal amount within the sandbox cap, in an allowed currency."""
    cur = (currency or "").upper()
    if cur not in ALLOWED_CURRENCIES:
        raise ValueError(f"currency {currency!r} not allowed; allowed: {sorted(ALLOWED_CURRENCIES)}")
    try:
        d = Decimal(str(value))
    except InvalidOperation as e:
        raise ValueError(f"invalid amount {value!r}") from e
    if not d.is_finite() or d <= 0:
        raise ValueError("amount must be a positive finite number")
    q = d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if q != d:
        raise ValueError("amount must have at most 2 decimal places")
    if q > MAX_SANDBOX_AMOUNT:
        raise ValueError(f"amount exceeds the sandbox cap of {MAX_SANDBOX_AMOUNT}")
    return q, cur


class OrderState(str, Enum):
    """Order `status` values as returned by PayPal. The lifecycle semantics are NOT verified against the docs; UNKNOWN is treated conservatively."""
    CREATED = "CREATED"
    SAVED = "SAVED"
    APPROVED = "APPROVED"
    VOIDED = "VOIDED"
    COMPLETED = "COMPLETED"
    PAYER_ACTION_REQUIRED = "PAYER_ACTION_REQUIRED"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def parse(cls, v: Any) -> "OrderState":
        try:
            return cls(v)
        except ValueError:
            return cls.UNKNOWN


@dataclass(frozen=True)
class Order:
    id: str
    state: OrderState
    raw_status: str
    amount: Optional[Decimal]
    currency: Optional[str]
    reference_id: Optional[str]
    approval_url: Optional[str]
    capture_ids: tuple[str, ...]
    raw: dict = field(repr=False, default_factory=dict)


@dataclass(frozen=True)
class VerificationResult:
    ok: bool
    problems: tuple[str, ...]
    capture_id: Optional[str]
    details: Mapping[str, Any] = field(default_factory=dict)
