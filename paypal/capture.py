"""Capture lookup and verification."""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Optional

from paypal.client import PayPalClient
from paypal.models import Order, OrderState, VerificationResult, validate_amount, validate_order_id


class CapturesApi:
    def __init__(self, client: PayPalClient):
        self.client = client

    def get_capture(self, capture_id: str, *, meta: Optional[dict] = None) -> dict:
        return self.client.request("GET", f"/v2/payments/captures/{validate_order_id(capture_id)}", operation="GET_CAPTURE", meta=meta)


def verify_capture(order: Order, capture: dict, *, expected_amount: Any, expected_currency: str,
                   expected_reference_id: Optional[str] = None) -> VerificationResult:
    """Independent verification from the order and the separately-fetched capture. All checks must pass:
    order COMPLETED, exactly one capture, capture id consistent, capture COMPLETED, amount/currency equal the request."""
    problems: list[str] = []
    amt, cur = validate_amount(expected_amount, expected_currency)
    if order.state != OrderState.COMPLETED:
        problems.append(f"order status {order.raw_status!r} != COMPLETED")
    if len(order.capture_ids) != 1:
        problems.append(f"expected exactly 1 capture on the order, found {len(order.capture_ids)}")
    cap_id = capture.get("id")
    if order.capture_ids and cap_id not in order.capture_ids:
        problems.append("capture id from GET capture is not among the order's captures")
    if capture.get("status") != "COMPLETED":
        problems.append(f"capture status {capture.get('status')!r} != COMPLETED")
    c_amt = capture.get("amount") or {}
    try:
        same = Decimal(str(c_amt.get("value"))) == amt and c_amt.get("currency_code") == cur
    except Exception:  # noqa: BLE001
        same = False
    if not same:
        problems.append(f"capture amount {c_amt} != requested {amt} {cur}")
    if order.amount is not None and (order.amount != amt or order.currency != cur):
        problems.append(f"order amount {order.amount} {order.currency} != requested {amt} {cur}")
    if expected_reference_id and order.reference_id != expected_reference_id:
        problems.append("order reference_id mismatch")
    return VerificationResult(not problems, tuple(problems), cap_id,
                              {"order_status": order.raw_status, "capture_status": capture.get("status"), "amount": c_amt,
                               "final_capture": capture.get("final_capture")})
