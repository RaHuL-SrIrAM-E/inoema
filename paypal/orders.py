"""PayPal Orders v2 operations (create / get / capture) with idempotency headers.

Every state-changing call carries a caller-supplied PayPal-Request-Id so that a re-send of the SAME logical
operation is safe (docs: retries with the same id are safe for POST calls; capture-order is not named explicitly).
`mock_code` adds the documented negative-testing header `PayPal-Mock-Response: {"mock_application_codes": "<CODE>"}`
(Sandbox only; requires Negative Testing ON for the business sandbox account).
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Optional

from paypal.client import PayPalClient
from paypal.models import (Order, OrderState, validate_amount, validate_mock_code, validate_order_id,
                           validate_reference_id)


def _mock_header(mock_code: Optional[str]) -> dict:
    return {"PayPal-Mock-Response": json.dumps({"mock_application_codes": validate_mock_code(mock_code)})} if mock_code else {}


def parse_order(body: dict) -> Order:
    pu = (body.get("purchase_units") or [{}])[0]
    amt = pu.get("amount") or {}
    value = amt.get("value")
    approval = None
    for link in body.get("links", []):
        if link.get("rel") in ("approve", "payer-action"):  # which rel PayPal uses has not been verified live
            approval = link.get("href")
            if link.get("rel") == "payer-action":
                break
    caps = tuple(c["id"] for c in (pu.get("payments") or {}).get("captures", []) if "id" in c)
    raw_status = str(body.get("status", ""))
    return Order(id=str(body.get("id", "")), state=OrderState.parse(raw_status), raw_status=raw_status,
                 amount=Decimal(str(value)) if value is not None else None, currency=amt.get("currency_code"),
                 reference_id=pu.get("reference_id"), approval_url=approval, capture_ids=caps, raw=body)


class OrdersApi:
    def __init__(self, client: PayPalClient):
        self.client = client

    def create_order(self, amount: Any, currency: str, reference_id: str, request_id: str, *, invoice_id: Optional[str] = None,
                     meta: Optional[dict] = None, mock_code: Optional[str] = None) -> Order:
        value, cur = validate_amount(amount, currency)
        unit: dict[str, Any] = {"reference_id": validate_reference_id(reference_id), "amount": {"currency_code": cur, "value": f"{value:.2f}"}}
        if invoice_id:
            unit["invoice_id"] = validate_reference_id(invoice_id)
        payload = {"intent": "CAPTURE", "purchase_units": [unit],
                   "payment_source": {"paypal": {"experience_context": {
                       "return_url": self.client.config.return_url, "cancel_url": self.client.config.cancel_url,
                       "user_action": "PAY_NOW", "shipping_preference": "NO_SHIPPING"}}}}
        body = self.client.request("POST", "/v2/checkout/orders", operation="CREATE_ORDER", json_body=payload, meta=meta,
                                   headers={"PayPal-Request-Id": request_id, "Prefer": "return=representation", **_mock_header(mock_code)})
        return parse_order(body)

    def get_order(self, order_id: str, *, meta: Optional[dict] = None, mock_code: Optional[str] = None) -> Order:
        body = self.client.request("GET", f"/v2/checkout/orders/{validate_order_id(order_id)}", operation="GET_ORDER", meta=meta,
                                   headers=_mock_header(mock_code))
        return parse_order(body)

    def capture_order(self, order_id: str, request_id: str, *, meta: Optional[dict] = None, mock_code: Optional[str] = None) -> Order:
        body = self.client.request("POST", f"/v2/checkout/orders/{validate_order_id(order_id)}/capture", operation="CAPTURE_ORDER",
                                   json_body={}, meta=meta,
                                   headers={"PayPal-Request-Id": request_id, "Prefer": "return=representation", **_mock_header(mock_code)})
        return parse_order(body)
