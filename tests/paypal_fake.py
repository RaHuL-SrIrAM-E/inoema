"""MOCK PayPal -- an in-memory HTTP test double (httpx.MockTransport). IT IS NOT PAYPAL.

Used only by offline unit tests to exercise our client/executor logic deterministically. Its behaviour encodes OUR
ASSUMPTIONS about the API (e.g. idempotent create, 422 on capturing an unapproved order); nothing it returns is evidence
of how the real Sandbox behaves. Real evidence comes only from tests/integration/test_paypal_sandbox_live.py and the
persisted sandbox_api_calls of genuine runs.
"""
from __future__ import annotations

import json
from typing import Any, Optional

import httpx

FAKE_CLIENT_ID = "FAKE-CLIENT-ID-abc123"
FAKE_CLIENT_SECRET = "FAKE-CLIENT-SECRET-zzz999"
FAKE_TOKEN = "FAKE-ACCESS-TOKEN-tok777"


def err(status: int, name: str, issue: Optional[str] = None, message: str = "mock error") -> httpx.Response:
    body: dict[str, Any] = {"name": name, "message": message, "debug_id": "mock-debug-id"}
    if issue:
        body["details"] = [{"issue": issue, "description": message}]
    return httpx.Response(status, json=body, headers={"paypal-debug-id": "mock-debug-id"})


class FakePayPal:
    def __init__(self) -> None:
        self.orders: dict[str, dict] = {}
        self.by_request_id: dict[str, str] = {}
        self.calls: list[tuple[str, str, dict]] = []  # (method, path, headers)
        self.capture_script: list[Any] = []  # items: "ok" | ("error", status, name, issue) | "timeout" | "ok_then_timeout" | ("bad_amount",)
        self.create_script: list[Any] = []
        self.token_fetches = 0
        self.token_responses: list[Any] = []  # optional scripted token endpoint behaviours
        self.capture_request_ids: list[Optional[str]] = []
        self._n = 0
        self.bad_amount = False
        self.supported_mock_codes: set[str] = set()  # which PayPal-Mock-Response codes THIS MOCK honours (an assumption for tests only)

    # -- helpers
    def approve(self, order_id: str) -> None:
        self.orders[order_id]["status"] = "APPROVED"

    def set_status(self, order_id: str, status: str) -> None:
        self.orders[order_id]["status"] = status

    def count(self, method: str, suffix: str) -> int:
        return sum(1 for m, p, _ in self.calls if m == method and p.endswith(suffix))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    # -- routing
    def _handle(self, req: httpx.Request) -> httpx.Response:
        path = req.url.path
        body = json.loads(req.content) if req.content and req.headers.get("content-type", "").startswith("application/json") else {}
        self.calls.append((req.method, path, dict(req.headers)))
        if path == "/v1/oauth2/token":
            self.token_fetches += 1
            if self.token_responses:
                r = self.token_responses.pop(0)
                if isinstance(r, httpx.Response):
                    return r
            return httpx.Response(200, json={"access_token": FAKE_TOKEN, "token_type": "Bearer", "expires_in": 3600, "scope": "x", "nonce": "n"},
                                  headers={"paypal-debug-id": "tok-debug"})
        if req.headers.get("authorization") != f"Bearer {FAKE_TOKEN}":
            return err(401, "AUTHENTICATION_FAILURE", None, "bad token")
        if req.method == "POST" and path == "/v2/checkout/orders":
            return self._create(req, body)
        if path.startswith("/v2/checkout/orders/"):
            oid = path.split("/")[4]
            if oid not in self.orders:
                return err(404, "RESOURCE_NOT_FOUND", "INVALID_RESOURCE_ID")
            if path.endswith("/capture"):
                return self._capture(req, oid)
            return httpx.Response(200, json=self.orders[oid], headers={"paypal-debug-id": "get-debug"})
        if path.startswith("/v2/payments/captures/"):
            cid = path.split("/")[4]
            for o in self.orders.values():
                for c in o["purchase_units"][0].get("payments", {}).get("captures", []):
                    if c["id"] == cid:
                        return httpx.Response(200, json=c, headers={"paypal-debug-id": "cap-debug"})
            return err(404, "RESOURCE_NOT_FOUND", "INVALID_RESOURCE_ID")
        return err(404, "RESOURCE_NOT_FOUND")

    def _create(self, req: httpx.Request, body: dict) -> httpx.Response:
        rid = req.headers.get("paypal-request-id")
        if self.create_script:
            step = self.create_script.pop(0)
            if step == "timeout":
                raise httpx.ConnectTimeout("mock timeout")
            if step == "ok_then_timeout":
                self._make_order(rid, body)
                raise httpx.ReadTimeout("mock lost response")
            if isinstance(step, tuple) and step[0] == "error":
                return err(step[1], step[2], step[3])
        if rid in self.by_request_id:  # our ASSUMPTION of idempotent create
            return httpx.Response(200, json=self.orders[self.by_request_id[rid]])
        return httpx.Response(201, json=self._make_order(rid, body), headers={"paypal-debug-id": "create-debug"})

    def _make_order(self, rid: Optional[str], body: dict) -> dict:
        self._n += 1
        oid = f"MOCKORDER{self._n:04d}"
        pu = body["purchase_units"][0]
        order = {"id": oid, "status": "CREATED", "intent": "CAPTURE",
                 "purchase_units": [{"reference_id": pu["reference_id"], "amount": pu["amount"],
                                    **({"invoice_id": pu["invoice_id"]} if "invoice_id" in pu else {})}],
                 "links": [{"rel": "approve", "href": f"https://www.sandbox.paypal.com/checkoutnow?token={oid}", "method": "GET"}]}
        self.orders[oid] = order
        if rid:
            self.by_request_id[rid] = oid
        return order

    def _complete(self, oid: str) -> None:
        o = self.orders[oid]
        amt = dict(o["purchase_units"][0]["amount"])
        if self.bad_amount:
            amt["value"] = "1.00"
        o["status"] = "COMPLETED"
        o["purchase_units"][0]["payments"] = {"captures": [{"id": f"MOCKCAP{oid[-4:]}", "status": "COMPLETED", "amount": amt, "final_capture": True}]}

    def _capture(self, req: httpx.Request, oid: str) -> httpx.Response:
        self.capture_request_ids.append(req.headers.get("paypal-request-id"))
        o = self.orders[oid]
        mock = req.headers.get("paypal-mock-response")
        if mock:
            code = json.loads(mock)["mock_application_codes"]
            if code in self.supported_mock_codes:
                return err(422, "UNPROCESSABLE_ENTITY", code)
        step = self.capture_script.pop(0) if self.capture_script else "ok"
        if step == "timeout":
            raise httpx.ConnectTimeout("mock timeout")
        if isinstance(step, tuple) and step[0] == "error":
            return err(step[1], step[2], step[3])
        if step == "race":  # another actor captures first; our request then fails with ORDER_ALREADY_CAPTURED
            self._complete(oid)
            return err(422, "UNPROCESSABLE_ENTITY", "ORDER_ALREADY_CAPTURED")
        if o["status"] == "COMPLETED":
            return err(422, "UNPROCESSABLE_ENTITY", "ORDER_ALREADY_CAPTURED")
        if o["status"] != "APPROVED":
            return err(422, "UNPROCESSABLE_ENTITY", "ORDER_NOT_APPROVED")
        self._complete(oid)
        if step == "ok_then_timeout":
            raise httpx.ReadTimeout("mock lost response")
        return httpx.Response(201, json=o, headers={"paypal-debug-id": "capture-debug"})
