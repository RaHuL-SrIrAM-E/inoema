"""Spike 1: minimal genuine PayPal Sandbox client (Orders v2 + capture lookup).

Every call goes to the real PayPal Sandbox over HTTPS. Nothing is faked.
Raw request/response metadata (secrets redacted) is persisted to SQLite.

Usage (from repo root):
    python -m spikes.paypal.client run          # full flow; needs a human buyer approval
    python -m spikes.paypal.client create
    python -m spikes.paypal.client status ORDER_ID
    python -m spikes.paypal.client capture ORDER_ID
Env: PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET (see .env.example).
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

import httpx

from spikes.envutil import ROOT, load_env

SANDBOX_BASE = "https://api-m.sandbox.paypal.com"
DEFAULT_DB = os.path.join(ROOT, "data", "spike_paypal.db")
TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=10.0)
REDACT_KEYS = {"access_token", "client_secret", "authorization", "id_token", "refresh_token"}


class PayPalError(Exception):
    def __init__(self, status: int | None, name: str, message: str, debug_id: str | None = None):
        super().__init__(f"[{status}] {name}: {message} (paypal-debug-id={debug_id})")
        self.status, self.name, self.message, self.debug_id = status, name, message, debug_id


def redact(obj):
    """Recursively replace secret-bearing values."""
    if isinstance(obj, dict):
        return {k: ("***REDACTED***" if k.lower() in REDACT_KEYS else redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    return obj


class ApiLog:
    """Append-only SQLite log of every API exchange (secrets excluded)."""

    def __init__(self, db_path: str = DEFAULT_DB):
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS paypal_api_calls (
                id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, run_id TEXT NOT NULL,
                method TEXT NOT NULL, url TEXT NOT NULL, request_headers TEXT, request_body TEXT,
                status INTEGER, response_headers TEXT, response_body TEXT, latency_ms REAL,
                paypal_debug_id TEXT, error TEXT, environment TEXT NOT NULL DEFAULT 'PAYPAL_SANDBOX')"""
        )
        self.conn.commit()

    def record(self, **kw):
        self.conn.execute(
            "INSERT INTO paypal_api_calls (ts,run_id,method,url,request_headers,request_body,status,"
            "response_headers,response_body,latency_ms,paypal_debug_id,error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (datetime.now(timezone.utc).isoformat(), kw["run_id"], kw["method"], kw["url"],
             json.dumps(redact(kw.get("request_headers"))), json.dumps(redact(kw.get("request_body"))),
             kw.get("status"), json.dumps(redact(kw.get("response_headers"))),
             json.dumps(redact(kw.get("response_body"))), kw.get("latency_ms"),
             kw.get("debug_id"), kw.get("error")),
        )
        self.conn.commit()


class PayPalSandbox:
    def __init__(self, client_id: str, client_secret: str, base_url: str = SANDBOX_BASE,
                 log: ApiLog | None = None, transport: httpx.BaseTransport | None = None):
        host = urlparse(base_url).hostname or ""
        if "sandbox" not in host:  # hard guard: never talk to live PayPal from the spike
            raise ValueError(f"Refusing non-sandbox base URL: {base_url}")
        if not client_id or not client_secret:
            raise ValueError("PAYPAL_CLIENT_ID and PAYPAL_CLIENT_SECRET are required")
        self._id, self._secret, self.base = client_id, client_secret, base_url.rstrip("/")
        self.log = log or ApiLog()
        self.run_id = str(uuid.uuid4())
        self._http = httpx.Client(timeout=TIMEOUT, transport=transport)
        self._token: str | None = None
        self._token_expiry = 0.0

    @classmethod
    def from_env(cls, **kw) -> "PayPalSandbox":
        load_env()
        return cls(os.environ.get("PAYPAL_CLIENT_ID", ""), os.environ.get("PAYPAL_CLIENT_SECRET", ""),
                   os.environ.get("PAYPAL_BASE_URL", SANDBOX_BASE), **kw)

    # -- low-level -----------------------------------------------------------
    def _send(self, method: str, path: str, *, headers=None, json_body=None, data=None, auth=None) -> dict:
        url = f"{self.base}{path}"
        hdrs = dict(headers or {})
        shown = {**hdrs, **({"Authorization": "Basic ***"} if auth else {})}
        t0 = time.monotonic()
        try:
            r = self._http.request(method, url, headers=hdrs, json=json_body, data=data, auth=auth)
        except httpx.HTTPError as e:
            self.log.record(run_id=self.run_id, method=method, url=url, request_headers=shown,
                            request_body=json_body or data, error=f"{type(e).__name__}: {e}",
                            latency_ms=(time.monotonic() - t0) * 1000)
            raise PayPalError(None, "NETWORK_ERROR", f"{type(e).__name__}: {e}") from e
        latency = (time.monotonic() - t0) * 1000
        debug_id = r.headers.get("paypal-debug-id")
        try:
            body = r.json() if r.content else {}
        except ValueError:
            body = {"_non_json_body": r.text[:500]}
        self.log.record(run_id=self.run_id, method=method, url=url, request_headers=shown,
                        request_body=json_body or data, status=r.status_code,
                        response_headers={k: v for k, v in r.headers.items()
                                          if k.lower() in ("paypal-debug-id", "content-type", "date")},
                        response_body=body, latency_ms=latency, debug_id=debug_id)
        if r.status_code >= 400:
            name = body.get("name") or body.get("error") or "HTTP_ERROR"
            msg = body.get("message") or body.get("error_description") or r.text[:200]
            raise PayPalError(r.status_code, name, msg, debug_id)
        return body

    def _auth_headers(self, extra=None) -> dict:
        if not self._token or time.time() >= self._token_expiry - 30:
            self.get_access_token()
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json", **(extra or {})}

    # -- operations ----------------------------------------------------------
    def get_access_token(self) -> dict:
        body = self._send("POST", "/v1/oauth2/token", data={"grant_type": "client_credentials"},
                          headers={"Accept": "application/json"}, auth=(self._id, self._secret))
        self._token = body["access_token"]
        self._token_expiry = time.time() + int(body.get("expires_in", 300))
        return redact(body)

    def create_order(self, value: str = "10.00", currency: str = "USD",
                     return_url: str = "https://example.com/return",
                     cancel_url: str = "https://example.com/cancel") -> dict:
        payload = {
            "intent": "CAPTURE",
            "purchase_units": [{"reference_id": "inoema-spike-1",
                                "amount": {"currency_code": currency, "value": value}}],
            "payment_source": {"paypal": {"experience_context": {
                "return_url": return_url, "cancel_url": cancel_url,
                "user_action": "PAY_NOW", "shipping_preference": "NO_SHIPPING"}}},
        }
        return self._send("POST", "/v2/checkout/orders",
                          headers=self._auth_headers({"PayPal-Request-Id": str(uuid.uuid4()),
                                                      "Prefer": "return=representation"}),
                          json_body=payload)

    def get_order(self, order_id: str) -> dict:
        return self._send("GET", f"/v2/checkout/orders/{order_id}", headers=self._auth_headers())

    def capture_order(self, order_id: str) -> dict:
        return self._send("POST", f"/v2/checkout/orders/{order_id}/capture",
                          headers=self._auth_headers({"PayPal-Request-Id": f"capture-{order_id}",
                                                      "Prefer": "return=representation"}),
                          json_body={})

    def get_capture(self, capture_id: str) -> dict:
        return self._send("GET", f"/v2/payments/captures/{capture_id}", headers=self._auth_headers())


def approval_link(order: dict) -> str | None:
    for link in order.get("links", []):
        if link.get("rel") in ("payer-action", "approve"):
            return link["href"]
    return None


def verify_capture(order_id: str, captured: dict, capture_detail: dict, expected_value: str,
                   expected_currency: str) -> list[str]:
    """Return a list of verification failures (empty == verified)."""
    problems = []
    if captured.get("id") != order_id:
        problems.append("capture response order id mismatch")
    if captured.get("status") != "COMPLETED":
        problems.append(f"order status {captured.get('status')} != COMPLETED")
    amt = capture_detail.get("amount", {})
    if amt.get("value") != expected_value or amt.get("currency_code") != expected_currency:
        problems.append(f"amount mismatch: {amt}")
    if capture_detail.get("status") != "COMPLETED":
        problems.append(f"capture status {capture_detail.get('status')} != COMPLETED")
    try:
        cap_id = captured["purchase_units"][0]["payments"]["captures"][0]["id"]
        if cap_id != capture_detail.get("id"):
            problems.append("capture id differs between capture response and GET capture")
    except (KeyError, IndexError):
        problems.append("capture id not found in capture response")
    return problems


def wait_for_approval(pp: PayPalSandbox, order_id: str, timeout_s: int = 300, poll_s: int = 5) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        order = pp.get_order(order_id)
        if order["status"] == "APPROVED":
            return order
        if order["status"] in ("COMPLETED", "VOIDED"):
            raise PayPalError(None, "UNEXPECTED_STATUS", f"order is {order['status']}")
        time.sleep(poll_s)
    raise PayPalError(None, "APPROVAL_TIMEOUT", f"buyer did not approve within {timeout_s}s")


def run_full_flow(pp: PayPalSandbox, value="10.00", currency="USD", approval_timeout_s=300) -> int:
    print("1. token:", pp.get_access_token())
    order = pp.create_order(value, currency)
    oid = order["id"]
    print(f"2. created order {oid} status={order['status']}")
    print(f"3. get order status={pp.get_order(oid)['status']}")
    link = approval_link(order)
    print("\n>>> BUYER ACTION REQUIRED (manual, cannot be automated):")
    print(f">>> Open in a browser and log in with a SANDBOX personal account, then approve:\n>>> {link}\n")
    wait_for_approval(pp, oid, approval_timeout_s)
    print("4. buyer approved; order status=APPROVED")
    captured = pp.capture_order(oid)
    print(f"5. captured: order status={captured['status']}")
    cap_id = captured["purchase_units"][0]["payments"]["captures"][0]["id"]
    detail = pp.get_capture(cap_id)
    print(f"6. capture {cap_id}: status={detail['status']} amount={detail['amount']}")
    problems = verify_capture(oid, captured, detail, value, currency)
    print("VERIFIED" if not problems else f"VERIFICATION FAILED: {problems}")
    print(f"API log: {DEFAULT_DB} (run_id={pp.run_id})")
    return 0 if not problems else 1


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "run"
    try:
        pp = PayPalSandbox.from_env()
        if cmd == "run":
            return run_full_flow(pp)
        if cmd == "create":
            o = pp.create_order()
            print(json.dumps({"id": o["id"], "status": o["status"], "approve": approval_link(o)}, indent=2))
        elif cmd == "status":
            print(json.dumps(pp.get_order(argv[2]), indent=2))
        elif cmd == "capture":
            print(json.dumps(pp.capture_order(argv[2]), indent=2))
        else:
            print(__doc__)
            return 2
        return 0
    except (PayPalError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
