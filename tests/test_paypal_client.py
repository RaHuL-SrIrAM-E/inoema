"""OFFLINE unit tests for the PayPal Sandbox integration (config, auth, client, orders, capture verification, classifier,
persistence). They use tests/paypal_fake.py -- a MOCK test double, NOT PayPal. They validate our logic only and are NOT
evidence about real Sandbox behaviour (see tests/integration/test_paypal_sandbox_live.py for genuine calls)."""
import json
import sqlite3
from decimal import Decimal

import httpx
import pytest

from engine.payments.models import FailureCategory
from paypal.authentication import TokenProvider
from paypal.capture import CapturesApi, verify_capture
from paypal.client import PayPalClient, redact
from paypal.error_classifier import RecoveryClass, classification_table, classify
from paypal.models import (OrderState, PayPalApiError, PayPalConfig, PayPalConfigError, SandboxViolation, validate_amount,
                           validate_mock_code, validate_order_id, validate_reference_id)
from paypal.orders import OrdersApi, parse_order
from paypal.persistence import SandboxStore
from tests.paypal_fake import FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_TOKEN, FakePayPal, err

CFG = PayPalConfig(client_id=FAKE_CLIENT_ID, client_secret=FAKE_CLIENT_SECRET)


def make(fake=None, store=None, clock=None):
    fake = fake or FakePayPal()
    kw = {"clock": clock} if clock else {}
    return fake, PayPalClient(CFG, sink=store, transport=fake.transport(), **kw)


# ============================================================ Sandbox-only enforcement
@pytest.mark.parametrize("url", ["https://api-m.paypal.com", "https://api.paypal.com", "http://api-m.sandbox.paypal.com",
                                 "https://api-m.sandbox.paypal.com:8443", "https://evil.example.com", "https://user:pw@api-m.sandbox.paypal.com",
                                 "https://api-m.sandbox.paypal.com.evil.com", "https://api-m.sandbox.paypal.com/v2", "https://sandbox.paypal.com"])
def test_non_sandbox_urls_are_refused(url):
    with pytest.raises(SandboxViolation):
        PayPalConfig(client_id="a", client_secret="b", base_url=url).validate()
    with pytest.raises(SandboxViolation):
        PayPalClient(PayPalConfig(client_id="a", client_secret="b", base_url=url))


def test_sandbox_urls_accepted():
    for url in ("https://api-m.sandbox.paypal.com", "https://api-m.sandbox.paypal.com/", "https://api.sandbox.paypal.com"):
        PayPalConfig(client_id="a", client_secret="b", base_url=url).validate()


@pytest.mark.parametrize("var,val", [("PAYPAL_ENV", "live"), ("PAYPAL_MODE", "production"), ("PAYPAL_ENVIRONMENT", "LIVE")])
def test_live_mode_env_is_refused(var, val):
    with pytest.raises(SandboxViolation):
        PayPalConfig.from_env({"PAYPAL_CLIENT_ID": "a", "PAYPAL_CLIENT_SECRET": "b", var: val})
    assert PayPalConfig.from_env({"PAYPAL_CLIENT_ID": "a", "PAYPAL_CLIENT_SECRET": "b", var: "sandbox"}).base_url.endswith("sandbox.paypal.com")


def test_missing_credentials_and_secret_hiding():
    with pytest.raises(PayPalConfigError):
        PayPalConfig.from_env({})
    assert FAKE_CLIENT_SECRET not in repr(CFG) and FAKE_CLIENT_ID not in repr(CFG)


@pytest.mark.parametrize("path", ["https://api-m.paypal.com/v2/x", "//evil.com/v2/x", "/v2/../../x", "v2/checkout/orders", "/health"])
def test_unsafe_request_paths_are_refused(path):
    _, c = make()
    with pytest.raises(SandboxViolation):
        c.request("GET", path, operation="X")


def test_redirects_are_not_followed():
    def handler(req):
        return httpx.Response(302, headers={"location": "https://api-m.paypal.com/evil"})
    c = PayPalClient(CFG, transport=httpx.MockTransport(handler))
    with pytest.raises(PayPalApiError):
        c._send("GET", "/v1/oauth2/token", operation="X")
    assert c._http.follow_redirects is False


# ============================================================ OAuth
def test_oauth_uses_basic_auth_and_caches_token():
    fake, c = make()
    c.request("GET", "/v2/checkout/orders/ABCDE12345", operation="X") if False else None
    OrdersApi(c).create_order("10.00", "USD", "ref-1", "rid-1")
    OrdersApi(c).create_order("11.00", "USD", "ref-2", "rid-2")
    assert fake.token_fetches == 1  # cached across calls
    tok_call = next(h for m, p, h in fake.calls if p == "/v1/oauth2/token")
    assert tok_call["authorization"].startswith("Basic ")


def test_token_refreshes_after_expiry_using_a_fake_clock():
    now = [1000.0]
    fake, c = make(clock=lambda: now[0])
    OrdersApi(c).create_order("10.00", "USD", "ref-1", "rid-1")
    now[0] += 3600 - 30  # inside the 60s skew window
    OrdersApi(c).create_order("10.00", "USD", "ref-2", "rid-2")
    assert fake.token_fetches == 2


def test_401_triggers_exactly_one_reauthentication():
    fake, c = make()
    c.tokens.get_token()
    c.tokens._token = "STALE"  # simulate a token PayPal rejects
    o = OrdersApi(c).create_order("10.00", "USD", "ref-1", "rid-1")
    assert o.state == OrderState.CREATED and fake.token_fetches == 2
    # a persistent 401 is not retried forever
    fake2, c2 = make()
    fake2.token_responses = [None]
    c2.tokens._send = lambda *a, **k: {"access_token": "ALWAYS-BAD", "expires_in": 3600}
    with pytest.raises(PayPalApiError) as e:
        OrdersApi(c2).create_order("10.00", "USD", "r", "rid")
    assert e.value.status == 401 and sum(1 for m, p, _ in fake2.calls if p == "/v2/checkout/orders") == 2


def test_bad_credentials_surface_as_api_error_with_debug_id():
    def handler(req):
        return httpx.Response(401, json={"error": "invalid_client", "error_description": "Client Authentication failed"}, headers={"paypal-debug-id": "dbg1"})
    c = PayPalClient(CFG, transport=httpx.MockTransport(handler))
    with pytest.raises(PayPalApiError) as e:
        c.tokens.get_token()
    assert e.value.status == 401 and e.value.name == "invalid_client" and e.value.debug_id == "dbg1"
    assert classify(e.value).category is None  # configuration problem, not a payment failure


def test_malformed_token_response():
    c = PayPalClient(CFG, transport=httpx.MockTransport(lambda r: httpx.Response(200, json={"nope": 1})))
    with pytest.raises(PayPalApiError) as e:
        c.tokens.get_token()
    assert e.value.name == "MALFORMED_TOKEN_RESPONSE"


# ============================================================ order creation / retrieval / approval state
def test_create_order_payload_and_idempotency_headers():
    fake, c = make()
    o = OrdersApi(c).create_order("10.5", "usd", "ref-1", "req-123", invoice_id="inv-1")
    method, path, headers = next(x for x in fake.calls if x[1] == "/v2/checkout/orders")
    assert headers["paypal-request-id"] == "req-123" and headers["prefer"] == "return=representation"
    assert o.amount == Decimal("10.50") and o.currency == "USD" and o.reference_id == "ref-1"
    assert o.state == OrderState.CREATED and o.approval_url.endswith(o.id)
    sent = fake.orders[o.id]["purchase_units"][0]["amount"]
    assert sent == {"currency_code": "USD", "value": "10.50"}


@pytest.mark.parametrize("amount,cur", [(0, "USD"), (-1, "USD"), ("abc", "USD"), ("10.001", "USD"), ("10001", "USD"), (float("nan"), "USD"),
                                         (float("inf"), "USD"), ("10", "JPY"), ("10", ""), (None, "USD")])
def test_invalid_amounts_and_currencies_rejected_before_any_http(amount, cur):
    fake, c = make()
    with pytest.raises(ValueError):
        OrdersApi(c).create_order(amount, cur, "ref-1", "rid")
    assert fake.calls == []


def test_amount_validation_values():
    assert validate_amount("10", "usd") == (Decimal("10.00"), "USD")
    assert validate_amount(Decimal("9999.99"), "EUR")[0] == Decimal("9999.99")
    with pytest.raises(ValueError):
        validate_reference_id("bad ref!")
    with pytest.raises(ValueError):
        validate_order_id("../../etc")
    with pytest.raises(ValueError):
        validate_mock_code("lower_case")
    assert validate_mock_code("INSTRUMENT_DECLINED") == "INSTRUMENT_DECLINED"


def test_get_order_parses_status_approval_url_and_captures():
    for rel in ("approve", "payer-action"):
        o = parse_order({"id": "O1", "status": "CREATED", "links": [{"rel": "self", "href": "x"}, {"rel": rel, "href": "https://u"}]})
        assert o.approval_url == "https://u"
    o = parse_order({"id": "O1", "status": "COMPLETED", "purchase_units": [{"reference_id": "r", "amount": {"value": "1.00", "currency_code": "USD"},
                                                                         "payments": {"captures": [{"id": "C1"}, {"id": "C2"}]}}]})
    assert o.capture_ids == ("C1", "C2") and o.state == OrderState.COMPLETED
    assert parse_order({"id": "O", "status": "SOMETHING_NEW"}).state == OrderState.UNKNOWN  # unverified statuses are not assumed
    fake, c = make()
    created = OrdersApi(c).create_order("10.00", "USD", "ref-1", "rid")
    assert OrdersApi(c).get_order(created.id).state == OrderState.CREATED
    fake.approve(created.id)
    assert OrdersApi(c).get_order(created.id).state == OrderState.APPROVED
    with pytest.raises(ValueError):
        OrdersApi(c).get_order("../x")


def test_capture_sends_request_id_and_mock_header():
    fake, c = make()
    o = OrdersApi(c).create_order("10.00", "USD", "ref-1", "rid")
    fake.approve(o.id)
    done = OrdersApi(c).capture_order(o.id, "cap-req-1", mock_code=None)
    hdr = next(h for m, p, h in fake.calls if p.endswith("/capture"))
    assert hdr["paypal-request-id"] == "cap-req-1" and "paypal-mock-response" not in hdr
    assert done.state == OrderState.COMPLETED and len(done.capture_ids) == 1
    fake2, c2 = make()
    o2 = OrdersApi(c2).create_order("10.00", "USD", "ref-1", "rid")
    with pytest.raises(PayPalApiError):
        OrdersApi(c2).capture_order(o2.id, "r", mock_code="INSTRUMENT_DECLINED")
    hdr2 = next(h for m, p, h in fake2.calls if p.endswith("/capture"))
    assert json.loads(hdr2["paypal-mock-response"]) == {"mock_application_codes": "INSTRUMENT_DECLINED"}


# ============================================================ capture verification
def _completed(fake, c, amount="10.00"):
    o = OrdersApi(c).create_order(amount, "USD", "ref-1", "rid")
    fake.approve(o.id)
    done = OrdersApi(c).capture_order(o.id, "cap")
    cap = CapturesApi(c).get_capture(done.capture_ids[0])
    return done, cap


def test_verify_capture_ok_and_each_failure_mode():
    fake, c = make()
    order, cap = _completed(fake, c)
    ok = verify_capture(order, cap, expected_amount="10.00", expected_currency="USD", expected_reference_id="ref-1")
    assert ok.ok and ok.problems == () and ok.capture_id == cap["id"]
    assert not verify_capture(order, cap, expected_amount="11.00", expected_currency="USD").ok
    assert not verify_capture(order, cap, expected_amount="10.00", expected_currency="EUR").ok
    assert not verify_capture(order, {**cap, "status": "PENDING"}, expected_amount="10.00", expected_currency="USD").ok
    assert not verify_capture(order, cap, expected_amount="10.00", expected_currency="USD", expected_reference_id="other").ok
    two = parse_order({**order.raw, "purchase_units": [{**order.raw["purchase_units"][0], "payments": {"captures": [{"id": "A"}, {"id": "B"}]}}]})
    assert any("exactly 1 capture" in p for p in verify_capture(two, cap, expected_amount="10.00", expected_currency="USD").problems)
    pending = parse_order({**order.raw, "status": "APPROVED"})
    assert any("!= COMPLETED" in p for p in verify_capture(pending, cap, expected_amount="10.00", expected_currency="USD").problems)
    assert any("not among" in p for p in verify_capture(order, {**cap, "id": "OTHER"}, expected_amount="10.00", expected_currency="USD").problems)


# ============================================================ evidence, redaction, errors
def test_evidence_never_contains_secrets_and_keeps_debug_ids(tmp_path):
    store = SandboxStore(str(tmp_path / "s.db"), secrets=[FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_TOKEN])
    fake, c = make(store=store)
    o = OrdersApi(c).create_order("10.00", "USD", "ref-1", "rid", meta={"execution_id": None})
    rows = store.api_calls()
    assert [r["operation"] for r in rows] == ["OAUTH_TOKEN", "CREATE_ORDER"]
    dump = json.dumps(rows)
    for secret in (FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_TOKEN):
        assert secret not in dump
    assert "***REDACTED***" in rows[0]["response_body"] and "***REDACTED***" in rows[1]["request_headers"]
    assert rows[1]["paypal_debug_id"] == "create-debug" and rows[1]["status"] == 201 and rows[1]["request_id"] == "rid"
    store.conn.commit()
    assert FAKE_CLIENT_SECRET.encode() not in open(tmp_path / "s.db", "rb").read() and FAKE_TOKEN.encode() not in open(tmp_path / "s.db", "rb").read()


def test_redact_helper():
    out = redact({"access_token": "t", "a": {"Authorization": "Bearer x", "ok": 1}, "l": ["has SECRET inside"]}, ("SECRET",))
    assert out == {"access_token": "***REDACTED***", "a": {"Authorization": "***REDACTED***", "ok": 1}, "l": ["has ***REDACTED*** inside"]}


def test_api_error_carries_issues_and_message_is_scrubbed():
    def handler(req):
        if req.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": FAKE_TOKEN, "expires_in": 3600})
        return err(422, "UNPROCESSABLE_ENTITY", "INSTRUMENT_DECLINED", message=f"declined for {FAKE_TOKEN}")
    c = PayPalClient(CFG, transport=httpx.MockTransport(handler))
    c.tokens.get_token()
    with pytest.raises(PayPalApiError) as e:
        c.request("POST", "/v2/checkout/orders/ABCDE12345/capture", operation="CAPTURE_ORDER", json_body={})
    assert e.value.status == 422 and e.value.issues == ("INSTRUMENT_DECLINED",) and e.value.debug_id == "mock-debug-id"
    assert FAKE_TOKEN not in str(e.value) and FAKE_TOKEN not in json.dumps(e.value.body)


def test_timeouts_and_connection_errors_are_transport_errors_without_leaking_details():
    def handler(req):
        raise httpx.ConnectTimeout(f"timeout talking to host with {FAKE_CLIENT_SECRET}")
    c = PayPalClient(CFG, transport=httpx.MockTransport(handler))
    with pytest.raises(PayPalApiError) as e:
        c._send("POST", "/v1/oauth2/token", operation="OAUTH_TOKEN", basic_auth=("a", "b"))
    assert e.value.transport and e.value.status is None and FAKE_CLIENT_SECRET not in str(e.value)
    assert classify(e.value).recovery_class == RecoveryClass.AUTO_RETRY


def test_non_json_error_body_is_handled():
    c = PayPalClient(CFG, transport=httpx.MockTransport(lambda r: httpx.Response(502, text="<html>bad gateway</html>")))
    with pytest.raises(PayPalApiError) as e:
        c._send("GET", "/v1/oauth2/token", operation="X")
    assert e.value.status == 502 and e.value.name == "HTTP_ERROR"


# ============================================================ error classification
def _e(status=422, name="UNPROCESSABLE_ENTITY", issues=(), transport=False):
    return PayPalApiError(status, name, "m", issues=tuple(issues), transport=transport)


@pytest.mark.parametrize("issue,cat,cls,retry", [
    ("INSTRUMENT_DECLINED", "INSTRUMENT_DECLINED", RecoveryClass.BUYER_ACTION, False),
    ("PAYER_ACTION_REQUIRED", "PAYER_ACTION_REQUIRED", RecoveryClass.BUYER_ACTION, False),
    ("ORDER_NOT_APPROVED", "ORDER_NOT_APPROVED", RecoveryClass.BUYER_ACTION, False),
    ("PAYMENT_SOURCE_DECLINED_BY_PROCESSOR", "INSTRUMENT_DECLINED", RecoveryClass.BUYER_ACTION, False),
    ("CARD_EXPIRED", "EXPIRED_PAYMENT_METHOD", RecoveryClass.BUYER_ACTION, False),
    ("TRANSACTION_REFUSED", "TERMINAL_PAYMENT_FAILURE", RecoveryClass.TERMINAL, False),
    ("PAYMENT_DENIED", "TERMINAL_PAYMENT_FAILURE", RecoveryClass.TERMINAL, False),
    ("DUPLICATE_INVOICE_ID", "VALIDATION_ERROR", RecoveryClass.TERMINAL, False),
    ("PAYPAL_REQUEST_ID_REQUIRED", "VALIDATION_ERROR", RecoveryClass.TERMINAL, False),
    ("ORDER_ALREADY_CAPTURED", None, RecoveryClass.RECONCILE, False),
    ("PREVIOUS_REQUEST_IN_PROGRESS", None, RecoveryClass.RECONCILE, True),
])
def test_issue_code_classification(issue, cat, cls, retry):
    c = classify(_e(issues=[issue]))
    assert (c.category, c.recovery_class, c.retry_same_request) == (cat, cls, retry)


def test_server_errors_and_transport_are_auto_retryable_with_state_check():
    for e, basis in [(_e(500, "INTERNAL_SERVER_ERROR"), "DOCUMENTED"), (_e(503, "SERVICE_UNAVAILABLE"), "MODELLING_ASSUMPTION"),
                     (_e(None, "NETWORK_ERROR", transport=True), "DOCUMENTED")]:
        c = classify(e)
        assert c.recovery_class == RecoveryClass.AUTO_RETRY and c.retry_same_request and c.requires_state_check and c.basis == basis


def test_unknown_errors_are_terminal_never_retried_and_marked_unknown():
    c = classify(_e(422, "UNPROCESSABLE_ENTITY", ["SOMETHING_PAYPAL_NEVER_DOCUMENTED"]))
    assert c.recovery_class == RecoveryClass.TERMINAL and not c.retry_same_request and c.basis == "UNKNOWN_CONSERVATIVE"
    assert classify(_e(418, "TEAPOT")).recovery_class == RecoveryClass.TERMINAL
    assert classify(_e(401, "AUTHENTICATION_FAILURE")).category is None


def test_classifier_only_uses_simulation_category_names_and_never_retries_buyer_or_terminal():
    sim = {c.name for c in FailureCategory}
    for row in classification_table():
        assert row["category"] is None or row["category"] in sim
        if row["recovery_class"] in ("BUYER_ACTION", "TERMINAL"):
            assert row["retry_same_request"] is False
    assert {r["basis"] for r in classification_table()} <= {"DOCUMENTED", "MODELLING_ASSUMPTION"}


# ============================================================ persistence / separation of simulated vs sandbox data
def test_sandbox_tables_are_pinned_to_environment_and_separate_from_simulation(tmp_path):
    store = SandboxStore(str(tmp_path / "s.db"))
    tables = {r[0] for r in store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert tables >= {"sandbox_executions", "sandbox_api_calls", "sandbox_transitions", "sandbox_validation_records", "sandbox_negative_tests"}
    assert not any(t.startswith("payment_sim") for t in tables)
    ex, created = store.get_or_create_execution("exp", "ref", "pid", "pname", {"a": 1}, "10.00", "USD")
    assert created and ex["environment"] == "PAYPAL_SANDBOX"
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("UPDATE sandbox_executions SET environment='SIMULATED' WHERE execution_id=?", (ex["execution_id"],))
    with pytest.raises(sqlite3.IntegrityError):
        store.conn.execute("INSERT INTO sandbox_api_calls (ts,operation,method,url,environment) VALUES ('t','o','GET','u','SIMULATED')")
    ex2, created2 = store.get_or_create_execution("exp", "ref", "pid", "pname", {"a": 1}, "10.00", "USD")
    assert not created2 and ex2["execution_id"] == ex["execution_id"]  # duplicate prevention


def test_validation_summary_has_counts_only_no_rates(tmp_path):
    store = SandboxStore(str(tmp_path / "s.db"))
    ex, _ = store.get_or_create_execution("exp", "ref", "pid", "pname", {}, "10.00", "USD")
    for outcome in ("CONSISTENT", "CONSISTENT", "DIVERGED"):
        store.add_validation_record({"execution_id": ex["execution_id"], "experiment_id": "exp", "selected_policy_id": "pid", "api_operation": "X",
                                     "actual_action": "A", "predicted_vs_observed": {}, "validation_outcome": outcome})
    s = store.validation_summary()
    assert s["outcome_counts"] == {"CONSISTENT": 2, "DIVERGED": 1} and s["environment"] == "PAYPAL_SANDBOX"
    assert not any("rate" in k or "success" in k for k in s) and "not comparable" in s["note"]


# ---- defence-in-depth layers (each guarded independently of the others)
def test_host_is_rechecked_on_every_request_even_if_the_base_url_is_tampered_with():
    fake, c = make()
    c.base = "https://api-m.paypal.com"  # simulate tampering after construction
    with pytest.raises(SandboxViolation):
        c._send("GET", "/v1/oauth2/token", operation="X")
    assert fake.calls == []  # nothing was sent


def test_the_store_itself_scrubs_secrets_even_if_a_caller_forgets_to_redact(tmp_path):
    store = SandboxStore(str(tmp_path / "s.db"), secrets=["TOPSECRETVALUE"])
    store.record_api_call({"operation": "X", "method": "GET", "url": "https://u", "request_headers": {"Authorization": "Bearer TOPSECRETVALUE"},
                           "request_body": {"note": "has TOPSECRETVALUE"}, "response_body": {"access_token": "abc", "m": "TOPSECRETVALUE"}, "status": 200})
    row = store.api_calls()[0]
    dump = json.dumps(row)
    assert "TOPSECRETVALUE" not in dump and "abc" not in dump
