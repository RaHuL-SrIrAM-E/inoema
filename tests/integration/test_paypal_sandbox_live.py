"""LIVE PayPal SANDBOX integration tests -- these call the real Sandbox API (no mocks).

Run:  PAYPAL_CLIENT_ID=... PAYPAL_CLIENT_SECRET=... python -m pytest tests/integration -m paypal_live -v -s
Without credentials they are SKIPPED. A skip is NOT a pass: it means the real Sandbox was not exercised.

Buyer approval cannot be automated (it needs a human logging in with a Sandbox PERSONAL account). Procedure:
  1) python -m scripts.paypal_sandbox_validation run --reference REF --demo-policy     (prints the approval URL)
  2) approve in the browser
  3) PAYPAL_APPROVED_REFERENCE=REF PAYPAL_EXPERIMENT_ID=demo-experiment python -m pytest tests/integration -m paypal_live -k approved -s
"""
import os
import uuid

import pytest

from spikes.envutil import load_env

load_env()
HAVE = bool(os.environ.get("PAYPAL_CLIENT_ID") and os.environ.get("PAYPAL_CLIENT_SECRET"))
pytestmark = [pytest.mark.paypal_live, pytest.mark.skipif(not HAVE, reason="PAYPAL_CLIENT_ID/SECRET not set: the real PayPal Sandbox was NOT exercised")]

from paypal.client import PayPalClient  # noqa: E402
from paypal.error_classifier import RecoveryClass, classify  # noqa: E402
from paypal.execution import ExecStatus, ExecutionPolicy, PaymentRequest, RecoveryExecutor  # noqa: E402
from paypal.models import OrderState, PayPalApiError, PayPalConfig  # noqa: E402
from paypal.orders import OrdersApi  # noqa: E402
from paypal.persistence import SandboxStore  # noqa: E402
from scripts.paypal_sandbox_validation import DEMO_POLICY  # noqa: E402


@pytest.fixture()
def rig(tmp_path):
    cfg = PayPalConfig.from_env()
    store = SandboxStore(str(tmp_path / "live.db"), secrets=cfg.secrets())
    return cfg, store, PayPalClient(cfg, sink=store)


def test_oauth_token_retrieval(rig):
    _, store, client = rig
    assert client.tokens.get_token()
    row = store.api_calls()[0]
    assert row["operation"] == "OAUTH_TOKEN" and row["status"] == 200 and row["paypal_debug_id"]
    assert "access_token" not in row["response_body"].replace("***REDACTED***", "") or '"access_token": "***REDACTED***"' in row["response_body"]


def test_create_and_get_order_and_approval_url(rig):
    _, store, client = rig
    api = OrdersApi(client)
    created = api.create_order("10.00", "USD", f"live-{uuid.uuid4().hex[:8]}", str(uuid.uuid4()))
    fetched = api.get_order(created.id)
    print("created.status =", created.raw_status, "| fetched.status =", fetched.raw_status, "| approval_url present:", bool(created.approval_url))
    assert created.id == fetched.id and created.approval_url
    assert fetched.state in (OrderState.CREATED, OrderState.PAYER_ACTION_REQUIRED), f"unexpected order status {fetched.raw_status!r}: update the lifecycle assumptions"


def test_capturing_an_unapproved_order_yields_a_real_buyer_action_error(rig):
    _, store, client = rig
    api = OrdersApi(client)
    order = api.create_order("10.00", "USD", f"live-{uuid.uuid4().hex[:8]}", str(uuid.uuid4()))
    with pytest.raises(PayPalApiError) as e:
        api.capture_order(order.id, str(uuid.uuid4()))
    c = classify(e.value)
    print("REAL unapproved-capture error:", e.value.status, e.value.name, e.value.issues, "->", c.to_dict(), "debug_id", e.value.debug_id)
    assert e.value.debug_id
    assert c.recovery_class == RecoveryClass.BUYER_ACTION, f"observed {e.value.issues}: classifier needs updating"


def test_executor_stops_at_buyer_approval_and_is_idempotent(rig):
    cfg, store, client = rig
    policy = ExecutionPolicy.from_definition(DEMO_POLICY, feasible=True, source="live-test")
    ex = RecoveryExecutor(client, store, policy, "live-exp", sleep=lambda s: None)
    req = PaymentRequest("10.00", "USD", f"live-{uuid.uuid4().hex[:8]}")
    a = ex.execute(req)
    b = ex.execute(req)
    assert a.status == ExecStatus.AWAITING_BUYER and a.order_id == b.order_id and a.approval_url
    assert sum(1 for c in store.api_calls() if c["operation"] == "CREATE_ORDER") == 1  # one real order, despite two executions
    assert not any(c["operation"] == "CAPTURE_ORDER" for c in store.api_calls())  # never captured an unapproved order


@pytest.mark.skipif(not os.environ.get("PAYPAL_APPROVED_REFERENCE"), reason="no buyer-approved reference supplied (manual step; see module docstring)")
def test_approved_order_is_captured_and_verified():
    cfg = PayPalConfig.from_env()
    store = SandboxStore(os.environ.get("INOEMA_PAYPAL_DB"), secrets=cfg.secrets())
    ex = RecoveryExecutor(PayPalClient(cfg, sink=store), store, ExecutionPolicy.from_definition(DEMO_POLICY, feasible=True, source="live-test"),
                          os.environ.get("PAYPAL_EXPERIMENT_ID", "demo-experiment"), sleep=lambda s: None)
    res = ex.resume(os.environ["PAYPAL_APPROVED_REFERENCE"])
    print("status", res.status, "order", res.order_id, "capture", res.capture_id, "verified", res.verification and res.verification.ok)
    assert res.status == ExecStatus.COMPLETED and res.capture_id and res.verification.ok
