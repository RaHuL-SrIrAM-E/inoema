"""OFFLINE tests for the resume-and-capture workflow (paypal/resume.py). HTTP is a MOCK (tests/paypal_fake.py) -- NOT PayPal.
They prove OUR safety logic: refusals, no blind capture, no re-capture, reconciliation, persistence, redaction.
No test here (or the code under test) can reach the network: every client uses httpx.MockTransport."""
import importlib
import json
import uuid

import pytest

from paypal.client import PayPalClient
from paypal.execution import NAMESPACE, ExecStatus
from paypal.models import PayPalConfig, SandboxViolation
from paypal.orders import OrdersApi
from paypal.persistence import SandboxStore
from paypal.resume import EXPERIMENT_ID, ResumeCapture, ResumeOutcome, check_identity
from tests.paypal_fake import FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_TOKEN, FakePayPal

REF = "INOEMA-MILESTONE-ORDER-001"
CFG = PayPalConfig(client_id=FAKE_CLIENT_ID, client_secret=FAKE_CLIENT_SECRET)


class Rig:
    def __init__(self, tmp_path):
        self.fake = FakePayPal()
        self.store = SandboxStore(str(tmp_path / "sandbox.db"), secrets=CFG.secrets())
        self.client = PayPalClient(CFG, sink=self.store, transport=self.fake.transport())
        # the "existing" order was created beforehand (by an earlier session): create it, then forget those calls
        o = OrdersApi(self.client).create_order("10.00", "USD", REF, str(uuid.uuid4()), invoice_id=REF)
        self.order_id = o.id
        self.store.conn.execute("DELETE FROM sandbox_api_calls")
        self.store.conn.commit()
        self.fake.calls.clear()
        self.wf = ResumeCapture(self.client, self.store)

    def run(self, confirm=False, **kw):
        a = dict(order_id=self.order_id, reference=REF, amount="10.00", currency="USD")
        a.update(kw)
        return self.wf.run(a["order_id"], a["reference"], a["amount"], a["currency"], confirm_capture=confirm)

    def ops(self):
        return [r["operation"] for r in self.store.api_calls()]

    def captures(self):
        return self.fake.count("POST", "/capture")

    def execution(self):
        return self.store.get_execution_by_reference(EXPERIMENT_ID, REF)

    def triggers(self):
        return [t["trigger"] for t in self.store.transitions(self.execution()["execution_id"])]

    @property
    def raw(self):
        return self.fake.orders[self.order_id]


@pytest.fixture()
def rig(tmp_path):
    return Rig(tmp_path)


# ---------------------------------------------------------------- order is fetched first; nothing is created
def test_first_api_call_is_get_of_the_real_order_and_no_order_is_created(rig):
    rig.run()
    non_oauth = [(m, p) for m, p, _ in rig.fake.calls if p != "/v1/oauth2/token"]
    assert non_oauth[0] == ("GET", f"/v2/checkout/orders/{rig.order_id}")
    assert rig.fake.count("POST", "/v2/checkout/orders") == 0
    assert "CREATE_ORDER" not in rig.ops()


# ---------------------------------------------------------------- refusal when not approved
@pytest.mark.parametrize("status", ["CREATED", "PAYER_ACTION_REQUIRED"])
def test_not_approved_refuses_capture_even_with_confirm(rig, status):
    rig.fake.set_status(rig.order_id, status)
    r = rig.run(confirm=True)
    assert r.outcome == ResumeOutcome.NOT_APPROVED and not r.ok
    assert "not APPROVED" in r.reason and r.approval_url
    assert rig.captures() == 0 and "CAPTURE_ORDER" not in rig.ops()
    assert rig.execution()["status"] == ExecStatus.AWAITING_BUYER.value


def test_voided_order_is_refused(rig):
    rig.fake.set_status(rig.order_id, "VOIDED")
    r = rig.run(confirm=True)
    assert r.outcome == ResumeOutcome.ORDER_TERMINAL and rig.captures() == 0


# ---------------------------------------------------------------- explicit confirmation
def test_approved_without_confirm_does_not_capture(rig):
    rig.fake.approve(rig.order_id)
    r = rig.run(confirm=False)
    assert r.outcome == ResumeOutcome.CONFIRMATION_REQUIRED and not r.capture_requested
    assert rig.captures() == 0 and rig.raw["status"] == "APPROVED"
    assert "CAPTURE_NOT_CONFIRMED" in rig.triggers()


# ---------------------------------------------------------------- identity mismatches: never capture, even if approved + confirmed
def _approved_then(rig, mutate):
    rig.fake.approve(rig.order_id)
    mutate(rig.raw)
    return rig.run(confirm=True)


@pytest.mark.parametrize("name,mutate,needle", [
    ("amount", lambda o: o["purchase_units"][0]["amount"].update(value="11.00"), "amount"),
    ("currency", lambda o: o["purchase_units"][0]["amount"].update(currency_code="EUR"), "currency"),
    ("invoice", lambda o: o["purchase_units"][0].update(invoice_id="SOMEONE-ELSES-INVOICE"), "invoice_id"),
    ("reference", lambda o: o["purchase_units"][0].update(reference_id="OTHER-REF"), "reference_id"),
    ("intent", lambda o: o.update(intent="AUTHORIZE"), "intent"),
])
def test_mismatch_refuses_capture(rig, name, mutate, needle):
    r = _approved_then(rig, mutate)
    assert r.outcome == ResumeOutcome.MISMATCH, name
    assert any(needle in p for p in r.problems)
    assert rig.captures() == 0 and "CAPTURE_ORDER" not in rig.ops()
    assert rig.execution()["status"] != ExecStatus.COMPLETED.value


def test_operator_supplied_wrong_amount_is_refused_before_any_capture(rig):
    rig.fake.approve(rig.order_id)
    r = rig.run(confirm=True, amount="25.00")  # order is 10.00
    assert r.outcome == ResumeOutcome.MISMATCH and rig.captures() == 0


def test_check_identity_flags_wrong_order_id_and_multiple_captures(rig):
    from paypal.orders import parse_order
    o = parse_order(rig.raw)
    assert check_identity(o, order_id=rig.order_id, reference=REF, amount=o.amount, currency="USD") == []
    assert any("order id" in p for p in check_identity(o, order_id="OTHERID123", reference=REF, amount=o.amount, currency="USD"))
    rig.raw["purchase_units"][0]["payments"] = {"captures": [{"id": "A"}, {"id": "B"}]}
    assert any("more than one capture" in p for p in check_identity(parse_order(rig.raw), order_id=rig.order_id, reference=REF, amount=o.amount, currency="USD"))


def test_reference_bound_to_another_order_is_refused(rig):
    rig.run()
    with pytest.raises(ValueError, match="different order"):
        rig.wf.run("ANOTHERORDER99", REF, "10.00", "USD")


# ---------------------------------------------------------------- happy path
def test_approved_and_confirmed_captures_once_and_verifies_from_fresh_get(rig):
    rig.fake.approve(rig.order_id)
    r = rig.run(confirm=True)
    assert r.outcome == ResumeOutcome.CAPTURED_AND_VERIFIED and r.ok and r.capture_id and r.verification.ok
    assert rig.captures() == 1
    ops = rig.ops()
    cap_i = ops.index("CAPTURE_ORDER")
    assert ops[cap_i + 1:] == ["GET_ORDER", "GET_CAPTURE"]  # fresh GET of the order then the capture, AFTER the capture call
    assert rig.execution()["status"] == ExecStatus.COMPLETED.value and rig.execution()["capture_id"] == r.capture_id
    expected_rid = str(uuid.uuid5(NAMESPACE, f"{r.execution_id}:capture_order"))
    assert rig.fake.capture_request_ids == [expected_rid]
    assert {"ORDER_ADOPTED", "CAPTURE_REQUESTED", "CAPTURE_RESPONSE", "CAPTURE_VERIFIED"} <= set(rig.triggers())


def test_evidence_is_persisted_and_redacted(rig):
    rig.fake.approve(rig.order_id)
    rig.run(confirm=True)
    dump = json.dumps([dict(r) for r in rig.store.conn.execute("SELECT * FROM sandbox_api_calls")])
    for secret in (FAKE_TOKEN, FAKE_CLIENT_SECRET, FAKE_CLIENT_ID):
        assert secret not in dump
    assert all(r["environment"] == "PAYPAL_SANDBOX" for r in rig.store.api_calls())
    assert all(r["execution_id"] == rig.execution()["execution_id"] for r in rig.store.api_calls() if r["operation"] != "OAUTH_TOKEN")


# ---------------------------------------------------------------- already captured / never re-capture
def test_order_already_completed_at_paypal_is_verified_never_captured(rig):
    rig.fake.approve(rig.order_id)
    rig.fake._complete(rig.order_id)  # someone captured it earlier
    r = rig.run(confirm=True)
    assert r.outcome == ResumeOutcome.ALREADY_CAPTURED_VERIFIED and r.ok and not r.capture_requested
    assert rig.captures() == 0 and "CAPTURE_ORDER" not in rig.ops()


def test_rerun_after_success_makes_no_api_calls_and_no_second_capture(rig):
    rig.fake.approve(rig.order_id)
    rig.run(confirm=True)
    calls_before = len(rig.fake.calls)
    r = rig.run(confirm=True)
    assert r.outcome == ResumeOutcome.ALREADY_FINISHED_LOCALLY and r.new_api_calls == 0
    assert len(rig.fake.calls) == calls_before and rig.captures() == 1


# ---------------------------------------------------------------- ambiguous capture responses
def test_lost_response_after_capture_succeeded_is_reconciled_by_get_not_recaptured(rig):
    rig.fake.approve(rig.order_id)
    rig.fake.capture_script = ["ok_then_timeout"]  # PayPal captured, but we never saw the response
    r = rig.run(confirm=True)
    assert rig.captures() == 1  # exactly one capture request
    assert r.outcome == ResumeOutcome.ALREADY_CAPTURED_VERIFIED and r.ok
    assert "RECONCILE_AFTER_AMBIGUOUS_CAPTURE" in rig.triggers()
    ops = rig.ops()
    assert ops.index("GET_ORDER", ops.index("CAPTURE_ORDER")) > ops.index("CAPTURE_ORDER")


@pytest.mark.parametrize("step", ["timeout", ("error", 500, "INTERNAL_SERVER_ERROR", None)])
def test_ambiguous_capture_with_order_still_approved_stops_without_second_capture(rig, step):
    rig.fake.approve(rig.order_id)
    rig.fake.capture_script = [step]
    r = rig.run(confirm=True)
    assert r.outcome == ResumeOutcome.AMBIGUOUS and not r.ok
    assert rig.captures() == 1  # no automatic second capture
    assert rig.ops()[-1] == "GET_ORDER"  # order re-fetched after the ambiguous response
    assert rig.execution()["status"] != ExecStatus.COMPLETED.value


def test_manual_rerun_after_ambiguity_uses_the_same_idempotency_key(rig):
    rig.fake.approve(rig.order_id)
    rig.fake.capture_script = ["timeout"]
    rig.run(confirm=True)
    r = rig.run(confirm=True)  # operator re-runs: fresh GET first, then one more capture with the SAME key
    assert r.outcome == ResumeOutcome.CAPTURED_AND_VERIFIED
    assert len(set(rig.fake.capture_request_ids)) == 1 and len(rig.fake.capture_request_ids) == 2


def test_already_captured_error_race_is_reconciled(rig):
    rig.fake.approve(rig.order_id)
    rig.fake.capture_script = ["race"]  # 422 ORDER_ALREADY_CAPTURED
    r = rig.run(confirm=True)
    assert rig.captures() == 1 and r.ok and r.outcome == ResumeOutcome.ALREADY_CAPTURED_VERIFIED


# ---------------------------------------------------------------- terminal failures are never retried
def test_terminal_capture_failure_is_not_retried_and_blocks_reruns(rig):
    rig.fake.approve(rig.order_id)
    rig.fake.capture_script = [("error", 422, "UNPROCESSABLE_ENTITY", "PAYMENT_DENIED")]
    r = rig.run(confirm=True)
    assert r.outcome == ResumeOutcome.CAPTURE_FAILED and rig.captures() == 1
    assert rig.execution()["status"] == ExecStatus.TERMINATED.value
    r2 = rig.run(confirm=True)
    assert r2.outcome == ResumeOutcome.ALREADY_FINISHED_LOCALLY and rig.captures() == 1


def test_verification_failure_is_terminal_and_not_recaptured(rig):
    rig.fake.approve(rig.order_id)
    rig.fake.bad_amount = True  # capture reports 1.00 instead of 10.00
    r = rig.run(confirm=True)
    assert r.outcome == ResumeOutcome.VERIFICATION_FAILED and any("amount" in p for p in r.problems)
    assert rig.execution()["status"] == ExecStatus.TERMINATED.value
    rig.run(confirm=True)
    assert rig.captures() == 1


# ---------------------------------------------------------------- sandbox-only + CLI + live opt-in
def test_live_host_is_refused():
    with pytest.raises(SandboxViolation):
        PayPalClient(PayPalConfig(client_id="a", client_secret="b", base_url="https://api-m.paypal.com"))


def test_cli_requires_explicit_inputs_and_makes_no_call(monkeypatch, tmp_path):
    monkeypatch.setenv("PAYPAL_CLIENT_ID", "x")
    monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "y")
    from scripts import paypal_sandbox_validation as cli
    with pytest.raises(SystemExit) as e:
        cli.main(["resume-capture", "--order-id", "ABC123", "--db", str(tmp_path / "x.db")])  # no reference/amount/currency
    assert e.value.code == 2


def _skip_conditions(monkeypatch, opt_in):
    monkeypatch.setenv("PAYPAL_CLIENT_ID", "x")
    monkeypatch.setenv("PAYPAL_CLIENT_SECRET", "y")
    if opt_in:
        monkeypatch.setenv("INOEMA_ALLOW_LIVE_PAYPAL", "1")
    else:
        monkeypatch.delenv("INOEMA_ALLOW_LIVE_PAYPAL", raising=False)
    mod = importlib.import_module("tests.integration.test_paypal_sandbox_live")
    mod = importlib.reload(mod)
    return [m.args[0] for m in mod.pytestmark if m.name == "skipif"]


def test_live_paypal_tests_are_skipped_without_explicit_opt_in_even_with_credentials(monkeypatch):
    conds = _skip_conditions(monkeypatch, opt_in=False)
    assert any(conds), "live PayPal tests would RUN without INOEMA_ALLOW_LIVE_PAYPAL=1"


def test_live_paypal_tests_run_only_with_opt_in_and_credentials(monkeypatch):
    assert not any(_skip_conditions(monkeypatch, opt_in=True))
    importlib.reload(importlib.import_module("tests.integration.test_paypal_sandbox_live"))
