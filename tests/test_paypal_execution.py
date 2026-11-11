"""OFFLINE unit tests for the recovery executor. HTTP is a MOCK (tests/paypal_fake.py) -- NOT PayPal. These tests prove
the executor's decision logic (idempotency, no blind retries, state checks, buyer/terminal handling, persistence); they do
not prove how the real Sandbox behaves."""
import json
from dataclasses import replace

import pytest

from paypal.client import PayPalClient
from paypal.execution import (ACTION_SUPPORT, ExecStatus, ExecutionPolicy, PaymentRequest, PolicyNotApproved, RecoveryExecutor)
from paypal.models import PayPalConfig
from paypal.persistence import SandboxStore
from tests.paypal_fake import FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FAKE_TOKEN, FakePayPal

CFG = PayPalConfig(client_id=FAKE_CLIENT_ID, client_secret=FAKE_CLIENT_SECRET)

INTELLIGENT = {"name": "intel", "retry_limit": 2, "retry_intervals": [5.0], "exponential_backoff": True, "backoff_base_minutes": 5.0,
               "backoff_multiplier": 2.0, "backoff_max_interval_minutes": 240.0, "maximum_recovery_window": 4320.0, "buyer_action_timeout": 1440.0,
               "request_buyer_action": True, "alternate_funding_action": "REQUEST_ALTERNATE", "terminal_failure_handling": "FLAG_FOR_REVIEW",
               "classification_aware": True, "idempotent_retries": True}
BLIND = {**INTELLIGENT, "name": "blind", "request_buyer_action": False, "alternate_funding_action": "NONE", "terminal_failure_handling": "ABANDON",
         "classification_aware": False}
REQ = PaymentRequest("25.00", "USD", "order-ref-001")


class Rig:
    def __init__(self, tmp_path, policy_def=INTELLIGENT, **kw):
        self.fake = FakePayPal()
        self.store = SandboxStore(str(tmp_path / "sandbox.db"))
        self.client = PayPalClient(CFG, transport=self.fake.transport())
        self.sleeps: list[float] = []
        self.policy = ExecutionPolicy.from_definition(policy_def, feasible=True, source="test")
        self.ex = RecoveryExecutor(self.client, self.store, self.policy, "exp-1", sleep=self.sleeps.append, **kw)

    def first_order(self):
        return next(iter(self.fake.orders))

    def records(self):
        return self.store.validation_records()

    def outcomes(self):
        return [r["validation_outcome"] for r in self.records()]


# ============================================================ happy path with buyer approval
def test_create_stops_for_buyer_approval_without_capturing_then_resume_captures_and_verifies(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    assert res.status == ExecStatus.AWAITING_BUYER and res.order_id and res.approval_url and res.approval_url.endswith(res.order_id)
    assert r.fake.count("POST", "/capture") == 0  # an unapproved order is never captured
    rec = r.records()[0]
    assert rec["validation_outcome"] == "PENDING_BUYER" and rec["actual_action"] == "AWAIT_BUYER_APPROVAL" and "state-derived" in rec["observed_error_category"]

    r.fake.approve(res.order_id)  # (the human buyer step in the real Sandbox)
    res2 = r.ex.resume(REQ.reference_id)
    assert res2.status == ExecStatus.COMPLETED and res2.capture_id and res2.verification.ok
    assert r.fake.count("POST", "/capture") == 1 and r.fake.count("POST", "/v2/checkout/orders") == 1
    final = r.records()[-1]
    assert final["validation_outcome"] == "CONSISTENT" and final["capture_id"] == res2.capture_id and final["order_id"] == res.order_id
    states = [(t["from_state"], t["to_state"], t["trigger"]) for t in r.store.transitions(res.execution_id)]
    assert ("NEW", "ORDER_CREATED", "ORDER_CREATED") in states and states[-1][1] == "COMPLETED"
    row = r.store.get_execution(res.execution_id)
    assert row["environment"] == "PAYPAL_SANDBOX" and row["capture_id"] == res2.capture_id


def test_validation_record_has_every_required_field(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.ex.resume(REQ.reference_id)
    for rec in r.records():
        for f in ("experiment_id", "selected_policy_id", "order_id", "api_operation", "actual_api_status", "executed_at", "predicted_action",
                  "actual_action", "predicted_vs_observed_json", "validation_outcome", "environment"):
            assert rec[f] not in (None, ""), f
        body = json.loads(rec["predicted_vs_observed_json"])
        assert {"predicted", "observed", "scope"} <= set(body) and "never combined" in body["scope"]
        assert rec["selected_policy_id"] == r.policy.policy_id and rec["experiment_id"] == "exp-1"
    assert r.records()[-1]["capture_id"]


# ============================================================ duplicate execution prevention + idempotency
def test_duplicate_execution_never_creates_a_second_order_or_capture(tmp_path):
    r = Rig(tmp_path)
    a = r.ex.execute(REQ)
    b = r.ex.execute(REQ)  # same reference while awaiting buyer: re-inspects the SAME order
    assert a.order_id == b.order_id and r.fake.count("POST", "/v2/checkout/orders") == 1
    r.fake.approve(a.order_id)
    done = r.ex.resume(REQ.reference_id)
    calls_before = len(r.fake.calls)
    again = r.ex.execute(REQ)
    assert again.status == ExecStatus.COMPLETED and again.new_api_calls == 0 and len(r.fake.calls) == calls_before  # zero API calls
    assert r.fake.count("POST", "/capture") == 1 and again.capture_id == done.capture_id
    with pytest.raises(ValueError):
        r.ex.execute(PaymentRequest("26.00", "USD", REQ.reference_id))  # same reference, different amount


def test_request_ids_are_deterministic_per_operation_and_reused_on_retry(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = [("error", 500, "INTERNAL_SERVER_ERROR", None), "ok"]
    r.ex.resume(REQ.reference_id)
    assert len(r.fake.capture_request_ids) == 2 and r.fake.capture_request_ids[0] == r.fake.capture_request_ids[1]  # SAME key on the retry
    assert r.fake.capture_request_ids[0] != next(h["paypal-request-id"] for m, p, h in r.fake.calls if p == "/v2/checkout/orders")
    # a fresh execution derives different keys
    r2 = Rig(tmp_path / "other")
    res2 = r2.ex.execute(PaymentRequest("25.00", "USD", "order-ref-002"))
    r2.fake.approve(res2.order_id)
    r2.ex.resume("order-ref-002")
    assert r2.fake.capture_request_ids[0] != r.fake.capture_request_ids[0]


# ============================================================ transient failures: retry only after a state check
def test_transient_500_is_retried_with_same_key_after_checking_order_state(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = [("error", 500, "INTERNAL_SERVER_ERROR", None), "ok"]
    gets_before = r.fake.count("GET", res.order_id)
    out = r.ex.resume(REQ.reference_id)
    assert out.status == ExecStatus.COMPLETED and r.fake.count("POST", "/capture") == 2
    assert r.fake.count("GET", res.order_id) > gets_before + 1  # GET order happened between the failed and the retried capture
    assert r.sleeps == [5.0]  # policy delay 5 min scaled at 1 s/min
    trig = [t["trigger"] for t in r.store.transitions(res.execution_id)]
    assert "API_ERROR" in trig and "RETRY_SCHEDULED" in trig and trig.index("API_ERROR") < trig.index("RETRY_SCHEDULED")
    rec = next(x for x in r.records() if x["api_operation"] == "CAPTURE_ORDER")
    assert rec["predicted_action"] == "RETRY" and rec["actual_action"] == "RETRY" and rec["validation_outcome"] == "CONSISTENT"
    assert r.outcomes()[-1] == "CONSISTENT"  # completed after recovery


def test_lost_response_after_successful_capture_is_reconciled_not_recaptured(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = ["ok_then_timeout"]  # PayPal captured, but we never saw the response
    out = r.ex.resume(REQ.reference_id)
    assert out.status == ExecStatus.COMPLETED and out.verification.ok
    assert r.fake.count("POST", "/capture") == 1  # NO second capture
    assert "RECONCILED_ALREADY_COMPLETED" in [t["trigger"] for t in r.store.transitions(res.execution_id)]


def test_order_already_captured_error_is_reconciled_by_fetching_the_order(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = ["race"]  # order is APPROVED when we look, but is captured by someone else before our capture lands
    out = r.ex.resume(REQ.reference_id)
    assert out.status == ExecStatus.COMPLETED and out.verification.ok
    assert r.fake.count("POST", "/capture") == 1  # exactly the one failed attempt; no re-capture
    trig = [t["trigger"] for t in r.store.transitions(res.execution_id)]
    assert trig.index("API_ERROR") < trig.index("RECONCILED_ALREADY_COMPLETED") and "RETRY_SCHEDULED" not in trig


def test_network_timeout_on_create_is_retried_with_same_key_and_creates_one_order(tmp_path):
    r = Rig(tmp_path)
    r.fake.create_script = ["ok_then_timeout"]  # order created server-side, response lost
    res = r.ex.execute(REQ)
    assert res.status == ExecStatus.AWAITING_BUYER and len(r.fake.orders) == 1  # same PayPal-Request-Id -> same order
    keys = [h["paypal-request-id"] for m, p, h in r.fake.calls if p == "/v2/checkout/orders" and m == "POST"]
    assert len(keys) == 2 and keys[0] == keys[1]


def test_retry_limit_hard_cap_and_recovery_window(tmp_path):
    r = Rig(tmp_path)  # policy retry_limit = 2
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = [("error", 500, "INTERNAL_SERVER_ERROR", None)] * 10
    out = r.ex.resume(REQ.reference_id)
    assert out.status == ExecStatus.TERMINATED and r.fake.count("POST", "/capture") == 3 and out.reason == "retry_limit_reached"  # 1 + 2 retries

    r2 = Rig(tmp_path / "cap", hard_max_retries=1)  # executor hard cap below the policy limit
    a = r2.ex.execute(REQ)
    r2.fake.approve(a.order_id)
    r2.fake.capture_script = [("error", 500, "INTERNAL_SERVER_ERROR", None)] * 10
    out2 = r2.ex.resume(REQ.reference_id)
    assert r2.fake.count("POST", "/capture") == 2 and out2.reason == "executor_hard_retry_cap"

    r3 = Rig(tmp_path / "win", policy_def={**INTELLIGENT, "maximum_recovery_window": 12.0})
    b = r3.ex.execute(REQ)
    r3.fake.approve(b.order_id)
    r3.fake.capture_script = [("error", 500, "INTERNAL_SERVER_ERROR", None)] * 10
    out3 = r3.ex.resume(REQ.reference_id)
    assert r3.fake.count("POST", "/capture") == 2 and out3.reason == "recovery_window_exceeded"  # retries at +5, +15 > 12
    assert r.sleeps == [5.0, 10.0]


def test_wait_is_scaled_and_capped(tmp_path):
    r = Rig(tmp_path, seconds_per_policy_minute=60.0, max_wait_s=7.0)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = [("error", 500, "INTERNAL_SERVER_ERROR", None), "ok"]
    r.ex.resume(REQ.reference_id)
    assert r.sleeps == [7.0]  # 5 policy-minutes * 60 = 300 s, capped to 7 s


# ============================================================ buyer intervention and terminal failures: never auto-retried
def test_buyer_action_failure_stops_for_the_buyer_and_is_not_retried(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = [("error", 422, "UNPROCESSABLE_ENTITY", "INSTRUMENT_DECLINED")]
    out = r.ex.resume(REQ.reference_id)
    assert out.status == ExecStatus.AWAITING_BUYER and r.fake.count("POST", "/capture") == 1 and r.sleeps == []
    assert out.last_classification["category"] == "INSTRUMENT_DECLINED" and out.last_classification["recovery_class"] == "BUYER_ACTION"
    rec = next(x for x in r.records() if x["api_operation"] == "CAPTURE_ORDER")
    assert (rec["predicted_action"], rec["actual_action"], rec["validation_outcome"]) == ("REQUEST_BUYER_ACTION", "REQUEST_BUYER_ACTION", "PENDING_BUYER")
    assert rec["observed_error_category"] == "INSTRUMENT_DECLINED"
    trans = r.store.transitions(res.execution_id)[-1]
    assert trans["trigger"] == "BUYER_ACTION_REQUESTED" and json.loads(trans["detail_json"])["buyer_notified"] is False  # notification NOT sent
    # after the buyer acts, resume captures once
    r.ex.resume(REQ.reference_id)
    assert r.fake.count("POST", "/capture") == 2


def test_blind_policy_is_blocked_from_retrying_buyer_action_failures(tmp_path):
    r = Rig(tmp_path, policy_def=BLIND)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = [("error", 422, "UNPROCESSABLE_ENTITY", "INSTRUMENT_DECLINED")] * 5
    out = r.ex.resume(REQ.reference_id)
    assert out.status == ExecStatus.TERMINATED and r.fake.count("POST", "/capture") == 1 and out.reason.startswith("blocked:")
    rec = next(x for x in r.records() if x["api_operation"] == "CAPTURE_ORDER")
    assert (rec["predicted_action"], rec["actual_action"], rec["validation_outcome"]) == ("RETRY", "BLOCKED_RETRY", "DIVERGED")


@pytest.mark.parametrize("policy_def,expected_actual", [(INTELLIGENT, "TERMINATE"), (BLIND, "BLOCKED_RETRY")])
def test_terminal_failures_are_never_retried(tmp_path, policy_def, expected_actual):
    r = Rig(tmp_path, policy_def=policy_def)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = [("error", 422, "UNPROCESSABLE_ENTITY", "TRANSACTION_REFUSED")] * 5
    out = r.ex.resume(REQ.reference_id)
    assert out.status == ExecStatus.TERMINATED and r.fake.count("POST", "/capture") == 1 and r.sleeps == []
    rec = next(x for x in r.records() if x["api_operation"] == "CAPTURE_ORDER")
    assert rec["actual_action"] == expected_actual and rec["observed_error_category"] == "TERMINAL_PAYMENT_FAILURE"
    # terminal executions stay terminal: re-executing makes no calls
    n = len(r.fake.calls)
    r.ex.execute(REQ)
    assert len(r.fake.calls) == n


def test_unknown_error_codes_are_treated_as_terminal(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = [("error", 422, "UNPROCESSABLE_ENTITY", "A_CODE_NOBODY_DOCUMENTED")] * 3
    out = r.ex.resume(REQ.reference_id)
    assert out.status == ExecStatus.TERMINATED and r.fake.count("POST", "/capture") == 1
    assert out.last_classification["basis"] == "UNKNOWN_CONSERVATIVE"


def test_authentication_failure_terminates_without_a_payment_decision(tmp_path):
    from tests.paypal_fake import err
    r = Rig(tmp_path)
    r.fake.token_responses = [err(401, "invalid_client")] * 3
    r.client.tokens._send = lambda *a, **k: (_ for _ in ()).throw(__import__("paypal.models", fromlist=["PayPalApiError"]).PayPalApiError(401, "invalid_client", "bad creds"))
    res = r.ex.execute(REQ)
    assert res.status == ExecStatus.TERMINATED and res.reason.startswith("not_a_payment_failure")
    assert r.outcomes() == ["NOT_COMPARABLE"] and r.fake.count("POST", "/v2/checkout/orders") == 0


# ============================================================ order state handling and verification
def test_voided_or_unknown_order_states_terminate_without_capture(tmp_path):
    for status, why in (("VOIDED", "order_voided"), ("WEIRD_NEW_STATUS", "unrecognised_order_status:WEIRD_NEW_STATUS")):
        r = Rig(tmp_path / status)
        res = r.ex.execute(REQ)
        r.fake.set_status(res.order_id, status)
        out = r.ex.resume(REQ.reference_id)
        assert out.status == ExecStatus.TERMINATED and out.reason == why and r.fake.count("POST", "/capture") == 0


def test_verification_failure_is_recorded_and_not_recaptured(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.bad_amount = True  # captured amount differs from the request
    out = r.ex.resume(REQ.reference_id)
    assert out.status == ExecStatus.TERMINATED and out.reason == "capture_verification_failed" and r.fake.count("POST", "/capture") == 1
    assert r.records()[-1]["validation_outcome"] == "DIVERGED" and "amount" in json.loads(r.records()[-1]["predicted_vs_observed_json"])["observed"]


# ============================================================ policy gating and action support
def test_only_selected_feasible_valid_policies_can_be_executed():
    with pytest.raises(PolicyNotApproved):
        ExecutionPolicy.from_definition(INTELLIGENT, feasible=False)
    with pytest.raises(PolicyNotApproved):
        ExecutionPolicy.from_definition({**INTELLIGENT, "retry_limit": 99}, feasible=True)
    report = {"execution_id": "agent-1", "selected_policy": {"name": "intel", "definition": INTELLIGENT},
              "final_evaluation": {"ranking": [{"policy": "intel", "feasible": True}, {"policy": "x", "feasible": False}]}}
    p = ExecutionPolicy.from_decision_report(report)
    assert p.name == "intel" and p.source == "agent_report:agent-1" and len(p.policy_id) == 16
    assert p.policy_id == ExecutionPolicy.from_definition(INTELLIGENT, feasible=True).policy_id  # stable id
    assert p.policy_id != ExecutionPolicy.from_definition({**INTELLIGENT, "retry_limit": 1}, feasible=True).policy_id
    with pytest.raises(PolicyNotApproved):
        ExecutionPolicy.from_decision_report({**report, "selected_policy": None})
    with pytest.raises(PolicyNotApproved):
        ExecutionPolicy.from_decision_report({**report, "final_evaluation": {"ranking": [{"policy": "intel", "feasible": False}]}})


def test_action_support_table_identifies_what_paypal_can_execute():
    by = {a["simulated_action"]: a for a in ACTION_SUPPORT}
    assert any(a["executable_via_paypal"] is True and "retry" in a["simulated_action"].lower() for a in ACTION_SUPPORT)
    buyer = next(a for a in ACTION_SUPPORT if "buyer action" in a["simulated_action"].lower())
    assert buyer["executable_via_paypal"] == "PARTIAL" and "NOT performed" in buyer["limits"]
    refused = [a for a in ACTION_SUPPORT if a["executable_via_paypal"] is False]
    assert {"Blind automatic retry of buyer-action or terminal failures", "Re-capture without first checking order state"} <= {a["simulated_action"] for a in refused}
    assert by["Flag terminal failure for manual review"]["executable_via_paypal"] is False


# ============================================================ security of what gets persisted + sim/sandbox separation
def test_no_secret_or_token_in_the_sandbox_database_after_a_full_run(tmp_path):
    r = Rig(tmp_path)
    res = r.ex.execute(REQ)
    r.fake.approve(res.order_id)
    r.fake.capture_script = [("error", 500, "INTERNAL_SERVER_ERROR", None), "ok"]
    r.ex.resume(REQ.reference_id)
    r.store.conn.commit()
    raw = open(tmp_path / "sandbox.db", "rb").read()
    for secret in (FAKE_CLIENT_SECRET, FAKE_TOKEN, FAKE_CLIENT_ID):
        assert secret.encode() not in raw
    assert r.store.validation_summary()["environment"] == "PAYPAL_SANDBOX"
    tables = {x[0] for x in r.store.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert not any("sim" in t for t in tables)
