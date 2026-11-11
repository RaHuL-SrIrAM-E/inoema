"""OFFLINE tests for the negative-testing harness and coverage honesty. HTTP is a MOCK (tests/paypal_fake.py), NOT PayPal:
these prove the harness records only what is returned and never reports an unattempted scenario as verified."""
import json

from engine.payments.models import FailureCategory
from paypal.client import PayPalClient
from paypal.models import PayPalConfig
from paypal.negative_testing import BASELINE_CODE, CANDIDATE_MOCK_CODES, SIMULATION_ONLY, coverage_report, run_negative_tests
from paypal.orders import OrdersApi
from paypal.persistence import SandboxStore
from tests.paypal_fake import FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FakePayPal

CFG = PayPalConfig(client_id=FAKE_CLIENT_ID, client_secret=FAKE_CLIENT_SECRET)


def rig(tmp_path, supported=()):
    fake = FakePayPal()
    fake.supported_mock_codes = set(supported)
    store = SandboxStore(str(tmp_path / "s.db"))
    return fake, store, OrdersApi(PayPalClient(CFG, sink=store, transport=fake.transport()))


def by_cat(report):
    return {c["category"]: c for c in report}


def test_with_no_attempts_nothing_is_reported_as_verified(tmp_path):
    _, store, _ = rig(tmp_path)
    rep = by_cat(coverage_report(store))
    assert set(rep) == {c.name for c in FailureCategory}  # every simulation category is accounted for
    assert {c: r["status"] for c, r in rep.items() if r["status"] != "SIMULATION_ONLY"} == {
        c: "NOT_ATTEMPTED" for c in rep if c not in SIMULATION_ONLY}
    assert rep["TRANSIENT_NETWORK_ERROR"]["status"] == "SIMULATION_ONLY" and rep["EXPIRED_PAYMENT_METHOD"]["status"] == "SIMULATION_ONLY"
    assert all(r["status"] != "REPRODUCED" for r in rep.values())


def test_only_what_is_actually_returned_counts_as_reproduced(tmp_path):
    fake, store, orders = rig(tmp_path, supported={"INSTRUMENT_DECLINED", "TRANSACTION_REFUSED"})
    results = run_negative_tests(orders, store)
    by_code = {r["mock_code"]: r for r in results}
    assert len(results) == 1 + len(CANDIDATE_MOCK_CODES) and BASELINE_CODE in by_code
    assert by_code["INSTRUMENT_DECLINED"]["reproduced"] is True and by_code["TRANSACTION_REFUSED"]["reproduced"] is True
    assert by_code["PAYER_CANNOT_PAY"]["reproduced"] is False  # the mock ignored the header -> real unapproved-capture error returned
    assert by_code["PAYER_CANNOT_PAY"]["observed_issues"] == ["ORDER_NOT_APPROVED"] and by_code["PAYER_CANNOT_PAY"]["observed_status"] == 422
    assert by_code[BASELINE_CODE]["detail"]["baseline"] is True
    assert len(store.negative_tests()) == len(results)  # persisted

    rep = by_cat(coverage_report(store))
    assert rep["INSTRUMENT_DECLINED"]["status"] == "REPRODUCED" and rep["TERMINAL_PAYMENT_FAILURE"]["status"] == "REPRODUCED"
    assert rep["VALIDATION_ERROR"]["status"] == "ATTEMPTED_NOT_REPRODUCED"  # DUPLICATE_INVOICE_ID attempted, not honoured
    assert rep["TEMPORARY_PROCESSING_ERROR"]["status"] == "ATTEMPTED_NOT_REPRODUCED"
    assert rep["ORDER_NOT_APPROVED"]["status"] == "REPRODUCED"  # the REAL unapproved capture returned it (via baseline)
    assert rep["TRANSIENT_NETWORK_ERROR"]["status"] == "SIMULATION_ONLY"  # never claimed, even if a code were observed


def test_every_attempt_leaves_redacted_api_evidence_with_debug_ids(tmp_path):
    fake, store, orders = rig(tmp_path, supported={"INSTRUMENT_DECLINED"})
    run_negative_tests(orders, store, codes=["INSTRUMENT_DECLINED"])
    caps = [c for c in store.api_calls() if c["operation"] == "CAPTURE_ORDER"]
    assert len(caps) == 2 and all(c["status"] == 422 and c["paypal_debug_id"] == "mock-debug-id" for c in caps)
    mocked = next(c for c in caps if "PayPal-Mock-Response" in c["request_headers"])
    assert json.loads(json.loads(mocked["request_headers"])["PayPal-Mock-Response"]) == {"mock_application_codes": "INSTRUMENT_DECLINED"}
    assert FAKE_CLIENT_SECRET not in json.dumps(store.api_calls())
