"""OFFLINE tests for the policy execution bridge. Persisted results here are SYNTHETIC fixtures written to a temp DB (clearly not genuine Gemini
output); the PayPal adapter is only ever an injected fake callable. No network, no Gemini, no PayPal, no existing database is touched."""
import copy
import json
import sqlite3

import pytest

from engine.payments.models import FailureCategory as FC
from execution_bridge.executor import (REAL_SANDBOX_READONLY, SIMULATED_STATE, Action, FailureEvent, PaymentState, PolicyExecutor)
from execution_bridge.paypal_gate import REAL_SANDBOX, apply_to_paypal, paypal_support
from execution_bridge.plan import (PLAN_VERSION, LoadedResult, PlanRefused, ResultNotFound, compile_plan, load_persisted_result, validate_result)
from execution_bridge.scenarios import SCENARIOS
from execution_bridge.store import BridgeStore
from storage.payment_agent_runs import AgentRunStore

DEF = {"name": "sel", "retry_limit": 2, "retry_intervals": [5.0], "exponential_backoff": True, "backoff_base_minutes": 45.0, "backoff_multiplier": 2.0,
       "backoff_max_interval_minutes": 240.0, "maximum_recovery_window": 4320.0, "buyer_action_timeout": 2880.0, "request_buyer_action": True,
       "alternate_funding_action": "REQUEST_ALTERNATE", "terminal_failure_handling": "FLAG_FOR_REVIEW", "classification_aware": True, "idempotent_retries": True}
CONS = {"max_retries": 2, "max_recovery_window_minutes": 4320.0, "forbid_terminal_retries": True, "require_idempotency": True, "require_buyer_approval": True}


def report(**over):
    r = {"execution_id": "run-1", "selected_policy": {"name": "sel", "definition": copy.deepcopy(DEF)},
         "final_evaluation": {"ranking": [{"policy": "sel", "feasible": True}]}, "confidence": {"label": "HIGH"}, "merchant_objective": {"constraints": dict(CONS)}}
    r.update(over)
    return r


def result(rep=None, provider="GEMINI_API", status="COMPLETED", term="finished_by_llm", eid="run-1"):
    return LoadedResult(eid, provider, status, term, report() if rep is None else rep, "x")


def plan(**kw):
    return compile_plan(result(), **kw)


def write_db(path, rows):
    s = AgentRunStore(str(path))
    for eid, provider, status, term, rep, started in rows:
        s.conn.execute("INSERT INTO payment_agent_runs (execution_id,started_at,finished_at,status,llm_provider,model,config_json,objective_text,objective_json,"
                       "experiment_id,termination_reason,final_report_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                       (eid, started, started, status, provider, "m", "{}", "o", "{}", None, term, json.dumps(rep) if rep is not None else None))
    s.conn.commit()
    s.conn.close()


# ---------------------------------------------------------------- loading / gating
def test_missing_database_and_missing_run_are_refused(tmp_path):
    with pytest.raises(ResultNotFound):
        load_persisted_result(db_path=str(tmp_path / "nope.db"))
    write_db(tmp_path / "a.db", [])
    with pytest.raises(ResultNotFound):
        load_persisted_result(db_path=str(tmp_path / "a.db"))
    write_db(tmp_path / "b.db", [("r", "GEMINI_API", "COMPLETED", "finished_by_llm", report(), "2026-01-01")])
    with pytest.raises(ResultNotFound):
        load_persisted_result("other-id", str(tmp_path / "b.db"))


def test_load_prefers_the_latest_complete_genuine_run_and_is_read_only(tmp_path):
    p = tmp_path / "a.db"
    write_db(p, [("old-complete", "GEMINI_API", "COMPLETED", "finished_by_llm", report(), "2026-01-01"),
                 ("new-interrupted", "GEMINI_API", "COMPLETED", "llm_error", report(), "2026-02-01"),
                 ("scripted", "SCRIPTED_TEST", "COMPLETED", "finished_by_llm", report(), "2026-03-01")])
    before = p.read_bytes()
    r = load_persisted_result(db_path=str(p))
    assert r.execution_id == "old-complete" and r.is_complete
    assert p.read_bytes() == before


def test_interrupted_or_scripted_runs_are_not_complete(tmp_path):
    p = tmp_path / "a.db"
    write_db(p, [("i", "GEMINI_API", "COMPLETED", "llm_error", report(), "2026-01-01"), ("s", "SCRIPTED_TEST", "COMPLETED", "finished_by_llm", report(), "2026-01-02")])
    assert not load_persisted_result("i", str(p)).is_complete and not load_persisted_result("s", str(p)).is_complete


def test_good_result_compiles_to_a_versioned_plan():
    pl = plan()
    assert pl.plan_version == PLAN_VERSION and pl.status == "APPROVED" and pl.source_run_id == "run-1" and pl.policy_name == "sel"
    assert pl.retry_limit == 2 and pl.retry_delays_minutes == (45.0, 90.0) and pl.buyer_action_required and pl.buyer_action_timeout_minutes == 2880.0
    assert pl.failure_rules["TERMINAL_PAYMENT_FAILURE"]["first_action"] == "STOP" and pl.failure_rules["TRANSIENT_NETWORK_ERROR"]["first_action"] == "RETRY"
    assert pl.failure_rules["INSTRUMENT_DECLINED"]["first_action"] == "REQUIRE_BUYER_ACTION"
    assert plan().plan_id == pl.plan_id  # deterministic


def test_missing_report_is_refused():
    with pytest.raises(PlanRefused, match="report"):
        compile_plan(LoadedResult("r", "GEMINI_API", "COMPLETED", "finished_by_llm", None, "x"))


@pytest.mark.parametrize("kw,needle", [(dict(status="FAILED"), "not COMPLETE"), (dict(term="llm_error"), "not COMPLETE"), (dict(provider="SCRIPTED_TEST"), "not COMPLETE")])
def test_incomplete_source_is_refused(kw, needle):
    with pytest.raises(PlanRefused, match=needle):
        compile_plan(result(**kw))


@pytest.mark.parametrize("label", ["LOW", "INSUFFICIENT", None])
def test_unacceptable_confidence_is_refused(label):
    with pytest.raises(PlanRefused, match="confidence"):
        compile_plan(result(report(confidence={"label": label})))


def test_no_selected_policy_and_infeasible_selected_policy_are_refused():
    with pytest.raises(PlanRefused, match="selected_policy_present"):
        compile_plan(result(report(selected_policy=None)))
    with pytest.raises(PlanRefused, match="not feasible"):
        compile_plan(result(report(final_evaluation={"ranking": [{"policy": "sel", "feasible": False}]})))


def test_unsupported_constraint_is_refused_and_cannot_be_overridden():
    cons = {**CONS, "max_daily_spend": 5}
    with pytest.raises(PlanRefused, match="unsupported constraint"):
        compile_plan(result(report(merchant_objective={"constraints": cons})), allow_incomplete_source=True)
    cons2 = {k: v for k, v in CONS.items() if k != "max_retries"}
    with pytest.raises(PlanRefused, match="missing"):
        compile_plan(result(report(merchant_objective={"constraints": cons2})))


@pytest.mark.parametrize("field,value,needle", [("retry_limit", 3, "retry_limit"), ("maximum_recovery_window", 9999.0, "window"),
                                                ("classification_aware", False, "terminal retries"), ("idempotent_retries", False, "idempotent")])
def test_policy_violating_a_constraint_is_refused(field, value, needle):
    d = {**DEF, field: value}
    if field == "classification_aware":
        d.update(request_buyer_action=False, alternate_funding_action="NONE")
    with pytest.raises(PlanRefused, match=needle):
        compile_plan(result(report(selected_policy={"name": "sel", "definition": d})), allow_incomplete_source=True)


def test_override_waives_only_completeness_and_confidence_and_is_labelled():
    r = result(report(confidence={"label": "LOW"}), term="llm_error")
    with pytest.raises(PlanRefused):
        compile_plan(r)
    pl = compile_plan(r, allow_incomplete_source=True)
    assert pl.status == "OVERRIDDEN_INCOMPLETE_SOURCE" and len(pl.waived_checks) == 2 and all("OVERRIDE" in w for w in pl.warnings)
    assert pl.source_run_state.startswith("INTERRUPTED") and pl.plan_id != plan().plan_id
    infeasible = result(report(final_evaluation={"ranking": [{"policy": "sel", "feasible": False}]}), term="llm_error")
    with pytest.raises(PlanRefused):
        compile_plan(infeasible, allow_incomplete_source=True)  # feasibility is never waivable
    assert not validate_result(infeasible, allow_incomplete_source=True).ok


# ---------------------------------------------------------------- executor: each action
def S(**kw):
    base = dict(payment_id="p", order_status="APPROVED", idempotency_key="k")
    base.update(kw)
    return PaymentState(**base)


def test_retry_with_policy_delay_and_budget():
    ex = PolicyExecutor(plan())
    d = ex.decide(S(), FailureEvent(FC.TRANSIENT_NETWORK_ERROR))
    assert (d.action, d.delay_minutes, d.rule) == (Action.RETRY, 45.0, "SCHEDULED_RETRY")
    d2 = ex.decide(S(retries_made=1, elapsed_minutes=45), FailureEvent(FC.TEMPORARY_PROCESSING_ERROR))
    assert (d2.action, d2.delay_minutes) == (Action.RETRY, 90.0)
    d3 = ex.decide(S(retries_made=2, elapsed_minutes=135), FailureEvent(FC.TRANSIENT_NETWORK_ERROR))
    assert (d3.action, d3.rule) == (Action.STOP, "RETRY_BUDGET_EXHAUSTED")


def test_wait_for_buyer_approval_and_pending_buyer_action():
    ex = PolicyExecutor(plan())
    assert ex.decide(S(order_status="CREATED")).action == Action.WAIT
    assert ex.decide(S(order_status="PAYER_ACTION_REQUIRED")).rule == "AWAITING_BUYER_APPROVAL"
    pending = S(order_status="PAYER_ACTION_REQUIRED", elapsed_minutes=600, buyer_action_used=True, awaiting_buyer_since=10.0)
    assert ex.decide(pending, FailureEvent(FC.INSTRUMENT_DECLINED)).action == Action.WAIT


def test_require_buyer_action_then_timeout_stop():
    ex = PolicyExecutor(plan())
    assert ex.decide(S(), FailureEvent(FC.INSTRUMENT_DECLINED)).action == Action.REQUIRE_BUYER_ACTION
    late = S(elapsed_minutes=3000, buyer_action_used=True, awaiting_buyer_since=10.0)
    d = ex.decide(late, FailureEvent(FC.INSTRUMENT_DECLINED))
    assert (d.action, d.rule) == (Action.STOP, "BUYER_ACTION_TIMEOUT")
    assert ex.decide(S(buyer_action_used=True), FailureEvent(FC.INSTRUMENT_DECLINED)).action == Action.STOP  # buyer is asked once


def test_stop_cases_terminal_window_and_flag_for_review():
    ex = PolicyExecutor(plan())
    d = ex.decide(S(), FailureEvent(FC.TERMINAL_PAYMENT_FAILURE))
    assert (d.action, d.rule, d.flag_for_review) == (Action.STOP, "TERMINAL_FAILURE_NO_RETRY", True)
    assert ex.decide(S(retries_made=1, elapsed_minutes=4300), FailureEvent(FC.TRANSIENT_NETWORK_ERROR)).rule == "RECOVERY_WINDOW_EXCEEDED"
    assert ex.decide(S(order_status="VOIDED"), FailureEvent(FC.TRANSIENT_NETWORK_ERROR)).rule == "ORDER_VOIDED"
    assert ex.decide(S(), None).rule == "NO_FAILURE_TO_RECOVER"


# ---------------------------------------------------------------- safety violations
def test_retry_without_idempotency_key_is_blocked():
    d = PolicyExecutor(plan()).decide(S(idempotency_key=None), FailureEvent(FC.TRANSIENT_NETWORK_ERROR))
    assert (d.action, d.rule) == (Action.STOP, "RETRY_BLOCKED_NO_IDEMPOTENCY_KEY")


@pytest.mark.parametrize("status,cap", [("COMPLETED", None), ("APPROVED", "CAP123")])
@pytest.mark.parametrize("cat", list(FC))
def test_completed_or_captured_payment_is_never_retried_for_any_event(status, cap, cat):
    d = PolicyExecutor(plan()).decide(S(order_status=status, capture_id=cap), FailureEvent(cat))
    assert (d.action, d.rule) == (Action.STOP, "NEVER_RECAPTURE")


def test_executor_enforces_forbid_terminal_retries_even_for_a_blind_policy():
    """Compile-time gating refuses blind policies; this proves the runtime check independently by hand-building a permissive plan."""
    from dataclasses import replace
    blind = {**DEF, "classification_aware": False, "request_buyer_action": False, "alternate_funding_action": "NONE"}
    pl = plan()
    pl = replace(pl, policy_definition=blind, constraints={**pl.constraints})
    d = PolicyExecutor(pl).decide(S(), FailureEvent(FC.TERMINAL_PAYMENT_FAILURE))
    assert d.action == Action.STOP and d.rule == "TERMINAL_FAILURE_NO_RETRY"


def test_retry_budget_cannot_exceed_the_objective_constraint():
    from dataclasses import replace
    pl = plan()
    tight = replace(pl, constraints={**pl.constraints, "max_retries": 1})
    d = PolicyExecutor(tight).decide(S(retries_made=1, elapsed_minutes=45), FailureEvent(FC.TRANSIENT_NETWORK_ERROR))
    assert (d.action, d.rule) == (Action.STOP, "RETRY_BUDGET_EXHAUSTED")


def test_executor_is_deterministic_and_trace_carries_provenance():
    runs = []
    for _ in range(2):
        ex = PolicyExecutor(plan())
        for _, st_, ev in SCENARIOS:
            ex.step(st_, ev)
        runs.append([s.__dict__ for s in ex.trace])
    assert runs[0] == runs[1] and len(runs[0]) == len(SCENARIOS)
    pl = plan()
    for s in runs[0]:
        assert (s["plan_id"], s["plan_version"], s["source_run_id"], s["policy_name"], s["policy_hash"]) == (pl.plan_id, PLAN_VERSION, "run-1", "sel", pl.policy_hash)
        assert s["rationale"] and s["rule"] and s["state_source"] == SIMULATED_STATE
    assert {s["action"] for s in runs[0]} == {"RETRY", "WAIT", "REQUIRE_BUYER_ACTION", "STOP"}  # every action is demonstrated


# ---------------------------------------------------------------- PayPal gate
class Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return "resumed"


def real(**kw):
    base = dict(payment_id="real", order_status="APPROVED", idempotency_key="k", source=REAL_SANDBOX)
    base.update(kw)
    return PaymentState(**base)


def test_simulated_and_readonly_states_never_reach_paypal():
    ex, spy = PolicyExecutor(plan()), Spy()
    for src in (SIMULATED_STATE, REAL_SANDBOX_READONLY):
        st_ = S(source=src)
        d = ex.decide(st_, FailureEvent(FC.TRANSIENT_NETWORK_ERROR))
        a = apply_to_paypal(d, st_, resume=spy, confirm=True)
        assert not a.performed and not paypal_support(d, st_).supported
    assert spy.calls == []


def test_completed_real_payment_is_never_retried_or_recaptured():
    ex, s = PolicyExecutor(plan()), Spy()
    st_ = real(order_status="COMPLETED", capture_id="CAP1")
    d = ex.decide(st_, FailureEvent(FC.TRANSIENT_NETWORK_ERROR))
    assert d.action == Action.STOP and not apply_to_paypal(d, st_, resume=s, confirm=True).performed and s.calls == []


def test_real_retry_requires_approved_key_confirm_and_injected_resume():
    ex, spy = PolicyExecutor(plan()), Spy()
    ev = FailureEvent(FC.TRANSIENT_NETWORK_ERROR)
    d = ex.decide(real(), ev)
    assert d.action == Action.RETRY
    assert not apply_to_paypal(d, real(), resume=spy, confirm=False).performed  # no confirm
    assert not apply_to_paypal(d, real(), resume=None, confirm=True).performed  # nothing injected
    assert not apply_to_paypal(d, real(order_status="CREATED"), resume=spy, confirm=True).performed  # not approved
    assert not apply_to_paypal(d, real(idempotency_key=None), resume=spy, confirm=True).performed  # no key
    assert spy.calls == []
    a = apply_to_paypal(d, real(), resume=spy, confirm=True)
    assert a.performed and spy.calls == [{"confirm_capture": True}]


def test_wait_and_buyer_action_are_read_only_and_stop_sends_nothing():
    ex, spy = PolicyExecutor(plan()), Spy()
    for st_, ev in ((real(order_status="CREATED"), None), (real(), FailureEvent(FC.INSTRUMENT_DECLINED)), (real(), FailureEvent(FC.TERMINAL_PAYMENT_FAILURE))):
        d = ex.decide(st_, ev)
        a = apply_to_paypal(d, st_, resume=spy, confirm=True)
        assert not a.performed
    assert spy.calls == []


def test_bridge_modules_import_no_network_or_llm_libraries():
    import ast
    import execution_bridge.executor as e, execution_bridge.paypal_gate as g, execution_bridge.plan as p, execution_bridge.scenarios as sc, execution_bridge.store as st_
    banned = {"httpx", "requests", "urllib3", "google", "fastmcp", "mcp", "paypal.client", "paypal.orders", "paypal.resume", "agent"}
    for m in (e, g, p, sc, st_):
        names = set()
        for n in ast.walk(ast.parse(open(m.__file__).read())):
            if isinstance(n, ast.Import):
                names |= {a.name for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                names.add(n.module or "")
        assert not {x for x in names if x.split(".")[0] in banned or x in banned}, (m.__name__, names)


# ---------------------------------------------------------------- persistence
def test_trace_persistence_is_separate_and_complete(tmp_path):
    pl, ex = plan(), PolicyExecutor(plan())
    for _, st_, ev in SCENARIOS:
        ex.step(st_, ev)
    store = BridgeStore(str(tmp_path / "bridge.db"))
    assert store.save(pl, "t", ex.trace) == len(SCENARIOS)
    rows = store.decisions(pl.plan_id)
    step = json.loads(rows[0]["step_json"])
    assert step["source_run_id"] == "run-1" and step["policy_hash"] == pl.policy_hash and step["rationale"]
    assert "sandbox" not in {r[0] for r in sqlite3.connect(str(tmp_path / "bridge.db")).execute("select name from sqlite_master")}
