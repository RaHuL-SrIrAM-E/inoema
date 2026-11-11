"""Shared fixtures for the dashboard tests.

Data provenance of the fixture (so no test result is mistaken for a live one):
  * simulation + agent DB: REAL MCP server + REAL simulation engine, driven by a SCRIPTED test-double LLM (provider SCRIPTED_TEST_DOUBLE);
  * one extra agent row is a COPY of that run relabelled provider='GEMINI_API' -- a UI-logic fixture to exercise demonstration-mode filtering.
    It is NOT a genuine Gemini execution;
  * sandbox DB: produced by the real executor against the MOCK HTTP double in tests/paypal_fake.py -- NOT PayPal.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import uuid

import pytest

from agent.payments.agent import PaymentAgent
from agent.payments.models import AgentConfig
from agent.payments.testing import ScriptedBackend
from paypal.client import PayPalClient
from paypal.execution import DEMO_POLICY, ExecutionPolicy, PaymentRequest, RecoveryExecutor
from paypal.models import PayPalConfig
from paypal.persistence import SandboxStore
from tests.paypal_fake import FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FakePayPal
from tests.test_payment_agent import OBJECTIVE, _script

FAKE_GEMINI_KEY = "AIzaFAKE-DASHBOARD-SECRET-KEY-0001"


def build_world(tmp):
    sim, agent, sbx = str(tmp / "sim.db"), str(tmp / "agent.db"), str(tmp / "sandbox.db")
    cfg = AgentConfig(api_key=FAKE_GEMINI_KEY, exploration_trials=3, final_trials=4, robustness_n_transactions=500, robustness_trials=3,
                      db_path=sim, agent_db_path=agent)
    report = asyncio.run(PaymentAgent(cfg, backend=ScriptedBackend(_script())).run(OBJECTIVE))
    scripted_id = report.execution_id

    # relabelled copy -> 'GEMINI_API' fixture row (UI-logic fixture only)
    conn = sqlite3.connect(agent)
    new_id = str(uuid.uuid4())
    conn.row_factory = sqlite3.Row
    run = dict(conn.execute("SELECT * FROM payment_agent_runs WHERE execution_id=?", (scripted_id,)).fetchone())
    rep = json.loads(run["final_report_json"])
    rep["execution_id"], rep["llm"] = new_id, {"provider": "GEMINI_API", "model": "fixture-model"}
    run.update(execution_id=new_id, llm_provider="GEMINI_API", model="fixture-model", final_report_json=json.dumps(rep), started_at="2099-01-01T00:00:00+00:00")
    conn.execute("INSERT INTO payment_agent_runs (%s) VALUES (%s)" % (",".join(run), ",".join("?" * len(run))), tuple(run.values()))
    for table, cols in (("payment_agent_tool_calls", "seq,iteration,source,tool,args_json,ok,result_json,duration_ms,work_units,created_at"),
                        ("payment_agent_events", "seq,kind,payload_json,created_at"), ("payment_agent_candidates", "name,origin,definition_json,observations_json")):
        for r in conn.execute(f"SELECT {cols} FROM {table} WHERE execution_id=?", (scripted_id,)).fetchall():
            conn.execute(f"INSERT INTO {table} (execution_id,{cols}) VALUES (?,{','.join('?' * len(cols.split(',')))})", (new_id, *tuple(r)))
    conn.commit()
    conn.close()

    # sandbox evidence from the real executor against the MOCK http double
    fake = FakePayPal()
    pcfg = PayPalConfig(client_id=FAKE_CLIENT_ID, client_secret=FAKE_CLIENT_SECRET)
    store = SandboxStore(sbx, secrets=pcfg.secrets())
    ex = RecoveryExecutor(PayPalClient(pcfg, sink=store, transport=fake.transport()), store,
                          ExecutionPolicy.from_definition(DEMO_POLICY, feasible=True), "fixture-exp", sleep=lambda s: None)
    r1 = ex.execute(PaymentRequest("25.00", "USD", "fixture-done"))
    fake.approve(r1.order_id)
    ex.resume("fixture-done")
    ex.execute(PaymentRequest("30.00", "USD", "fixture-waiting"))
    store.conn.commit()
    return {"sim": sim, "agent": agent, "sandbox": sbx, "scripted_id": scripted_id, "gemini_fixture_id": new_id, "report": report.to_dict(),
            "experiment_id": report.to_dict()["execution_id"] and json.loads(run["objective_json"]) and None}


def variant_agent_db(src: str, dst: str, *, interrupt_genuine: bool = False, add_newer_interrupted: bool = False) -> dict:
    """Copy the agent DB and reshape its GENUINE-labelled rows (test fixtures only, see module docstring):
       interrupt_genuine      -> every GEMINI_API row becomes an interrupted run (termination llm_error), so NO complete genuine run exists
       add_newer_interrupted  -> add a NEWER interrupted GEMINI_API row, leaving the older complete one in place"""
    import shutil
    shutil.copy(src, dst)
    conn = sqlite3.connect(dst)
    conn.row_factory = sqlite3.Row
    out: dict = {}
    if interrupt_genuine:
        for r in conn.execute("SELECT execution_id, final_report_json FROM payment_agent_runs WHERE llm_provider='GEMINI_API'").fetchall():
            rep = json.loads(r["final_report_json"])
            rep["termination_reason"] = "llm_error"
            conn.execute("UPDATE payment_agent_runs SET termination_reason='llm_error', final_report_json=? WHERE execution_id=?", (json.dumps(rep), r["execution_id"]))
    if add_newer_interrupted:
        base = dict(conn.execute("SELECT * FROM payment_agent_runs WHERE llm_provider='GEMINI_API'").fetchone())
        rep = json.loads(base["final_report_json"])
        new_id = str(uuid.uuid4())
        rep.update(execution_id=new_id, termination_reason="llm_error")
        base.update(execution_id=new_id, started_at="2100-01-01T00:00:00+00:00", termination_reason="llm_error", model="fixture-model-newer", final_report_json=json.dumps(rep))
        conn.execute("INSERT INTO payment_agent_runs (%s) VALUES (%s)" % (",".join(base), ",".join("?" * len(base))), tuple(base.values()))
        out["newer_interrupted_id"] = new_id
    conn.commit()
    conn.close()
    return out
