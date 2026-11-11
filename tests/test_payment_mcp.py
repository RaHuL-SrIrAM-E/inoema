"""MCP tool tests: a real FastMCP server subprocess, driven by an independent client,
against a temporary experiment database. Every assertion compares against direct engine calls
or exact expectations -- tools are not mocked."""
import asyncio
import os
import sys

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from engine.objectives import HardConstraints, ObjectiveWeights, evaluate_policy
from engine.payments.recovery_policy import exponential_backoff, intelligent_recovery
from engine.payments.scenarios import build_scenario
from engine.payments.simulator import compare_policies_monte_carlo

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAYMENT_TOOLS = {"create_payment_experiment", "generate_payment_scenarios", "run_payment_simulation",
                 "compare_recovery_policies", "get_payment_experiment_results", "get_payment_transaction_trace"}


def _client(tmp_path):
    t = StdioTransport(command=sys.executable, args=["-m", "spikes.mcp_sim.server"], cwd=ROOT,
                       env={**os.environ, "INOEMA_PAYMENT_DB": str(tmp_path / "sim.db")})
    return Client(t, timeout=180)


def _run(coro):
    return asyncio.run(coro)


def test_end_to_end_tools_match_direct_engine_and_persist(tmp_path):
    N, TRIALS, SEED = 2000, 4, 5

    async def go():
        async with _client(tmp_path) as c:
            names = {t.name for t in await c.list_tools()}
            exp = (await c.call_tool("create_payment_experiment", {"name": "e2e"})).structured_content
            eid = exp["experiment_id"]
            sc = (await c.call_tool("generate_payment_scenarios", {
                "experiment_id": eid, "scenarios": [{"preset": "baseline", "n_transactions": N}]})).structured_content
            sid = sc["scenarios"][0]["scenario_id"]
            pols = [{"base": "exponential_backoff"}, {"base": "intelligent_recovery"}]
            cmp1 = (await c.call_tool("compare_recovery_policies", {
                "experiment_id": eid, "scenario_id": sid, "policies": pols, "n_trials": TRIALS, "master_seed": SEED})).structured_content
            cmp2 = (await c.call_tool("compare_recovery_policies", {
                "experiment_id": eid, "scenario_id": sid, "policies": pols, "n_trials": TRIALS, "master_seed": SEED})).structured_content
            single = (await c.call_tool("run_payment_simulation", {
                "experiment_id": eid, "scenario_id": sid, "policy": {"base": "intelligent_recovery"},
                "n_trials": TRIALS, "master_seed": SEED})).structured_content
            trace = (await c.call_tool("get_payment_transaction_trace", {
                "experiment_id": eid, "scenario_id": sid, "policy": {"base": "intelligent_recovery"},
                "master_seed": SEED, "trial_index": 0, "outcome_filter": "RECOVERED"})).structured_content
            stored = (await c.call_tool("get_payment_experiment_results", {"experiment_id": eid})).structured_content
            return names, sc, cmp1, cmp2, single, trace, stored

    names, sc, cmp1, cmp2, single, trace, stored = _run(go())
    assert PAYMENT_TOOLS <= names and "simulate_release_schedule" in names  # legacy tool preserved

    # Same numbers as calling the engine directly.
    scenario = build_scenario("baseline", N, "baseline")
    direct = compare_policies_monte_carlo(scenario, [exponential_backoff(), intelligent_recovery()], TRIALS, SEED)
    for name in ("exponential_backoff", "intelligent_recovery"):
        d = direct[name].aggregate.pooled["recovery_rate"]["value"]
        assert cmp1["policies"][name]["summary"]["recovery_rate"]["value"] == pytest.approx(d)
    ev = evaluate_policy(direct["intelligent_recovery"], scenario.rules, ObjectiveWeights(), HardConstraints())
    assert single["evaluation"]["score_mean"] == pytest.approx(ev.score_mean) and single["evaluation"]["feasible"] is True

    # Ranking: hard constraints make the blind policy infeasible; deterministic and reproducible.
    assert [r["policy"] for r in cmp1["ranking"]] == ["intelligent_recovery", "exponential_backoff"]
    assert [r["feasible"] for r in cmp1["ranking"]] == [True, False]
    assert cmp1["ranking"] == cmp2["ranking"] and cmp1["policies"] == cmp2["policies"]
    assert cmp1["provenance"] == "SIMULATED" and "NOT_PAYPAL" in cmp1["label"]
    assert cmp1["paired_score_difference_vs_best"]["exponential_backoff"]["ci_excludes_zero"] is True

    # Trace is a genuine event sequence for a recovered transaction.
    assert trace["transaction"]["recovery_status"] == "RECOVERED"
    types = [e["type"] for e in trace["events"]]
    assert types[0] == "TRANSACTION_CREATED" and types[-1] == "PAYMENT_RECOVERED" and "PAYMENT_FAILED" in types

    # Persistence: 3 stored runs (2 comparisons + 1 run), all marked SIMULATED.
    assert [r["kind"] for r in stored["runs"]] == ["comparison", "comparison", "run"]
    assert stored["scenarios"][0]["spec"] == {"preset": "baseline", "n_transactions": N}
    assert os.path.exists(tmp_path / "sim.db")


def test_invalid_inputs_are_rejected(tmp_path):
    async def attempt(c, tool, args):
        with pytest.raises(Exception) as ei:
            await c.call_tool(tool, args)
        return str(ei.value)

    async def go():
        async with _client(tmp_path) as c:
            eid = (await c.call_tool("create_payment_experiment", {"name": "bad"})).structured_content["experiment_id"]
            sid = (await c.call_tool("generate_payment_scenarios", {
                "experiment_id": eid, "scenarios": [{"preset": "baseline", "n_transactions": 500}]})).structured_content["scenarios"][0]["scenario_id"]
            ok = {"experiment_id": eid, "scenario_id": sid, "policy": {"base": "intelligent_recovery"}}
            errs = {}
            errs["unknown_exp"] = await attempt(c, "run_payment_simulation", {**ok, "experiment_id": "nope"})
            errs["unknown_scn"] = await attempt(c, "run_payment_simulation", {**ok, "scenario_id": "nope"})
            errs["retry_limit"] = await attempt(c, "run_payment_simulation", {**ok, "policy": {"base": "exponential_backoff", "retry_limit": 99}})
            errs["trials_zero"] = await attempt(c, "run_payment_simulation", {**ok, "n_trials": 0})
            errs["trials_big"] = await attempt(c, "run_payment_simulation", {**ok, "n_trials": 1000})
            errs["no_name_or_base"] = await attempt(c, "run_payment_simulation", {**ok, "policy": {"retry_limit": 2}})
            errs["bad_preset"] = await attempt(c, "generate_payment_scenarios", {"experiment_id": eid, "scenarios": [{"preset": "x"}]})
            errs["tiny_workload"] = await attempt(c, "generate_payment_scenarios", {"experiment_id": eid, "scenarios": [{"n_transactions": 5}]})
            errs["one_policy"] = await attempt(c, "compare_recovery_policies", {**{k: ok[k] for k in ("experiment_id", "scenario_id")}, "policies": [{"base": "immediate_retry"}]})
            errs["dup_names"] = await attempt(c, "compare_recovery_policies", {**{k: ok[k] for k in ("experiment_id", "scenario_id")}, "policies": [{"base": "immediate_retry"}, {"base": "immediate_retry"}]})
            errs["inconsistent_policy"] = await attempt(c, "run_payment_simulation", {**ok, "policy": {"name": "x", "request_buyer_action": True}})
            errs["trace_index"] = await attempt(c, "get_payment_transaction_trace", {**ok, "transaction_index": 10**6})
            errs["bad_weight"] = await attempt(c, "create_payment_experiment", {"name": "w", "objective": {"revenue_weight": 0}})
            # server must still be healthy after errors
            await c.call_tool("run_payment_simulation", {**ok, "n_trials": 1})
            return errs

    errs = _run(go())
    assert all(errs.values()) and len(errs) == 13
    assert "unknown experiment_id" in errs["unknown_exp"] and "unknown scenario_id" in errs["unknown_scn"]
    assert "retry_limit" in errs["retry_limit"] or "less than or equal" in errs["retry_limit"]
