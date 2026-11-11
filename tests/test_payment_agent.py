"""Unit tests for the autonomous payment agent.

EVIDENCE CLASS: these tests use a SCRIPTED LLM (agent.payments.testing.ScriptedBackend, SDK-typed responses) or
stubbed Gemini clients. They prove the adapter/loop/evaluator/persistence logic. They are NOT evidence that the real
Gemini API was called -- see tests/integration/test_gemini_live.py and the demo for that.
The MCP server, simulation engine and SQLite persistence ARE real in the tests that say so.
"""
import ast
import asyncio
import json
import os
import sqlite3

import pytest
from google.genai import errors as genai_errors
from google.genai import types

from agent.payments import evaluator
from agent.payments.agent import HIDDEN_PARAMS, HIDDEN_TOOLS, PaymentAgent
from agent.payments.llm import GeminiBackend, LLMError
from agent.payments.models import (DEFAULT_GEMINI_MODEL, AgentConfig, AgentError, MissingCredentialsError, ObjectiveSpec)
from agent.payments.optimizer import CandidateRegistry, POLICY_FIELDS, propose_variations
from agent.payments.planner import parse_objective
from agent.payments.testing import FailingBackend, ScriptedBackend, last_function_responses
from agent.payments.tool_adapter import McpToolAdapter, compact_result, make_stdio_client, to_gemini_schema
from storage.payment_agent_runs import AgentRunStore, scrub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FAKE_KEY = "AIzaFAKE-SECRET-KEY-FOR-TEST-0001"
OBJECTIVE = ("Optimize payment recovery for a merchant processing 1,000 transactions. Maximize recovered revenue while minimizing "
             "recovery cost and latency. Do not exceed two automatic retries. Never automatically retry terminal failures.")


def cfg_for(tmp, **kw):
    base = dict(api_key=FAKE_KEY, exploration_trials=3, final_trials=4, robustness_n_transactions=500, robustness_trials=3,
                db_path=str(tmp / "sim.db"), agent_db_path=str(tmp / "agent.db"))
    base.update(kw)
    return AgentConfig(**base)


def run(coro):
    return asyncio.run(coro)


# =============================================================== objective parsing
def test_parse_example_objective():
    s = parse_objective(OBJECTIVE)
    assert (s.n_transactions, s.max_retries, s.forbid_terminal_retries) == (1000, 2, True)
    assert s.provenance["max_retries"] == "parsed" and s.provenance["max_recovery_window_minutes"] == "default"
    assert s.warnings == () and s.constraints_payload()["max_retries"] == 2


@pytest.mark.parametrize("text,expected", [
    ("Do not exceed three automatic retries.", 3), ("At most 4 retries please.", 4), ("no more than one retry", 1)])
def test_parse_retry_limits(text, expected):
    assert parse_objective(text).max_retries == expected


def test_parse_window_units_overrides_and_unrecognised_constraints_warn():
    s = parse_objective("Recover within 6 hours. Never charge a card twice. Do not touch refunds.")
    assert s.max_recovery_window_minutes == 360.0
    assert len(s.warnings) == 2 and all("NOT enforced" in w for w in s.warnings)
    o = parse_objective("Do not exceed two retries.", max_retries=1, weight_latency=0.9)
    assert o.max_retries == 1 and o.weight_latency == 0.9 and o.provenance["max_retries"] == "override"
    with pytest.raises(ValueError):
        parse_objective("x", nonsense=1)


# =============================================================== schema conversion (Gemini function declarations)
def test_schema_conversion_rules():
    js = {"$defs": {"Step": {"type": "object", "properties": {"n": {"type": "integer", "exclusiveMinimum": 0, "title": "N", "default": 3}},
                             "required": ["n"], "additionalProperties": False}},
          "type": "object", "additionalProperties": False,
          "properties": {"steps": {"type": "array", "items": {"$ref": "#/$defs/Step"}, "minItems": 1, "maxItems": 4},
                         "opt": {"anyOf": [{"type": "number", "minimum": 1}, {"type": "null"}]},
                         "mode": {"anyOf": [{"const": "a"}, {"const": "b"}]},
                         "kind": {"type": "string", "enum": ["x", "y"], "description": "d"},
                         "max_work_units": {"type": "integer"}},
          "required": ["steps", "max_work_units"]}
    out = to_gemini_schema(js, hidden=frozenset({"max_work_units"}))
    assert "max_work_units" not in out["properties"] and out["required"] == ["steps"]
    step = out["properties"]["steps"]["items"]
    assert step["properties"]["n"] == {"type": "integer", "minimum": 0} and step["required"] == ["n"]
    assert out["properties"]["steps"]["minItems"] == 1 and out["properties"]["steps"]["maxItems"] == 4
    assert out["properties"]["opt"] == {"type": "number", "minimum": 1, "nullable": True}
    assert out["properties"]["mode"] == {"type": "string", "enum": ["a", "b"]}
    assert out["properties"]["kind"]["enum"] == ["x", "y"]
    assert "additionalProperties" not in json.dumps(out) and "$defs" not in json.dumps(out) and "default" not in json.dumps(out)
    types.Schema.model_validate(out)  # accepted by the real SDK type


# =============================================================== real MCP: discovery, declarations, execution
@pytest.fixture(scope="module")
def mcp_tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("mcp")


def test_mcp_discovery_declarations_and_execution(mcp_tmp):
    async def go():
        ad = McpToolAdapter(lambda: make_stdio_client("spikes.mcp_sim.server", str(mcp_tmp / "sim.db")), HIDDEN_TOOLS, HIDDEN_PARAMS)
        async with ad:
            tools = await ad.discover()
            decls = ad.declarations()
            exp = await ad.call("create_payment_experiment", {"name": "t"})  # hidden from the LLM, still callable by the harness
            bad = await ad.call("run_payment_simulation", {"experiment_id": "nope", "scenario_id": "x", "policy": {"base": "intelligent_recovery"}})
            invalid = await ad.call("create_payment_experiment", {"name": "t", "objective": {"revenue_weight": 0}})
            unknown = await ad.call("does_not_exist", {})
            return tools, decls, exp, bad, invalid, unknown

    tools, decls, exp, bad, invalid, unknown = run(go())
    names = {t.name for t in tools}
    assert {"run_sensitivity_analysis", "compare_recovery_policies", "run_payment_simulation", "generate_payment_scenarios",
            "get_payment_experiment_results", "get_payment_transaction_trace", "create_payment_experiment", "simulate_release_schedule"} == names
    declared = {d.name for d in decls}
    assert declared == names - HIDDEN_TOOLS  # legacy + harness-owned tool hidden from the LLM
    for d in decls:
        assert isinstance(d.parameters, types.Schema)
        assert "max_work_units" not in json.dumps(d.parameters.model_dump(exclude_none=True, mode="json"))
    cmp_ = next(d for d in decls if d.name == "compare_recovery_policies").parameters
    assert cmp_.properties["policies"].items.properties["alternate_funding_action"].enum == ["NONE", "REQUEST_ALTERNATE"]
    assert exp.ok and exp.data["provenance"] == "SIMULATED"
    assert not bad.ok and "unknown experiment_id" in bad.data
    assert not invalid.ok and not unknown.ok and "unknown tool" in unknown.data


class _StubClient:
    def __init__(self, behaviour):
        self.behaviour = behaviour

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return None

    async def list_tools(self):
        return []

    async def call_tool(self, name, args):
        return await self.behaviour(name, args)


def test_tool_failures_and_timeouts_become_error_results():
    async def boom(name, args):
        raise RuntimeError("server exploded")

    async def slow(name, args):
        await asyncio.sleep(5)

    async def go(behaviour, timeout):
        ad = McpToolAdapter(lambda: _StubClient(behaviour), tool_timeout_s=timeout)
        async with ad:
            ad.tools = {"t": object()}
            return await ad.call("t", {})

    r1 = run(go(boom, 5))
    r2 = run(go(slow, 0.05))
    assert not r1.ok and "RuntimeError: server exploded" in r1.data
    assert not r2.ok and "timed out" in r2.data


# =============================================================== candidate generation
DEF = {"name": "base", "retry_limit": 2, "retry_intervals": [5.0], "exponential_backoff": True, "backoff_base_minutes": 5.0,
       "backoff_multiplier": 2.0, "backoff_max_interval_minutes": 240.0, "maximum_recovery_window": 1440.0,
       "buyer_action_timeout": 600.0, "request_buyer_action": True, "alternate_funding_action": "REQUEST_ALTERNATE",
       "terminal_failure_handling": "FLAG_FOR_REVIEW", "classification_aware": True, "idempotent_retries": True}


def test_variations_change_exactly_one_dimension_deterministically():
    v1, v2 = propose_variations(DEF, max_variants=20), propose_variations(DEF, max_variants=20)
    assert v1 == v2 and len({x["name"] for x in v1}) == len(v1)
    for x in v1:
        diff = [k for k in POLICY_FIELDS if x.get(k) != DEF[k]]
        assert x["name"].startswith("base~") and x["change"]
        assert 1 <= len(diff) <= 2  # base-interval variants also lift the cap when needed
    by = {x["name"]: x for x in v1}
    assert by["base~retries1"]["retry_limit"] == 1 and by["base~retries3"]["retry_limit"] == 3
    assert by["base~base10"]["backoff_base_minutes"] == 10.0 and by["base~buyer300"]["buyer_action_timeout"] == 300.0
    assert by["base~altfund_none"]["alternate_funding_action"] == "NONE"
    assert by["base~terminal_abandon"]["terminal_failure_handling"] == "ABANDON"


def test_variations_respect_policy_shape_and_filters():
    blind = {**DEF, "exponential_backoff": False, "request_buyer_action": False, "alternate_funding_action": "NONE", "retry_limit": 0}
    names = {x["name"] for x in propose_variations(blind, max_variants=20)}
    assert "base~retries1" in names and not any(n.startswith("base~retries-") for n in names)
    assert not any("base" in n.split("~")[1][:4] and "mult" in n for n in names) and not any("buyer" in n.split("~")[1][:5] for n in names)
    assert all("retries" in x["name"] for x in propose_variations(DEF, ["retry_limit"]))
    assert len(propose_variations(DEF, max_variants=3)) == 3
    assert {x["name"] for x in propose_variations({**DEF, "retry_limit": 20}, ["retry_limit"])} == {"base~retries19"}


def test_registry_discards_observations_when_name_is_reused_for_a_different_policy():
    reg = CandidateRegistry()
    reg.register("p", DEF, "gemini")
    assert reg.register("p", DEF, "gemini").revisions == 0
    changed = reg.register("p", {**DEF, "retry_limit": 1}, "gemini")
    assert changed.revisions == 1 and changed.definition["retry_limit"] == 1 and changed.observations == []


# =============================================================== deterministic evaluator (pure functions)
def _row(name, feasible, score, violations=(), ci=(0.0, 1.0)):
    return {"rank": 0, "policy": name, "feasible": feasible, "score_mean": score, "score_ci95": list(ci), "violations": list(violations)}


def _final(rows, diffs=None, n_trials=20):
    return {"ranking": rows, "policies": {r["policy"]: {} for r in rows}, "n_trials": n_trials,
            "paired_score_difference_vs_best": diffs or {}}


def test_selection_rejects_infeasible_and_explains_every_rejection():
    final = _final([_row("a", True, 0.5), _row("b", True, 0.4), _row("c", False, 0.9, ["retry_limit 5 exceeds max_retries 2"])],
                   {"b": {"mean_difference": 0.1, "ci95": [0.05, 0.15], "ci_excludes_zero": True},
                    "c": {"mean_difference": -0.4, "ci95": [-0.5, -0.3], "ci_excludes_zero": True}})
    sel = evaluator.select_winner(final)
    assert sel.winner == "a" and sel.n_feasible == 2
    rej = {r.policy: r for r in sel.rejected}
    assert rej["c"].kind == "INFEASIBLE" and "max_retries" in rej["c"].reasons[0]  # highest score but infeasible
    assert rej["b"].kind == "LOWER_SCORE" and rej["b"].score_gap_to_winner == 0.1 and rej["b"].gap_significant is True
    assert evaluator.select_winner(final) == sel  # deterministic


def test_selection_with_no_feasible_candidate():
    sel = evaluator.select_winner(_final([_row("x", False, 0.3, ["v"]), _row("y", False, 0.1, ["v"])]))
    assert sel.winner is None and all(r.kind == "INFEASIBLE" for r in sel.rejected)
    conf = evaluator.assess_confidence(None, {}, None, None, [])
    assert conf.label == "NONE" and "hard constraints" in conf.reasons[0]


def _sens(points, most=()):
    return {"points": [{"parameter": p, "value": v, "winner": w} for p, v, w in points], "most_sensitive_parameters": list(most)}


GOOD_DIFFS = {"b": {"mean_difference": 0.1, "ci95": [0.05, 0.15], "ci_excludes_zero": True}}


def test_sensitivity_findings_change_recommendation_confidence():
    final = _final([_row("a", True, 0.5), _row("b", True, 0.4)], GOOD_DIFFS)
    scen = [{"scenario": "s1", "winner": "a"}, {"scenario": "s2", "winner": "a"}]
    robust = _sens([("p", i, "a") for i in range(10)])
    moderate = _sens([("p", i, "a") for i in range(7)] + [("q", i, "b") for i in range(3)], ["q"])
    fragile = _sens([("p", i, "a") for i in range(3)] + [("q", i, "b") for i in range(7)], ["q"])
    high = evaluator.assess_confidence("a", final, final, robust, scen)
    mid = evaluator.assess_confidence("a", final, final, moderate, scen)
    low = evaluator.assess_confidence("a", final, final, fragile, scen)
    assert (high.label, mid.label, low.label) == ("HIGH", "MEDIUM", "LOW")
    assert high.assumption_sensitivity["label"] == "ROBUST" and mid.assumption_sensitivity["flip_conditions"][0]["new_winner"] == "b"
    assert any("assumption variations" in r for r in low.reasons)


def test_confidence_distinguishes_four_notions_and_never_claims_real_world():
    final = _final([_row("a", True, 0.5), _row("b", True, 0.4)], GOOD_DIFFS)
    c = evaluator.assess_confidence("a", final, final, _sens([("p", 1, "a")]), [{"scenario": "s", "winner": "a"}])
    assert {"repeatability", "statistical_uncertainty", "assumption_sensitivity", "scenario_robustness"} <= set(c.to_dict())
    assert "not confidence in real-world" in c.disclaimer
    # non-reproducible re-run => repeatability fails => cannot be HIGH
    other = _final([_row("a", True, 0.51), _row("b", True, 0.4)], GOOD_DIFFS)
    c2 = evaluator.assess_confidence("a", final, other, _sens([("p", 1, "a")]), [{"scenario": "s", "winner": "a"}])
    assert c2.repeatability["verified"] is False and c2.label == "LOW"
    # inseparable margin or lost scenarios lowers confidence
    flat = _final([_row("a", True, 0.5), _row("b", True, 0.49)], {"b": {"mean_difference": 0.01, "ci95": [-0.02, 0.04], "ci_excludes_zero": False}})
    assert evaluator.assess_confidence("a", flat, flat, _sens([("p", 1, "a")]), [{"scenario": "s", "winner": "a"}]).label == "LOW"
    lost = evaluator.assess_confidence("a", final, final, _sens([("p", 1, "a")]), [{"scenario": "s", "winner": "b"}])
    assert lost.scenario_robustness["label"] == "WINS_NONE" and lost.label == "LOW"
    solo = _final([_row("a", True, 0.5), _row("b", False, 0.7, ["v"])])
    assert evaluator.assess_confidence("a", solo, solo, _sens([("p", 1, "a")]), [{"scenario": "s", "winner": "a"}]).label == "MEDIUM"  # capped


def test_compact_result_keeps_what_the_llm_needs():
    full = {"best_policy": "a", "best_policy_feasible": True, "n_trials": 3, "n_transactions": 10, "master_seed": 1,
            "ranking": [{**_row("a", True, 0.5), "rank": 1}],
            "policies": {"a": {"summary": {"recovery_rate": {"value": 0.5, "ci95_wilson": [0.4, 0.6]},
                                           "per_trial_means": {"recovered_revenue": 10, "total_retries": 3, "unnecessary_retries": 0,
                                                               "total_processing_cost": 1.5, "buyer_actions_requested": 2},
                                           "avg_recovery_time_minutes": {"value": 12.0}}}},
            "paired_score_difference_vs_best": {}}
    c = compact_result("compare_recovery_policies", full)
    assert c["ranking"][0]["recovery_rate"] == 0.5 and c["ranking"][0]["feasible"] is True and c["best_policy"] == "a"


# =============================================================== full agent runs (real MCP server + engine + SQLite; SCRIPTED LLM)
def _script():
    def t2(contents):
        r = last_function_responses(contents)[0]["result"]
        best = [x for x in r["ranking"] if x["feasible"]][0]["policy"]  # adapts to the previous result
        return [("propose_policy_variations", {"policy_name": best, "dimensions": ["backoff_base_minutes", "buyer_action_timeout"]})]

    def t3(contents):
        v = last_function_responses(contents)[0]["result"]["variants"]
        return [("compare_recovery_policies", {"policies": [{k: x[k] for k in x if k != "change"} for x in v[:2]]
                                               + [{"name": "bad_retries4", "base": "intelligent_recovery", "retry_limit": 4}], "n_trials": 3})]

    return [[("compare_recovery_policies", {"policies": [{"base": "intelligent_recovery"}, {"name": "c2", "base": "intelligent_recovery", "retry_limit": 2}], "n_trials": 3})],
            t2, t3, [("finish_optimization", {"shortlist": ["c2"], "rationale": "c2 satisfies the retry cap"})]]


@pytest.fixture(scope="module")
def agent_run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("agentrun")
    backend = ScriptedBackend(_script())
    agent = PaymentAgent(cfg_for(tmp), backend=backend)
    report = run(agent.run(OBJECTIVE))
    return report, backend, agent, tmp


def test_multi_iteration_adaptive_run_and_real_mcp_calls(agent_run):
    report, backend, agent, _ = agent_run
    assert report.termination_reason == "finished_by_llm" and report.llm_activity["turns"] == 4 and report.llm_activity["function_calls"] == 4
    assert report.llm["provider"] == "SCRIPTED_TEST_DOUBLE"  # never mislabelled as Gemini
    assert report.experiments["by_gemini"] == 2 and report.experiments["by_harness"] >= 5
    # turn 3 consumed turn 2's output (adaptation): the variants it compared exist as variation-origin candidates
    origins = {c["name"]: c["origin"] for c in report.candidates_explored}
    assert origins["c2"] == "gemini" and sum(o == "variation" for o in origins.values()) == 2 and origins["immediate_retry"] == "baseline"
    # real results went back to the model as function responses
    seen = backend.seen[1][-1].parts[0].function_response
    assert seen.name == "compare_recovery_policies" and seen.response["result"]["ranking"][0]["policy"] and "_harness" in seen.response
    assert backend.configs[0].automatic_function_calling.disable is True
    decl_names = {d.name for d in backend.configs[0].tools[0].function_declarations}
    assert {"propose_policy_variations", "finish_optimization", "compare_recovery_policies", "run_sensitivity_analysis"} <= decl_names
    assert not decl_names & HIDDEN_TOOLS


def test_infeasible_policies_rejected_and_never_selected(agent_run):
    report = agent_run[0]
    rej = {r["policy"]: r for r in report.rejected_alternatives}
    assert rej["immediate_retry"]["kind"] == "INFEASIBLE" and any("max_retries" in x for x in rej["immediate_retry"]["reasons"])
    assert report.selected_policy["definition"]["retry_limit"] <= 2 and report.selected_policy["definition"]["classification_aware"] is True
    bad = next(c for c in report.candidates_explored if c["name"] == "bad_retries4")
    assert bad["ever_feasible"] is False  # LLM proposed it; the evaluator flagged it
    fe = {r["policy"]: r for r in report.final_evaluation["ranking"]}
    assert report.selected_policy["name"] in fe and fe[report.selected_policy["name"]]["feasible"] is True
    assert all(not r["feasible"] or r["policy"] == report.selected_policy["name"] or r["score"] <= fe[report.selected_policy["name"]]["score"]
               for r in fe.values())


def test_selection_is_deterministic_given_identical_results(agent_run, tmp_path):
    report, _, _, _ = agent_run
    # re-running the whole pipeline with the same script/objective/config reproduces the selection and numbers
    again = run(PaymentAgent(cfg_for(tmp_path), backend=ScriptedBackend(_script())).run(OBJECTIVE))
    assert again.selected_policy["name"] == report.selected_policy["name"]
    assert again.final_evaluation == report.final_evaluation
    assert again.confidence["label"] == report.confidence["label"]


def test_sensitivity_and_confidence_in_report(agent_run):
    report = agent_run[0]
    conf = report.confidence
    assert report.sensitivity_findings["label"] in ("ROBUST", "MODERATE", "FRAGILE") and report.sensitivity_findings["n_points"] > 10
    assert conf["repeatability"]["verified"] is True and conf["label"] in ("HIGH", "MEDIUM", "LOW")
    assert conf["scenario_robustness"]["scenarios"] and "not confidence in real-world" in conf["disclaimer"]
    sens_calls = [c for c in report.simulation_configurations if c["tool"] == "run_sensitivity_analysis"]
    assert sens_calls and sens_calls[0]["phase"] == "final"


def test_decision_report_has_every_required_section(agent_run):
    report = agent_run[0]
    d = report.to_dict()
    for k in ("merchant_objective", "initial_policy", "candidates_explored", "experiments", "simulation_configurations", "final_evaluation",
              "feasibility_decisions", "sensitivity_findings", "selected_policy", "rejected_alternatives", "confidence",
              "assumptions_and_limitations", "execution_id", "llm", "budget", "termination_reason"):
        assert d[k] not in (None, "", []) or k == "rejected_alternatives", k
    assert d["initial_policy"]["name"] == "immediate_retry" and d["initial_policy"]["feasible"] is False
    assert d["experiments"]["simulation_calls"] == len(d["simulation_configurations"])
    json.dumps(d)  # serialisable
    md = report.to_markdown()
    for h in ("## Merchant objective", "## Selected policy", "## Rejected alternatives", "## Sensitivity findings",
              "## Recommendation confidence", "## Assumptions and limitations", "SIMULATED"):
        assert h in md
    assert "LLM explanation (written by SCRIPTED_TEST_DOUBLE" in md


def test_budget_and_guard_arguments_were_injected(agent_run):
    _, _, agent, _ = agent_run
    store = agent.store
    run_row = store.conn.execute("SELECT execution_id FROM payment_agent_runs").fetchone()
    calls = store.tool_calls(run_row[0])
    llm_sim = [c for c in calls if c["source"] == "gemini" and c["tool"] == "compare_recovery_policies"]
    assert llm_sim and all("max_work_units" in json.loads(c["args_json"]) for c in llm_sim)
    assert all(json.loads(c["args_json"])["scenario_id"] for c in llm_sim)


def test_persistence_of_traces_and_no_secrets_stored(agent_run):
    report, _, agent, tmp = agent_run
    store = agent.store
    eid = report.execution_id
    run_row = store.get_run(eid)
    assert run_row["status"] == "COMPLETED" and run_row["model"] == DEFAULT_GEMINI_MODEL and run_row["experiment_id"]
    assert "api_key" not in json.loads(run_row["config_json"])
    calls = store.tool_calls(eid)
    assert len(calls) == report.experiments["tool_calls_total"]  # every call (MCP + local) is persisted
    assert sum(c["tool"] in ("propose_policy_variations", "finish_optimization") for c in calls) == 2
    assert {"gemini", "harness"} == {c["source"] for c in calls}
    assert [e["kind"] for e in store.events(eid)][:2] == ["objective_parsed", "tools_discovered"]
    assert {e["kind"] for e in store.events(eid)} >= {"llm_turn", "finish_requested", "final_selection", "confidence"}
    names = {c["name"] for c in store.candidates(eid)}
    assert {"c2", "immediate_retry"} <= names
    assert json.loads(run_row["final_report_json"])["selected_policy"]["name"] == report.selected_policy["name"]
    store.conn.commit()
    for f in os.listdir(tmp):  # raw bytes of every database file: the key must not appear anywhere
        if f.endswith(".db"):
            assert FAKE_KEY.encode() not in open(tmp / f, "rb").read()


def test_scrub_removes_secrets_by_key_and_by_value():
    out = scrub({"api_key": "x", "nested": {"Authorization": "Bearer abc", "note": f"key={FAKE_KEY}!"}, "list": [FAKE_KEY, 1]}, [FAKE_KEY])
    s = json.dumps(out)
    assert FAKE_KEY not in s and "Bearer abc" not in s and s.count("REDACTED") == 4
    assert scrub({"max_tokens": 5}) == {"max_tokens": 5}


# =============================================================== guardrails, limits and failures (scripted LLM, real MCP)
def test_tool_failures_are_returned_to_the_llm_and_the_agent_continues(tmp_path):
    script = [[("compare_recovery_policies", {"policies": [{"base": "intelligent_recovery"}, {"name": "x", "base": "exponential_backoff", "retry_limit": 99}], "n_trials": 3}),
               ("compare_recovery_policies", {"experiment_id": "someone-elses", "policies": [{"base": "intelligent_recovery"}, {"base": "no_recovery"}]}),
               ("compare_recovery_policies", {"scenario_id": "bogus", "policies": [{"base": "intelligent_recovery"}, {"base": "no_recovery"}]}),
               ("simulate_release_schedule", {"params": {}}), ("create_payment_experiment", {"name": "hijack"}),
               ("no_such_tool", {})],
              [("compare_recovery_policies", {"policies": [{"name": "ok1", "base": "intelligent_recovery", "retry_limit": 2},
                                                           {"name": "ok2", "base": "intelligent_recovery", "retry_limit": 1}], "n_trials": 3})],
              [("finish_optimization", {"shortlist": ["ok1", "ghost"], "rationale": "r"})]]
    backend = ScriptedBackend(script)
    report = run(PaymentAgent(cfg_for(tmp_path, include_llm_explanation=False), backend=backend).run(OBJECTIVE))
    responses = [p.function_response.response for p in backend.seen[1][-1].parts]
    errs = [r["error"] for r in responses if "error" in r]
    assert len(responses) == 6 and len(errs) == 6
    assert any("retry_limit" in e or "less than or equal" in e for e in errs)
    assert any("does not belong to this run" in e for e in errs) and any("unknown scenario_id" in e for e in errs)
    assert sum("not available" in e for e in errs) == 3  # legacy tool, harness-only tool, nonexistent tool
    assert report.termination_reason == "finished_by_llm" and report.selected_policy is not None
    fin = [p.function_response.response for p in backend.seen[-2][-1].parts] if False else None  # noqa: F841
    assert report.llm_activity["llm_shortlist"] == ["ok1"]  # unknown candidate name dropped
    assert report.llm_explanation is None


def test_iteration_limit_enforced(tmp_path):
    forever = [[("compare_recovery_policies", {"policies": [{"name": f"p{i}", "base": "intelligent_recovery", "retry_limit": 1},
                                                            {"name": f"q{i}", "base": "intelligent_recovery", "retry_limit": 2}], "n_trials": 2})] for i in range(10)]
    backend = ScriptedBackend(forever)
    report = run(PaymentAgent(cfg_for(tmp_path, max_iterations=3, include_llm_explanation=False), backend=backend).run(OBJECTIVE))
    assert report.termination_reason == "max_iterations" and report.llm_activity["turns"] == 3 and report.llm_activity["function_calls"] == 3
    assert report.selected_policy is not None  # evaluator still selects from what was explored


def test_tool_call_limit_enforced(tmp_path):
    many = [("compare_recovery_policies", {"policies": [{"name": f"p{i}", "base": "intelligent_recovery", "retry_limit": 1},
                                                        {"name": f"q{i}", "base": "intelligent_recovery", "retry_limit": 2}], "n_trials": 2}) for i in range(4)]
    backend = ScriptedBackend([many, [("finish_optimization", {"shortlist": ["p0"], "rationale": "r"})]])
    report = run(PaymentAgent(cfg_for(tmp_path, max_tool_calls=2, include_llm_explanation=False), backend=backend).run(OBJECTIVE))
    responses = [p.function_response.response for p in backend.seen[1][-1].parts]
    assert sum("result" in r for r in responses) == 2 and sum("tool-call limit" in r.get("error", "") for r in responses) == 2
    assert report.experiments["by_gemini"] == 2 and report.termination_reason == "finished_by_llm"  # got a grace turn to finish


def test_model_that_ignores_the_tool_call_limit_is_stopped(tmp_path):
    call = lambda i: ("compare_recovery_policies", {"policies": [{"name": f"p{i}", "base": "intelligent_recovery", "retry_limit": 1},  # noqa: E731
                                                                 {"name": f"q{i}", "base": "intelligent_recovery", "retry_limit": 2}], "n_trials": 2})
    backend = ScriptedBackend([[call(0), call(1), call(2)], [call(3)], [call(4)], [call(5)]])
    report = run(PaymentAgent(cfg_for(tmp_path, max_tool_calls=1, include_llm_explanation=False), backend=backend).run(OBJECTIVE))
    assert report.termination_reason == "max_tool_calls" and report.llm_activity["turns"] == 2 and report.experiments["by_gemini"] == 1


def test_simulation_budget_enforced(tmp_path):
    big = [("compare_recovery_policies", {"policies": [{"name": f"p{i}", "base": "intelligent_recovery", "retry_limit": 1},
                                                       {"name": f"q{i}", "base": "intelligent_recovery", "retry_limit": 2}], "n_trials": 5}) for i in range(2)]
    backend = ScriptedBackend([big, [("finish_optimization", {"shortlist": ["p0"], "rationale": "r"})]])
    # initial policy costs 1000*3 = 3000 units; each 2-policy x 5-trial compare costs 10,000 -> budget 14,000 allows exactly one
    report = run(PaymentAgent(cfg_for(tmp_path, exploration_budget_units=14_000, include_llm_explanation=False), backend=backend).run(OBJECTIVE))
    responses = [p.function_response.response for p in backend.seen[1][-1].parts]
    assert "result" in responses[0] and "budget exceeded" in responses[1]["error"]
    assert report.budget["exploration_used"] == 13_000 and report.budget["exploration_used"] <= report.budget["exploration_limit"]


def test_llm_failure_ends_exploration_but_run_degrades_gracefully(tmp_path):
    report = run(PaymentAgent(cfg_for(tmp_path), backend=FailingBackend(LLMError("boom"))).run(OBJECTIVE))
    assert report.termination_reason == "llm_error" and report.selected_policy is None
    assert report.confidence["label"] == "NONE" and any("LLM error" in n for n in report.notes)
    assert report.llm_activity["function_calls"] == 0  # no claim that the model did anything


def test_model_that_never_calls_tools_is_reported_honestly(tmp_path):
    report = run(PaymentAgent(cfg_for(tmp_path, include_llm_explanation=False), backend=ScriptedBackend(["I think intelligent recovery is best."])).run(OBJECTIVE))
    assert report.termination_reason == "model_stopped_without_finish" and report.llm_activity["function_calls"] == 0
    assert report.experiments["by_gemini"] == 0


def test_failed_run_is_persisted_with_status_failed(tmp_path):
    cfg = cfg_for(tmp_path)
    agent = PaymentAgent(cfg, backend=ScriptedBackend([]), client_factory=lambda: make_stdio_client("module.that.does.not.exist", None, 20))
    with pytest.raises(Exception):
        run(agent.run(OBJECTIVE))
    row = agent.store.conn.execute("SELECT status, error FROM payment_agent_runs").fetchone()
    assert row["status"] == "FAILED" and row["error"]


# =============================================================== credentials, model configuration, Gemini backend behaviour
def test_missing_credentials_are_reported_not_faked():
    with pytest.raises(MissingCredentialsError):
        AgentConfig.from_env({})
    with pytest.raises(MissingCredentialsError):
        PaymentAgent(AgentConfig())  # no key and no injected backend
    with pytest.raises(MissingCredentialsError):
        GeminiBackend(AgentConfig())


def test_model_selection_is_configurable_from_environment():
    assert AgentConfig.from_env({"GEMINI_API_KEY": "k"}).model == DEFAULT_GEMINI_MODEL
    assert AgentConfig.from_env({"GEMINI_API_KEY": "k", "GEMINI_MODEL": "my-model"}).model == "my-model"
    assert AgentConfig.from_env({"GOOGLE_API_KEY": "g"}).api_key == "g"
    assert AgentConfig.from_env({"GEMINI_API_KEY": "k"}, model="override").model == "override"
    assert "api_key" not in AgentConfig.from_env({"GEMINI_API_KEY": "k"}).redacted() and "k" not in repr(AgentConfig(api_key="k-secret"))[:0] + ""
    assert "k-secret" not in repr(AgentConfig(api_key="k-secret"))


def test_model_name_is_not_hardcoded_outside_the_config_module():
    hits = []
    for dp, _, files in os.walk(os.path.join(ROOT, "agent", "payments")):
        for f in files:
            if f.endswith(".py") and "gemini-" in open(os.path.join(dp, f)).read():
                hits.append(f)
    assert hits == ["models.py"]


class _FakeModels:
    def __init__(self, outcomes):
        self.outcomes, self.calls = list(outcomes), 0

    async def generate_content(self, model, contents, config):
        self.calls += 1
        o = self.outcomes.pop(0)
        if isinstance(o, Exception):
            raise o
        return o


class _FakeClient:
    def __init__(self, outcomes):
        self.aio = type("A", (), {"models": _FakeModels(outcomes)})()


def test_gemini_backend_retries_transient_errors_only():
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    ok = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="hi")]))])
    client = _FakeClient([genai_errors.ServerError(503, {"error": {"message": "overloaded"}}), genai_errors.ClientError(429, {"error": {"message": "rate"}}), ok])
    b = GeminiBackend(AgentConfig(api_key="k", model="m", llm_retries=3), client=client, sleep=fake_sleep)
    assert run(b.generate([], types.GenerateContentConfig())) is ok and client.aio.models.calls == 3 and slept == [1, 2]

    bad = _FakeClient([genai_errors.ClientError(400, {"error": {"message": "bad request"}})])
    with pytest.raises(LLMError) as ei:
        run(GeminiBackend(AgentConfig(api_key="k"), client=bad, sleep=fake_sleep).generate([], types.GenerateContentConfig()))
    assert "400" in str(ei.value) and bad.aio.models.calls == 1  # not retried

    always = _FakeClient([genai_errors.ServerError(503, {"error": {"message": "x"}})] * 3)
    with pytest.raises(LLMError):
        run(GeminiBackend(AgentConfig(api_key="k", llm_retries=2), client=always, sleep=fake_sleep).generate([], types.GenerateContentConfig()))
    assert always.aio.models.calls == 3


def _quota_error(delay, code=429):
    return genai_errors.ClientError(code, {"error": {"code": code, "message": "You exceeded your current quota", "status": "RESOURCE_EXHAUSTED", "details": [
        {"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [{"description": "Generate content requests per day limit exceeded on Free Tier.",
                                                                              "quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20",
                                                                              "quotaDimensions": {"model": "m-served"}}]},
        {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": delay}]}})


def test_daily_quota_is_not_retried_and_error_is_reported_in_full():
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    client = _FakeClient([_quota_error("60110s")] * 4)
    with pytest.raises(LLMError) as ei:
        run(GeminiBackend(AgentConfig(api_key="k"), client=client, sleep=fake_sleep).generate([], types.GenerateContentConfig()))
    msg = str(ei.value)
    assert client.aio.models.calls == 1 and slept == []  # no pointless retries
    assert "per day limit exceeded" in msg and "limit=20" in msg and "model=m-served" in msg and "16.7 h" in msg
    assert len(msg) > 300  # not truncated to the first 300 characters


def test_short_rate_limit_waits_the_server_advised_delay_then_succeeds():
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    ok = types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="hi")]))])
    client = _FakeClient([_quota_error("7.5s"), ok])
    assert run(GeminiBackend(AgentConfig(api_key="k"), client=client, sleep=fake_sleep).generate([], types.GenerateContentConfig())) is ok
    assert slept == [8.5] and client.aio.models.calls == 2
    from agent.payments.llm import rate_limit_info
    assert rate_limit_info(ValueError("x")) == (None, None)


def test_response_meta_records_the_model_that_actually_served_the_request():
    from agent.payments.agent import response_meta
    r = types.GenerateContentResponse(candidates=[], model_version="gemini-served-version", response_id="rid-1",
                                      usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=10, candidates_token_count=5, total_token_count=15))
    assert response_meta(r) == {"model_version": "gemini-served-version", "response_id": "rid-1", "prompt_tokens": 10, "output_tokens": 5, "total_tokens": 15}


# =============================================================== architecture boundary
def test_agent_never_imports_the_simulation_engine():
    bad = []
    for dp, _, files in os.walk(os.path.join(ROOT, "agent", "payments")):
        for f in (x for x in files if x.endswith(".py")):
            tree = ast.parse(open(os.path.join(dp, f)).read())
            for n in ast.walk(tree):
                mods = [n.module or ""] if isinstance(n, ast.ImportFrom) else [a.name for a in n.names] if isinstance(n, ast.Import) else []
                bad += [(f, m) for m in mods if m == "engine" or m.startswith("engine.") or m.startswith("mcp_server") or m.startswith("spikes")]
    assert bad == []
