"""Dashboard data layer, source labelling, redaction, error handling, objective composition and the background job runner.
(No Streamlit rendering here: see test_dashboard_payments_ui.py.) See tests/dash_fixtures.py for the provenance of fixture data."""
import ast
import asyncio
import os
import time

import pytest

from agent.payments.models import MissingCredentialsError
from dashboard.payments import data, errors, objective, paths, redaction
from dashboard.payments.page_experiments import apply_filters
from dashboard.payments.runner import JobBusy, JobManager
from dashboard.payments.sources import DataSource, Metric
from tests.dash_fixtures import FAKE_GEMINI_KEY, build_world

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dashdata")
    mp = pytest.MonkeyPatch()
    w = build_world(tmp)
    mp.setenv("INOEMA_PAYMENT_DB", w["sim"])
    mp.setenv("INOEMA_AGENT_DB", w["agent"])
    mp.setenv("INOEMA_PAYPAL_DB", w["sandbox"])
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET"):
        mp.delenv(k, raising=False)
    yield w
    mp.undo()


# ============================================================ dashboard data retrieval
def test_paths_follow_the_backend_env_vars(world):
    assert (paths.sim_db_path(), paths.agent_db_path(), paths.sandbox_db_path()) == (world["sim"], world["agent"], world["sandbox"])


def test_missing_databases_yield_empty_results_and_are_not_created(tmp_path, monkeypatch):
    for k in ("INOEMA_PAYMENT_DB", "INOEMA_AGENT_DB", "INOEMA_PAYPAL_DB"):
        monkeypatch.setenv(k, str(tmp_path / f"{k}.db"))
    assert data.list_experiments() == [] and data.list_agent_runs() == [] and data.list_sandbox_executions() == []
    assert data.latest_genuine_agent_run() is None and data.build_trace("nope") is None
    m = data.overview_metrics()
    assert all(x.value in (None, 0, "—") or x.label == "Total experiments" for g in ("simulated", "agent", "sandbox") for x in m[g])
    assert os.listdir(tmp_path) == []  # viewing the dashboard never creates databases


def test_experiments_are_linked_to_the_newest_agent_run_and_runs_are_annotated(world):
    exps = data.list_experiments()
    assert len(exps) == 1 and exps[0]["config"]["constraints"]["max_retries"] == 2
    e = exps[0]
    # two agent rows reference this experiment (the scripted-double run and the relabelled GEMINI_API fixture copy): the newest one is linked
    assert e["agent_run"]["execution_id"] == world["gemini_fixture_id"] and e["agent_run"]["llm_provider"] == "GEMINI_API"
    detail = data.get_agent_run(world["scripted_id"])
    runs = data.annotate_runs(data.get_experiment_runs(e["experiment_id"]), detail)
    who = {r["requested_by"] for r in runs}
    assert {"harness setup", "Gemini turn 1", "deterministic final evaluation"} <= who
    assert {r["kind"] for r in runs} == {"run", "comparison", "sensitivity"}
    assert all(r["requested_by"] == "no agent trace persisted" for r in data.annotate_runs(runs, None))


def test_comparison_rows_are_copied_not_recomputed(world):
    detail = data.get_agent_run(world["scripted_id"])
    final = next(c for c in detail["tool_calls"] if c["tool"] == "compare_recovery_policies" and c["iteration"] < 0)
    rows = {r["Policy"]: r for r in data.comparison_rows(final["result"])}
    for name, pol in final["result"]["policies"].items():
        s = pol["summary"]
        r = rows[name]
        assert r["Recovery rate"] == s["recovery_rate"]["value"] and (r["RR CI low"], r["RR CI high"]) == tuple(s["recovery_rate"]["ci95_wilson"])
        assert r["Recovered revenue / trial"] == s["per_trial_means"]["recovered_revenue"] and r["Avg recovery (min)"] == s["avg_recovery_time_minutes"]["value"]
        rk = next(x for x in final["result"]["ranking"] if x["policy"] == name)
        assert r["Score"] == rk["score_mean"] and r["Feasible"] == rk["feasible"]
    assert any(not r["Feasible"] for r in rows.values()) and any(r["Feasible"] for r in rows.values())
    single = next(c for c in detail["tool_calls"] if c["tool"] == "run_payment_simulation")
    assert len(data.comparison_rows(single["result"])) == 1  # single-run results are handled too


def test_filters_and_sorting_logic(world):
    detail = data.get_agent_run(world["scripted_id"])
    final = next(c for c in detail["tool_calls"] if c["tool"] == "compare_recovery_policies" and c["iteration"] < 0)
    import pandas as pd
    df = pd.DataFrame(data.comparison_rows(final["result"]))
    allr = apply_filters(df, feasibility="All", min_rr=0, max_minutes=1e9, name="", sort_by="Score", ascending=False)
    assert list(allr["Score"]) == sorted(df["Score"], reverse=True)
    feas = apply_filters(df, feasibility="Feasible only", min_rr=0, max_minutes=1e9, name="", sort_by="Score", ascending=False)
    assert feas["Feasible"].all() and 0 < len(feas) < len(df)
    inf = apply_filters(df, feasibility="Infeasible only", min_rr=0, max_minutes=1e9, name="", sort_by="Score", ascending=True)
    assert (~inf["Feasible"]).all() and len(feas) + len(inf) == len(df)
    thr = float(df["Recovery rate"].median())
    assert (apply_filters(df, feasibility="All", min_rr=thr, max_minutes=1e9, name="", sort_by="Rank", ascending=True)["Recovery rate"] >= thr).all()
    assert (apply_filters(df, feasibility="All", min_rr=0, max_minutes=100, name="", sort_by="Rank", ascending=True)["Avg recovery (min)"] <= 100).all()
    one = df["Policy"].iloc[0]
    assert list(apply_filters(df, feasibility="All", min_rr=0, max_minutes=1e9, name=one[:6].upper(), sort_by="Rank", ascending=True)["Policy"])[0:1] != []
    assert apply_filters(df, feasibility="All", min_rr=0, max_minutes=1e9, name="zzz-no-match", sort_by="Score", ascending=True).empty


def test_sensitivity_frames_shape(world):
    detail = data.get_agent_run(world["scripted_id"])
    sens = next(c for c in detail["tool_calls"] if c["tool"] == "run_sensitivity_analysis" and c["ok"])
    f = data.sensitivity_frames(sens["result"])
    assert len(f["points"]) == sum(len(p["ranking"]) for p in sens["result"]["points"]) and f["robustness"]["label"] in ("ROBUST", "MODERATE", "FRAGILE")
    assert f["summary"] and f["assumptions"]["failure_model"]


# ============================================================ metric source labelling: never mixed
def test_every_overview_metric_has_exactly_one_correct_source(world):
    m = data.overview_metrics()
    assert {m_.source for m_ in m["simulated"]} == {DataSource.SIMULATED}
    assert {m_.source for m_ in m["agent"]} == {DataSource.AGENT_DERIVED}
    assert {m_.source for m_ in m["sandbox"]} == {DataSource.SANDBOX_ACTUAL}
    labels = [x.label for g in ("simulated", "agent", "sandbox") for x in m[g]]
    assert len(labels) == len(set(labels))  # no metric appears under two sources
    for must in ("Total experiments", "Recovery rate (selected policy)", "Recovered revenue / trial", "Avg recovery latency", "Selected recovery policy", "Policy feasibility"):
        assert must in labels


def test_sandbox_metrics_come_only_from_the_sandbox_store(world, tmp_path, monkeypatch):
    base = {x.label: x.value for x in data.overview_metrics()["sandbox"]}
    assert base["Sandbox executions"] == 2 and base["Captures completed"] == 1 and base["Last PayPal order id"]
    sim_before = {x.label: x.value for x in data.overview_metrics()["simulated"]}
    monkeypatch.setenv("INOEMA_PAYPAL_DB", str(tmp_path / "empty.db"))  # empty sandbox store, simulation untouched
    empty = {x.label: x.value for x in data.overview_metrics()["sandbox"]}
    assert empty["Sandbox executions"] == 0 and empty["Captures completed"] == 0
    assert {x.label: x.value for x in data.overview_metrics()["simulated"]} == sim_before  # sandbox data never affects simulated metrics
    monkeypatch.setenv("INOEMA_PAYMENT_DB", str(tmp_path / "nosim.db"))
    assert {x.label: x.value for x in data.overview_metrics()["sandbox"]} == empty  # ... and vice versa


def test_render_metrics_refuses_mixed_sources():
    from dashboard.payments.ui import render_metrics
    with pytest.raises(AssertionError):
        render_metrics([Metric("a", 1, DataSource.SIMULATED), Metric("b", 2, DataSource.SANDBOX_ACTUAL)], key_prefix="x")


def test_selected_policy_view_uses_the_stored_final_evaluation(world):
    detail = data.get_agent_run(world["scripted_id"])
    v = data.selected_policy_view(detail)
    rep = detail["run"]["report"]
    assert v["name"] == rep["selected_policy"]["name"] and v["feasible"] is True and v["recovered_revenue"] > 0 and v["recovery_rate"] > 0
    row = next(r for r in rep["final_evaluation"]["ranking"] if r["policy"] == v["name"])
    assert v["score"] == pytest.approx(row["score"], abs=1e-4) and v["recovery_rate"] == pytest.approx(row["recovery_rate"], abs=1e-4)


# ============================================================ demonstration mode source of truth
def test_latest_genuine_run_never_returns_a_scripted_double(world):
    last = data.latest_genuine_agent_run()
    assert last["llm_provider"] == "GEMINI_API" and last["execution_id"] == world["gemini_fixture_id"]
    assert {r["llm_provider"] for r in data.list_agent_runs(provider="GEMINI_API")} == {"GEMINI_API"}
    assert {r["llm_provider"] for r in data.list_agent_runs()} == {"GEMINI_API", "SCRIPTED_TEST_DOUBLE"}


# ============================================================ traces + redaction
def test_trace_structure_and_narratives(world):
    t = data.build_trace(world["scripted_id"])
    assert t["run"]["llm_provider"] == "SCRIPTED_TEST_DOUBLE" and t["served_models"] == []  # no per-turn served model recorded for this run
    titles = [p["title"] for p in t["phases"]]
    assert titles[0].startswith("Harness") and "Gemini turn 1" in titles and titles[-1].startswith("Deterministic final")
    calls = [c for p in t["phases"] for c in p["calls"]]
    assert {c["tool"] for c in calls} >= {"create_payment_experiment", "compare_recovery_policies", "run_sensitivity_analysis", "propose_policy_variations"}
    cmp_ = next(c for c in calls if c["tool"] == "compare_recovery_policies")
    assert "Compared" in cmp_["narrative"] and "rejected as infeasible" in cmp_["narrative"] and cmp_["simulation_run_id"]
    assert t["experiment_ids"]["experiment_id"] and len(t["experiment_ids"]["simulation_run_ids"]) >= 5
    cands = {c["name"]: c for c in t["candidates"]}
    assert cands["immediate_retry"]["origin"] == "baseline" and cands["c2"]["changes_vs_initial"]
    var = next(c for c in t["candidates"] if "~" in c["name"])
    assert var["changes_vs_parent"] and len(var["changes_vs_parent"]) <= 2  # a variation differs from its parent in one dimension


def test_trace_output_is_redacted_even_if_a_secret_reached_the_database(world, tmp_path, monkeypatch):
    import shutil, sqlite3, json
    agent_copy = str(tmp_path / "agent_leaky.db")
    shutil.copy(world["agent"], agent_copy)
    conn = sqlite3.connect(agent_copy)
    leaky = json.dumps({"note": f"key was {FAKE_GEMINI_KEY}", "api_key": FAKE_GEMINI_KEY, "Authorization": "Bearer abcdefghijklmnop", "ok": 1})
    conn.execute("UPDATE payment_agent_tool_calls SET args_json=?, result_json=? WHERE execution_id=? AND seq=?", (leaky, leaky, world["scripted_id"], 2))
    conn.commit()
    conn.close()
    monkeypatch.setenv("INOEMA_AGENT_DB", agent_copy)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    t = data.build_trace(world["scripted_id"])
    dump = repr(t["phases"])
    assert FAKE_GEMINI_KEY not in dump and "abcdefghijklmnop" not in dump and "***REDACTED***" in dump


def test_redaction_rules():
    s = redaction.sanitize({"api_key": "k", "client_secret": "s", "access_token": "t", "n": {"x": "AIzaSyAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "b": "Bearer abcdefgh12345"}, "long": "y" * 9000}, extra_secrets=("hunter2secret",))
    blob = repr(s)
    assert "AIzaSy" not in blob and "abcdefgh12345" not in blob and s["api_key"] == s["client_secret"] == s["access_token"] == "***REDACTED***"
    assert len(s["long"]) < 4100 and "truncated" in s["long"]
    assert redaction.sanitize("pw hunter2secret", extra_secrets=("hunter2secret",)) == "pw ***REDACTED***"
    assert redaction.sanitize({"max_tokens": 5, "n": 1}) == {"max_tokens": 5, "n": 1}  # ordinary fields untouched


# ============================================================ error handling
@pytest.mark.parametrize("text,kind", [
    ("Gemini rate limit/quota exhausted (429); server advises waiting 60110s (~16.7 h), longer than 90s. RESOURCE_EXHAUSTED", "QUOTA_EXHAUSTED"),
    ("ClientError: 429 RESOURCE_EXHAUSTED. You exceeded your current quota", "QUOTA_EXHAUSTED"),
    ("404 NOT_FOUND. This model models/x is no longer available to new users", "MODEL_UNAVAILABLE"),
    ("API key not valid. Please pass a valid API key. API_KEY_INVALID", "AUTH"),
    ("504 DEADLINE_EXCEEDED. Deadline expired", "TIMEOUT"),
    ("MCP server is missing required tools: ['x']", "MCP_ERROR"),
    ("something odd", "UNKNOWN")])
def test_error_classification(text, kind):
    assert errors.classify_error(text).kind == kind


def test_quota_error_gives_retry_hint_and_never_suggests_a_fake_model():
    e = errors.classify_error("quota exhausted; server advises waiting 60110s (~16.7 h)")
    assert e.retry_after == "about 16.7 hours" and "Demonstration mode" in e.advice and "will not substitute a scripted model" in e.advice
    m = errors.classify_error(MissingCredentialsError("GEMINI_API_KEY is not set"))
    assert m.kind == "MISSING_CREDENTIALS" and "No fake model" in m.advice


# ============================================================ objective composition
def test_form_to_objective_roundtrip_through_the_agents_parser():
    f = objective.MerchantForm(n_transactions=12_000, max_retries=3, recovery_window_hours=48, forbid_terminal_retries=True, latency_weight=0.7,
                               cost_weight=0.5, amount_median=120.0, amount_sigma=0.6, scenario_preset="buyer_friction")
    spec = objective.preview(f)
    assert (spec.n_transactions, spec.max_retries, spec.max_recovery_window_minutes) == (12_000, 3, 2880.0)
    assert spec.forbid_terminal_retries and spec.weight_latency == 0.7 and spec.weight_cost == 0.5
    assert (spec.amount_median, spec.amount_sigma, spec.scenario_preset) == (120.0, 0.6, "buyer_friction")
    assert spec.provenance["max_retries"] == "override" and "three automatic retries" in spec.text
    off = objective.preview(objective.MerchantForm(forbid_terminal_retries=False))
    assert off.forbid_terminal_retries is False and "terminal" not in off.text


def test_form_validation():
    assert objective.validate(objective.MerchantForm()) == []
    bad = objective.MerchantForm(n_transactions=5, max_retries=50, recovery_window_hours=0, latency_weight=-1, amount_median=0, amount_sigma=9)
    assert len(objective.validate(bad)) == 6
    notes = objective.preview(objective.MerchantForm(extra_notes="Never charge a card twice."))
    assert any("NOT enforced" in w for w in notes.warnings)  # unrecognised constraint is surfaced, not silently dropped


# ============================================================ optimisation invocation (fake agent: no network, no model)
class _FakeReport:
    def __init__(self, term="finished_by_llm", notes=()):
        self.execution_id, self.termination_reason, self.notes = "exec-fake-1", term, list(notes)

    def to_dict(self):
        return {"execution_id": self.execution_id, "termination_reason": self.termination_reason}


class _FakeAgent:
    calls: list = []

    def __init__(self, cfg, progress, behaviour):
        self.cfg, self.progress, self.behaviour = cfg, progress, behaviour

    async def run(self, text, **overrides):
        _FakeAgent.calls.append((self.cfg, text, overrides))
        self.progress("fake agent progress line")
        return await self.behaviour()


def _manager(behaviour):
    return JobManager(agent_factory=lambda cfg, progress: _FakeAgent(cfg, progress, behaviour))


def test_start_invokes_the_agent_with_objective_overrides_and_config_but_never_a_key_in_the_job(world, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    _FakeAgent.calls.clear()

    async def ok():
        return _FakeReport()

    m = _manager(ok)
    f = objective.MerchantForm(max_retries=2, latency_weight=0.4)
    job = m.start(objective.compose_objective_text(f), objective.to_overrides(f), max_iterations=5, exploration_trials=7, final_trials=9, model="my-model")
    job = m.wait(job.job_id)
    cfg, text, ov = _FakeAgent.calls[0]
    assert job.state == "COMPLETED" and job.execution_id == "exec-fake-1" and job.report["termination_reason"] == "finished_by_llm"
    assert "two automatic retries" in text and ov["max_retries"] == 2 and ov["weight_latency"] == 0.4 and ov["max_recovery_window_minutes"] == 4320.0
    assert (cfg.max_iterations, cfg.exploration_trials, cfg.final_trials, cfg.model) == (5, 7, 9, "my-model")
    assert cfg.db_path == world["sim"] and cfg.agent_db_path == world["agent"]  # same stores the dashboard reads
    assert FAKE_GEMINI_KEY not in repr(job.config) and "api_key" not in job.config
    assert job.lines() == ["fake agent progress line"]


def test_start_without_credentials_raises_and_does_not_fall_back(world):
    m = JobManager()  # the real manager/factory
    with pytest.raises(MissingCredentialsError):
        m.start("x", {})
    assert m.latest() is None  # no job was created


def test_only_one_job_runs_at_a_time(world, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    release = {"go": False}

    async def slow():
        while not release["go"]:
            await asyncio.sleep(0.01)
        return _FakeReport()

    m = _manager(slow)
    j = m.start("x", {})
    assert j.running and m.active() is j
    with pytest.raises(JobBusy):
        m.start("y", {})
    release["go"] = True
    assert m.wait(j.job_id).state == "COMPLETED" and m.active() is None
    assert m.start("z", {}) is not None  # next job allowed once the first has finished
    m.wait(m.latest().job_id)


def test_job_failures_and_llm_errors_are_classified(world, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)

    async def boom():
        raise RuntimeError("Gemini rate limit/quota exhausted (429); server advises waiting 60110s (~16.7 h)")

    async def interrupted():
        return _FakeReport("llm_error", ["LLM error ended exploration: 404 NOT_FOUND no longer available to new users"])

    j1 = _manager(boom)
    a = j1.wait(j1.start("x", {}).job_id)
    assert a.state == "FAILED" and a.error.kind == "QUOTA_EXHAUSTED" and a.error.retry_after == "about 16.7 hours" and a.report is None
    j2 = _manager(interrupted)
    b = j2.wait(j2.start("x", {}).job_id)
    assert b.state == "COMPLETED_WITH_LLM_ERROR" and b.error.kind == "MODEL_UNAVAILABLE" and b.report["termination_reason"] == "llm_error"


def test_job_progress_is_sanitised(world, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)

    async def leaky():
        raise RuntimeError("x")

    m = _manager(leaky)
    j = m.start("x", {})
    m.wait(j.job_id)
    j.log(f"token {FAKE_GEMINI_KEY} Bearer abcdefghijklmnop")
    assert FAKE_GEMINI_KEY not in " ".join(j.lines()) and "abcdefghijklmnop" not in " ".join(j.lines())


def test_dashboard_never_imports_the_scripted_test_double():
    """No scripted/mock LLM may be reachable from the dashboard's production code paths."""
    bad = []
    for dp, _, files in os.walk(os.path.join(ROOT, "dashboard")):
        for f in (x for x in files if x.endswith(".py")):
            tree = ast.parse(open(os.path.join(dp, f)).read())
            for n in ast.walk(tree):
                mods = [n.module or ""] if isinstance(n, ast.ImportFrom) else [a.name for a in n.names] if isinstance(n, ast.Import) else []
                names = [a.name for a in n.names] if isinstance(n, ast.ImportFrom) else []
                bad += [(f, m) for m in mods if "payments.testing" in m or "paypal_fake" in m] + [(f, x) for x in names if x in ("ScriptedBackend", "FailingBackend")]
    assert bad == []


def test_dashboard_does_not_reimplement_business_logic():
    """The UI layer calls backend modules; it must not contain the simulation maths, scoring, or PayPal decision code."""
    for dp, _, files in os.walk(os.path.join(ROOT, "dashboard", "payments")):
        for f in (x for x in files if x.endswith(".py")):
            src = open(os.path.join(dp, f)).read()
            for forbidden in ("np.random", "simpy", "def next_action", "def classify(", "def score_trial", "wilson_interval(", "httpx"):
                assert forbidden not in src, (f, forbidden)


# ============================================================ Phase 5.1: completeness, demo-run selection, workflow states
@pytest.mark.parametrize("run,complete,label", [
    ({"llm_provider": "GEMINI_API", "status": "COMPLETED", "termination_reason": "finished_by_llm"}, True, "COMPLETE"),
    ({"llm_provider": "GEMINI_API", "status": "COMPLETED", "termination_reason": "llm_error"}, False, "INTERRUPTED (llm_error)"),
    ({"llm_provider": "GEMINI_API", "status": "COMPLETED", "termination_reason": "max_iterations"}, False, "INTERRUPTED (max_iterations)"),
    ({"llm_provider": "GEMINI_API", "status": "COMPLETED", "termination_reason": "model_stopped_without_finish"}, False, "INTERRUPTED (model_stopped_without_finish)"),
    ({"llm_provider": "GEMINI_API", "status": "FAILED", "termination_reason": "finished_by_llm"}, False, "INTERRUPTED (finished_by_llm)"),
    ({"llm_provider": "SCRIPTED_TEST_DOUBLE", "status": "COMPLETED", "termination_reason": "finished_by_llm"}, False, "NOT A GEMINI RUN (SCRIPTED_TEST_DOUBLE)"),
])
def test_only_a_normally_finished_gemini_run_counts_as_complete(run, complete, label):
    assert data.is_complete(run) is complete and data.run_state_label(run) == label


def test_demo_run_prefers_the_latest_complete_genuine_run_even_if_a_newer_interrupted_one_exists(world, tmp_path, monkeypatch):
    from tests.dash_fixtures import variant_agent_db
    v = variant_agent_db(world["agent"], str(tmp_path / "a.db"), add_newer_interrupted=True)
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "a.db"))
    newest_any = data.list_agent_runs(provider="GEMINI_API")[0]
    assert newest_any["execution_id"] == v["newer_interrupted_id"]  # the interrupted one really is the newest genuine row
    pick = data.latest_genuine_agent_run()
    assert pick["execution_id"] == world["gemini_fixture_id"] and pick["complete"] is True and pick["state"] == "COMPLETE"


def test_demo_run_with_only_interrupted_genuine_runs_is_flagged_incomplete(world, tmp_path, monkeypatch):
    from tests.dash_fixtures import variant_agent_db
    variant_agent_db(world["agent"], str(tmp_path / "a.db"), interrupt_genuine=True)
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "a.db"))
    pick = data.latest_genuine_agent_run()
    assert pick["complete"] is False and pick["completed_autonomously"] is False and pick["state"] == "INTERRUPTED (llm_error)"
    steps = {s["key"]: s["state"][0] for s in data.workflow_steps(gemini_configured=True, paypal_configured=True)}
    assert steps["optimize"] == "INTERRUPTED"


def test_overview_metrics_can_be_pinned_to_a_specific_execution(world, tmp_path, monkeypatch):
    from tests.dash_fixtures import variant_agent_db
    v = variant_agent_db(world["agent"], str(tmp_path / "a.db"), add_newer_interrupted=True)
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "a.db"))
    unpinned = {m.label: m for m in data.overview_metrics()["agent"]}
    pinned = {m.label: m for m in data.overview_metrics(execution_id=world["gemini_fixture_id"])["agent"]}
    assert unpinned["Latest agent execution"].value == v["newer_interrupted_id"][:8] and "INTERRUPTED (llm_error)" in unpinned["Latest agent execution"].caption
    assert pinned["Latest agent execution"].value == world["gemini_fixture_id"][:8] and pinned["Latest agent execution"].caption.endswith("COMPLETE")


def test_workflow_steps_cover_the_five_sections_with_the_right_sources(world):
    steps = data.workflow_steps(gemini_configured=True, paypal_configured=True)
    assert [s["section"] for s in steps] == ["Overview", "Policy Optimization", "Experiments", "PayPal Sandbox", "Agent Execution Traces"]
    assert [s["source"] for s in steps] == [DataSource.SYSTEM, DataSource.AGENT_DERIVED, DataSource.SIMULATED, DataSource.SANDBOX_ACTUAL, DataSource.AGENT_DERIVED]
    st_ = {s["key"]: s["state"][0] for s in steps}
    assert st_ == {"overview": "DONE", "optimize": "COMPLETE", "experiments": "DONE", "sandbox": "VERIFIED", "traces": "DONE"}


def test_sandbox_workflow_step_reflects_awaiting_buyer_and_unconfigured(world, tmp_path, monkeypatch):
    import sqlite3 as _s, shutil
    shutil.copy(world["sandbox"], str(tmp_path / "sb.db"))
    c = _s.connect(str(tmp_path / "sb.db"))
    c.execute("UPDATE sandbox_executions SET status='AWAITING_BUYER' WHERE status='COMPLETED'")
    c.commit()
    c.close()
    monkeypatch.setenv("INOEMA_PAYPAL_DB", str(tmp_path / "sb.db"))
    sb = {s["key"]: s["state"] for s in data.workflow_steps(gemini_configured=True, paypal_configured=True)}["sandbox"]
    assert sb[0] == "IN_PROGRESS" and "awaiting buyer approval" in sb[1]
    monkeypatch.setenv("INOEMA_PAYPAL_DB", str(tmp_path / "none.db"))
    assert {s["key"]: s["state"] for s in data.workflow_steps(gemini_configured=True, paypal_configured=False)}["sandbox"] == ("NOT_STARTED", "PayPal Sandbox not configured")
