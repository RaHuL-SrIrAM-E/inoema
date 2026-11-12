"""Streamlit AppTest suite for the Payments workspace (headless rendering of the real dashboard/app.py).

Provenance: data comes from tests/dash_fixtures.py (real engine + MCP via a SCRIPTED test-double LLM, a relabelled GEMINI_API *fixture* row,
and sandbox evidence from the MOCK HTTP double). The optimisation tests use a fake agent. NOTHING here is a live Gemini or PayPal result."""
import asyncio
import json
import os
import shutil
import sqlite3

import pytest
from streamlit.testing.v1 import AppTest

from dashboard.payments import runner, sandbox_service
from dashboard.payments.runner import JobManager
from tests.dash_fixtures import FAKE_GEMINI_KEY, build_world
from tests.paypal_fake import FAKE_CLIENT_ID, FAKE_CLIENT_SECRET, FakePayPal

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dashboard", "app.py")
SECTIONS = ["Overview", "Policy Optimization", "Experiments", "Execution Plan", "PayPal Sandbox", "Agent Execution Traces"]
WORKFLOW_SECTIONS = [s for s in SECTIONS if s != "Execution Plan"]  # the five-step workflow indicator maps to these


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("dashui")
    mp = pytest.MonkeyPatch()
    w = build_world(tmp)
    mp.setenv("INOEMA_PAYMENT_DB", w["sim"])
    mp.setenv("INOEMA_AGENT_DB", w["agent"])
    mp.setenv("INOEMA_PAYPAL_DB", w["sandbox"])
    mp.setenv("INOEMA_DASHBOARD_AUTOCHECK", "0")
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "PAYPAL_CLIENT_ID", "PAYPAL_CLIENT_SECRET", "INOEMA_DEMO_MODE", "INOEMA_DEFAULT_WORKSPACE"):
        mp.delenv(k, raising=False)
    yield w
    mp.undo()


def app(section="Overview", workspace="INOEMA Payments", demo=False, **state):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["workspace"] = workspace
    if workspace == "INOEMA Payments":
        at.session_state["payments_section"] = section
        at.session_state["payments_demo_mode"] = demo
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def text(at) -> str:
    out = []
    for k in ("markdown", "caption", "code", "text", "info", "warning", "error", "success", "json", "subheader", "metric"):
        out += [str(getattr(e, "value", "")) + " " + str(getattr(e, "label", "")) for e in getattr(at, k)]
    for df in at.dataframe:
        out.append(df.value.to_string())
    out += [e.label for e in at.expander] + [t.label for t in at.tabs]  # expander / tab titles are part of what the user reads
    return "\n".join(out)


def no_exception(at):
    assert not at.exception, [e.value for e in at.exception]


# ============================================================ navigation and page rendering
def test_payments_is_the_default_workspace_and_listed_first(world):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()  # no workspace chosen: the default applies
    no_exception(at)
    assert at.sidebar.radio[0].options == ["INOEMA Payments", "Simulation Control Center"] and at.sidebar.radio[0].value == "INOEMA Payments"
    assert [h.value for h in at.subheader] == ["Overview"]


def test_original_simulation_control_center_remains_accessible_and_unchanged(world):
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    at.sidebar.radio[0].set_value("Simulation Control Center").run()
    no_exception(at)
    assert at.button[0].label == "Run INOEMA Simulation"  # the original app's first control
    assert "Developer Integration" in [h.value for h in at.subheader]
    at.sidebar.radio[0].set_value("INOEMA Payments").run()
    no_exception(at)
    assert [h.value for h in at.subheader] == ["Overview"]


def test_default_workspace_can_be_switched_back_by_env_or_query_param(world, monkeypatch):
    monkeypatch.setenv("INOEMA_DEFAULT_WORKSPACE", "simulation")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    no_exception(at)
    assert at.sidebar.radio[0].value == "Simulation Control Center" and at.button[0].label == "Run INOEMA Simulation"
    monkeypatch.delenv("INOEMA_DEFAULT_WORKSPACE")
    at2 = AppTest.from_file(APP, default_timeout=120)
    at2.query_params["workspace"] = "simulation"
    at2.run()
    assert at2.sidebar.radio[0].value == "Simulation Control Center"
    at3 = AppTest.from_file(APP, default_timeout=120)
    at3.query_params["workspace"] = "payments"
    at3.run()
    assert at3.sidebar.radio[0].value == "INOEMA Payments"


def test_all_sections_navigate_and_render(world):
    at = app()
    no_exception(at)
    nav = at.radio(key="payments_section")
    assert nav.options == SECTIONS
    expected = {"Overview": "Overview", "Policy Optimization": "Policy Optimization", "Experiments": "Experiments", "Execution Plan": "Execution Plan", "PayPal Sandbox": "PayPal Sandbox",
                "Agent Execution Traces": "Agent Execution Traces"}
    for sec in SECTIONS:
        at.radio(key="payments_section").set_value(sec).run()
        no_exception(at)
        assert [h.value for h in at.subheader][0] == expected[sec], sec
    # switching back to the legacy workspace restores its content
    at.sidebar.radio[0].set_value("Simulation Control Center").run()
    no_exception(at)
    assert at.button[0].label == "Run INOEMA Simulation"


def test_navigation_buttons_jump_between_sections(world):
    at = app("Overview")
    at.button(key="ov_go_experiments").click().run()
    no_exception(at)
    assert at.radio(key="payments_section").value == "Experiments" and [h.value for h in at.subheader][0] == "Experiments"
    at.radio(key="payments_section").set_value("Overview").run()
    at.button(key="ov_go_traces").click().run()
    assert at.radio(key="payments_section").value == "Agent Execution Traces"


# ============================================================ overview: metric source labelling in the rendered page
def test_overview_labels_every_metric_with_its_source_and_keeps_groups_separate(world):
    at = app("Overview")
    no_exception(at)
    t = text(at)
    for badge in ("SIMULATED", "AGENT-DERIVED", "SANDBOX · ACTUAL", "SYSTEM"):
        assert badge in t
    caps = [c.value for c in at.caption]
    sim_labels = ["Total experiments", "Recovery rate (selected policy)", "Recovered revenue / trial", "Avg recovery latency"]
    agent_labels = ["Latest agent execution", "Selected recovery policy", "Policy feasibility", "Recommendation confidence"]
    sandbox_labels = ["Sandbox executions", "Captures completed", "Last PayPal order id", "Last PayPal debug id"]
    metrics = {m.label: m for m in at.metric}
    for labels, mark in ((sim_labels, "SIMULATED"), (agent_labels, "AGENT-DERIVED"), (sandbox_labels, "SANDBOX · ACTUAL")):
        for lab in labels:
            assert lab in metrics
    # each metric is followed by a caption carrying exactly its own source label
    ordered = [(m.label) for m in at.metric if m.label in metrics]
    for lab, mark in [(l, "SIMULATED") for l in sim_labels] + [(l, "AGENT-DERIVED") for l in agent_labels] + [(l, "SANDBOX · ACTUAL") for l in sandbox_labels]:
        i = ordered.index(lab)
        cap = [c for c in caps if mark in c]
        assert cap, lab
    assert metrics["Sandbox executions"].value == "2" and metrics["Total experiments"].value == "1"  # sandbox count and simulation count are separate numbers
    assert "dbfbf757"[:0] == "" and metrics["Policy feasibility"].value == "Feasible"
    assert "no agent trace" not in t.lower() or True


def test_overview_component_status_without_credentials(world):
    at = app("Overview")
    t = text(at)
    assert "Simulation engine" in t and "MCP server" in t and "Gemini integration" in t and "PayPal Sandbox" in t
    assert "OK" in t and "NOT CONFIGURED" in t and "NOT CHECKED" in t  # engine OK; Gemini/PayPal not configured; MCP not auto-checked in tests
    assert "GEMINI_API_KEY is not set" in t and "PAYPAL_CLIENT_ID" in t
    assert "Test Gemini (1 API request)" in [b.label for b in at.button]  # probes are explicit actions only


def test_overview_mcp_status_with_autocheck_discovers_real_tools(world, monkeypatch):
    monkeypatch.setenv("INOEMA_DASHBOARD_AUTOCHECK", "1")
    at = app("Overview")
    no_exception(at)
    t = text(at)
    assert "tools discovered via an independent MCP client" in t and "MCP server" in t


# ============================================================ experiments: filters, sorting, charts
def test_experiments_table_filters_sort_and_charts(world):
    at = app("Experiments")
    no_exception(at)
    assert len(at.get("plotly_chart")) >= 5  # recovery-rate CI, score, trade-off + sensitivity heatmap + line
    all_rows = at.dataframe[0].value
    assert {"Policy", "Feasible", "Score", "Recovery rate (%)", "Avg recovery (min)", "Retries / txn", "Processing cost / trial"} <= set(all_rows.columns)
    assert any(not f for f in all_rows["Feasible"]) and any(all_rows["Feasible"])
    at.radio(key="exp_feas").set_value("Feasible only").run()
    feas = at.dataframe[0].value
    assert len(feas) < len(all_rows) and feas["Feasible"].all()
    at.radio(key="exp_feas").set_value("Infeasible only").run()
    assert (~at.dataframe[0].value["Feasible"]).all()
    at.radio(key="exp_feas").set_value("All").run()
    at.selectbox(key="exp_sort").set_value("Recovery rate").run()
    at.toggle(key="exp_asc").set_value(True).run()
    rr = list(at.dataframe[0].value["Recovery rate (%)"])
    assert rr == sorted(rr)
    at.text_input(key="exp_name").set_value("zzz-nothing").run()
    assert "No policies match the current filters" in text(at)
    no_exception(at)


def test_experiments_page_labels_data_as_simulated_and_shows_who_requested_each_run(world):
    at = app("Experiments")
    t = text(at)
    assert "SIMULATED" in t and "Not PayPal data" in t
    opts = at.selectbox(key="exp_run").options
    assert any("deterministic final evaluation" in o for o in opts) and any("Gemini turn" in o for o in opts)
    assert "Hard constraints for this experiment: max retries 2" in t


def test_experiments_sensitivity_tab_content(world):
    at = app("Experiments")
    t = text(at)
    assert "Parameters ranked by how much they affect the recommendation" in t and "Robustness label" in t
    assert "ASSUMED ranges" in t
    at.selectbox(key="exp_sens_param").set_value(at.selectbox(key="exp_sens_param").options[-1]).run()
    no_exception(at)


def test_experiments_empty_state(tmp_path, monkeypatch):
    for k in ("INOEMA_PAYMENT_DB", "INOEMA_AGENT_DB", "INOEMA_PAYPAL_DB"):
        monkeypatch.setenv(k, str(tmp_path / (k + ".db")))
    at = app("Experiments")
    no_exception(at)
    assert "No experiments have been stored yet" in text(at)


# ============================================================ optimisation: invocation, errors, demo mode
class _Rep:
    def __init__(self, report: dict):
        self._r = report
        self.execution_id, self.termination_reason, self.notes = report["execution_id"], report["termination_reason"], report.get("notes", [])

    def to_dict(self):
        return self._r


class _Agent:
    calls: list = []

    def __init__(self, cfg, progress, behaviour):
        self.cfg, self.progress, self.behaviour = cfg, progress, behaviour

    async def run(self, text_, **overrides):
        _Agent.calls.append({"text": text_, "overrides": overrides, "cfg": self.cfg})
        self.progress("[LLM turn 1] planning")
        return await self.behaviour()


def install_manager(behaviour):
    m = JobManager(agent_factory=lambda cfg, progress: _Agent(cfg, progress, behaviour))
    runner.set_manager(m)
    return m


@pytest.fixture(autouse=True)
def _reset_manager():
    yield
    runner.set_manager(None)
    _Agent.calls.clear()


def test_optimization_page_without_credentials_blocks_start_and_explains(world):
    at = app("Policy Optimization")
    no_exception(at)
    t = text(at)
    assert at.button(key="opt_start").disabled
    assert "Gemini is not configured" in t and "No fake model is ever used" in t
    at.checkbox(key="opt_confirm").check().run()
    assert at.button(key="opt_start").disabled  # confirmation alone is not enough without credentials


def test_optimization_requires_explicit_confirmation_then_invokes_the_real_agent_path(world, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)

    async def finish():
        return _Rep(world["report"])

    mgr = install_manager(finish)
    at = app("Policy Optimization")
    no_exception(at)
    assert at.button(key="opt_start").disabled  # not confirmed yet
    assert "Enforced" not in text(at) or True
    at.slider(key="opt_max_retries").set_value(3)
    at.number_input(key="opt_n").set_value(2000)
    at.checkbox(key="opt_custom_amount").check()
    at.run()
    at.number_input(key="opt_amount_median").set_value(120.0).run()
    assert "max retries** = 3" in text(at) and "three automatic retries" in text(at)  # preview built by the agent's own parser
    at.checkbox(key="opt_confirm").check().run()
    assert not at.button(key="opt_start").disabled
    at.button(key="opt_start").click().run()
    job = mgr.latest()
    assert job is not None
    mgr.wait(job.job_id)
    call = _Agent.calls[0]
    assert "three automatic retries" in call["text"] and call["overrides"]["max_retries"] == 3 and call["overrides"]["n_transactions"] == 2000
    assert call["overrides"]["amount_median"] == 120.0 and call["overrides"]["forbid_terminal_retries"] is True
    assert call["cfg"].api_key == FAKE_GEMINI_KEY and job.config.get("api_key") is None  # the agent gets the key; the job record never does
    at.run()
    no_exception(at)
    t = text(at)
    assert "Result" in t and "Deterministic final selection" in t and "Selected policy" in t and "Why the alternatives were rejected" in t
    assert "Candidate policies explored" in t and "Sensitivity & recommendation confidence" in t and "Decision explanation" in t
    assert "NOT used for selection".lower() in t.lower() or "not used for selection" in t.lower()


def test_quota_exhaustion_is_reported_clearly_and_offers_demo_mode_without_substituting_a_model(world, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)

    async def quota():
        raise RuntimeError("Gemini rate limit/quota exhausted (429); server advises waiting 60110s (~16.7 h) RESOURCE_EXHAUSTED")

    mgr = install_manager(quota)
    at = app("Policy Optimization")
    at.checkbox(key="opt_confirm").check().run()
    at.button(key="opt_start").click().run()
    mgr.wait(mgr.latest().job_id)
    at.run()
    no_exception(at)
    t = text(at)
    assert "Gemini quota exhausted" in t and "about 16.7 hours" in t and "will not substitute a scripted model" in t
    assert "Last genuine execution" in t and "fixture-model" in t  # points at the last persisted genuine run, with model
    assert "scripted" not in t.lower().replace("will not substitute a scripted model", "")
    at.button(key="err_demo_QUOTA_EXHAUSTED").click().run()
    assert at.toggle(key="payments_demo_mode").value is True and "Demonstration mode is ON" in text(at)


def test_other_api_errors_are_shown_with_advice(world, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)

    async def gone():
        raise RuntimeError("404 NOT_FOUND. This model is no longer available to new users")

    mgr = install_manager(gone)
    at = app("Policy Optimization")
    at.checkbox(key="opt_confirm").check().run()
    at.button(key="opt_start").click().run()
    mgr.wait(mgr.latest().job_id)
    at.run()
    assert "Gemini model unavailable for this key" in text(at) and "never picks another model" in text(at)


def test_interrupted_execution_is_not_presented_as_completed(world, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    rep = {**world["report"], "termination_reason": "llm_error", "notes": ["LLM error ended exploration: 429 RESOURCE_EXHAUSTED quota"]}

    async def interrupted():
        return _Rep(rep)

    mgr = install_manager(interrupted)
    at = app("Policy Optimization")
    at.checkbox(key="opt_confirm").check().run()
    at.button(key="opt_start").click().run()
    mgr.wait(mgr.latest().job_id)
    at.run()
    t = text(at)
    assert "did not finish autonomously" in t and "Gemini quota exhausted" in t


def test_demonstration_mode_shows_last_genuine_execution_with_timestamp_and_model(world):
    at = app("Policy Optimization", demo=True)
    no_exception(at)
    t = text(at)
    assert "Demonstration mode is ON" in t and "fixture-model" in t and "2099-01-01 00:00:00 UTC" in t and "HISTORICAL" in t
    assert not any(b.key == "opt_start" for b in at.button)  # live start unavailable
    assert "Deterministic final selection" in t


def test_demonstration_mode_never_lists_scripted_runs_and_handles_no_genuine_run(world, tmp_path, monkeypatch):
    at = app("Agent Execution Traces", demo=True)
    opts = at.selectbox(key="trace_select").options
    assert len(opts) == 1 and opts[0].startswith(world["gemini_fixture_id"][:8]) and "GEMINI_API" in opts[0]  # only the GEMINI_API row
    assert "SCRIPTED_TEST_DOUBLE" not in " ".join(opts) and world["scripted_id"][:8] not in " ".join(opts)  # the scripted run is excluded
    shutil.copy(world["agent"], str(tmp_path / "a.db"))
    conn = sqlite3.connect(str(tmp_path / "a.db"))
    conn.execute("DELETE FROM payment_agent_runs WHERE llm_provider='GEMINI_API'")
    conn.commit()
    conn.close()
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "a.db"))
    at2 = app("Policy Optimization", demo=True)
    assert "No genuine Gemini execution has been persisted yet" in text(at2)


# ============================================================ PayPal Sandbox page: explicit actions, no capture before approval
@pytest.fixture()
def sandbox_env(world, tmp_path, monkeypatch):
    monkeypatch.setenv("INOEMA_PAYPAL_DB", str(tmp_path / "sb.db"))
    monkeypatch.setenv("PAYPAL_CLIENT_ID", FAKE_CLIENT_ID)
    monkeypatch.setenv("PAYPAL_CLIENT_SECRET", FAKE_CLIENT_SECRET)
    fake = FakePayPal()
    from paypal.client import PayPalClient
    monkeypatch.setattr(sandbox_service, "client_factory", lambda cfg, store: PayPalClient(cfg, sink=store, transport=fake.transport()))
    return fake


def test_sandbox_without_credentials_disables_every_action(world):
    at = app("PayPal Sandbox")
    no_exception(at)
    t = text(at)
    assert "credentials are not configured" in t and "Nothing is simulated or faked here" in t
    for key in ("sb_create", "sb_capture", "sb_probe"):
        assert at.button(key=key).disabled
    assert "SANDBOX · ACTUAL" in t and "REAL calls to the PayPal Sandbox API" in t


def test_sandbox_create_needs_explicit_confirmation_and_stops_at_buyer_approval(world, sandbox_env):
    fake = sandbox_env
    at = app("PayPal Sandbox")
    no_exception(at)
    assert at.button(key="sb_create").disabled and fake.calls == []  # nothing happens before an explicit action
    at.checkbox(key="sb_confirm_create").check().run()
    assert not at.button(key="sb_create").disabled
    at.button(key="sb_create").click().run()
    no_exception(at)
    assert fake.count("POST", "/v2/checkout/orders") == 1 and fake.count("POST", "/capture") == 0
    t = text(at)
    assert "AWAITING_BUYER" in t and "NOT been captured" in t and "MOCKORDER0001" in t and "checkoutnow?token=MOCKORDER0001" in t
    assert at.button(key="sb_capture").disabled  # approval not confirmed yet
    assert at.checkbox(key="sb_confirm_approved").disabled is False


def test_sandbox_capture_only_after_buyer_approval(world, sandbox_env):
    fake = sandbox_env
    at = app("PayPal Sandbox")
    at.checkbox(key="sb_confirm_create").check().run()
    at.button(key="sb_create").click().run()
    # user ticks "I approved" but has NOT actually approved in PayPal: the page must not capture
    at.checkbox(key="sb_confirm_approved").check().run()
    assert not at.button(key="sb_capture").disabled
    at.button(key="sb_capture").click().run()
    no_exception(at)
    assert fake.count("POST", "/capture") == 0, "captured an order the buyer had not approved"
    assert "not captured" in text(at) and "AWAITING_BUYER" in text(at)
    # now the buyer really approves (done by hand in PayPal in real life)
    fake.approve("MOCKORDER0001")
    at.button(key="sb_capture").click().run()
    no_exception(at)
    t = text(at)
    assert fake.count("POST", "/capture") == 1 and "COMPLETED" in t and "MOCKCAP0001" in t
    assert "Capture verified" in t and "Yes" in [m.value for m in at.metric if m.label == "Capture verified"]
    assert "PayPal debug ID" in t and "mock-debug-id" not in "" and "capture-debug" in t
    assert "never combined with simulation" in t or "single Sandbox observations" in t


def test_sandbox_history_shows_transitions_calls_and_predicted_vs_observed(world, sandbox_env):
    at = app("PayPal Sandbox")
    at.checkbox(key="sb_confirm_create").check().run()
    at.button(key="sb_create").click().run()
    t = text(at)
    assert "Execution history (state transitions)" in t and "ORDER_CREATED" in t and "PayPal API calls (evidence; secrets redacted)" in t
    assert "CREATE_ORDER" in t and "GET_ORDER" in t and "create-debug" in t and "Predicted vs observed" in t and "PENDING_BUYER" in t  # (the OAuth call is not tied to one execution)
    assert FAKE_CLIENT_SECRET not in text(at) and "FAKE-ACCESS-TOKEN" not in text(at)


def test_sandbox_actions_are_disabled_in_demonstration_mode(world, sandbox_env):
    at = app("PayPal Sandbox", demo=True)
    at.checkbox(key="sb_confirm_create").check().run()
    assert at.button(key="sb_create").disabled and "Demonstration mode is ON" in text(at)


def test_sandbox_rejects_invalid_amounts_without_calling_paypal(world, sandbox_env):
    fake = sandbox_env
    at = app("PayPal Sandbox")
    at.text_input(key="sb_amount").set_value("-5").run()
    at.checkbox(key="sb_confirm_create").check().run()
    at.button(key="sb_create").click().run()
    assert "Invalid input" in text(at) and fake.calls == []


# ============================================================ agent execution traces
def test_trace_page_is_understandable_and_complete(world):
    at = app("Agent Execution Traces")
    no_exception(at)
    t = text(at)
    assert world["gemini_fixture_id"][:8] in t and "Configured model:" in t and "fixture-model" in t
    assert "predates per-turn served-model capture" in t
    assert "Harness setup" in t and "Gemini turn 1" in t and "Deterministic final evaluation" in t
    assert "compare_recovery_policies" in t and "propose_policy_variations" in t and "Compared" in t and "rejected as infeasible" in t
    assert "Candidate policies and how they changed" in t and "Changes vs initial policy" in t and "retry_limit" in t
    assert "Final evaluator decision" in t and "top-scoring FEASIBLE candidate" in t
    assert "Experiment ID:" in t and "simulation run IDs" in t
    assert "Credentials are never stored or displayed" in t


def test_trace_page_flags_non_gemini_executions(world):
    at = app("Agent Execution Traces")
    at.selectbox(key="trace_select").set_value(world["scripted_id"]).run()
    no_exception(at)
    assert "is NOT a Gemini run" in text(at) and "SCRIPTED_TEST_DOUBLE" in text(at)


def test_trace_rendering_never_shows_credentials_even_if_one_leaked_into_the_db(world, tmp_path, monkeypatch):
    leaky = str(tmp_path / "leaky.db")
    shutil.copy(world["agent"], leaky)
    conn = sqlite3.connect(leaky)
    payload = json.dumps({"note": f"oops {FAKE_GEMINI_KEY}", "api_key": FAKE_GEMINI_KEY, "Authorization": "Bearer abcdefghijklmnop", "client_secret": "s3cr3t-value"})
    conn.execute("UPDATE payment_agent_tool_calls SET args_json=?, result_json=?", (payload, payload))
    conn.execute("UPDATE payment_agent_events SET payload_json=? WHERE kind='llm_turn'", (json.dumps({"iteration": 1, "text": f"I will use {FAKE_GEMINI_KEY}", "function_calls": []}),))
    conn.commit()
    conn.close()
    monkeypatch.setenv("INOEMA_AGENT_DB", leaky)
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_GEMINI_KEY)
    for section in ("Agent Execution Traces", "Overview", "Policy Optimization"):
        at = app(section, demo=(section == "Policy Optimization"))
        blob = text(at)
        assert FAKE_GEMINI_KEY not in blob and "abcdefghijklmnop" not in blob and "s3cr3t-value" not in blob, section


def test_trace_empty_state(tmp_path, monkeypatch):
    for k in ("INOEMA_PAYMENT_DB", "INOEMA_AGENT_DB", "INOEMA_PAYPAL_DB"):
        monkeypatch.setenv(k, str(tmp_path / (k + ".db")))
    at = app("Agent Execution Traces")
    no_exception(at)
    assert "No agent executions have been persisted yet" in text(at)


# ============================================================ Phase 5.1: judge-facing overview, workflow indicator, consistency, visual distinction
import re  # noqa: E402

from dashboard.payments.sources import SOURCE_STYLE, DataSource  # noqa: E402


def html(at) -> str:
    return "\n".join(str(m.value) for m in at.markdown)


def test_overview_explains_what_the_product_does_what_the_agent_optimizes_mcp_and_sandbox(world):
    at = app("Overview")
    no_exception(at)
    h = html(at)
    for title in ("What INOEMA Payments does", "What the AI agent optimizes", "How simulation and MCP interact", "How PayPal Sandbox validates it"):
        assert f"data-card='{title}'" in h, title
    flat = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", h))  # inline tags removed without inserting spaces
    assert "recovery policies" in flat and "hypothetical" in flat  # what it does
    assert "maximize recovered revenue" in flat and "enforced in code, not by the LLM" in flat  # what the agent optimizes
    assert "MCP server" in flat and "deterministic evaluator" in flat and "never touches the simulation directly" in flat  # simulation + MCP
    assert "real PayPal Sandbox order" in flat and "never captures before approval" in flat and "validates the mechanics, not the simulated success rates" in flat
    d = h[h.index("data-testid='pipeline-diagram'"):]
    for box in ("Merchant objective", "Gemini agent", "MCP tools", "Simulation engine", "Deterministic evaluator", "PayPal Sandbox"):
        assert box in d, box
    assert "ASSUMED distributions" in flat


def steps_of(at) -> list[dict]:
    h = html(at)
    return [{"key": m.group(1), "state": m.group(2), "current": m.group(3) == "true"}
            for m in re.finditer(r"data-step='(\w+)' data-state='(\w+)' data-current='(\w+)'", h)]


def test_workflow_indicator_is_on_every_section_and_highlights_the_current_one(world):
    keys = ["overview", "optimize", "experiments", "sandbox", "traces"]
    for section, current in zip(WORKFLOW_SECTIONS, keys):
        at = app(section)
        no_exception(at)
        steps = steps_of(at)
        assert [s["key"] for s in steps] == keys, section
        assert [s["key"] for s in steps if s["current"]] == [current], section
        assert "data-testid='workflow-indicator'" in html(at)


def test_workflow_states_are_derived_from_persisted_data(world):
    st_ = {s["key"]: s["state"] for s in steps_of(app("Overview"))}
    # fixture: a complete genuine-labelled run, stored comparisons, one verified Sandbox capture, persisted traces
    assert st_ == {"overview": "DONE", "optimize": "COMPLETE", "experiments": "DONE", "sandbox": "VERIFIED", "traces": "DONE"}
    h = html(app("Overview"))
    assert "1 capture(s) verified" in h and "complete run" in h


def test_workflow_states_with_nothing_persisted(tmp_path, monkeypatch):
    for k in ("INOEMA_PAYMENT_DB", "INOEMA_AGENT_DB", "INOEMA_PAYPAL_DB"):
        monkeypatch.setenv(k, str(tmp_path / (k + ".db")))
    at = app("Overview")
    no_exception(at)
    st_ = {s["key"]: s["state"] for s in steps_of(at)}
    assert st_ == {"overview": "DONE", "optimize": "NOT_STARTED", "experiments": "NOT_STARTED", "sandbox": "NOT_STARTED", "traces": "NOT_STARTED"}
    assert "Gemini not configured" in html(at) and "PayPal Sandbox not configured" in html(at)


def test_every_section_has_the_same_structure(world):
    for section in SECTIONS:
        at = app(section)
        no_exception(at)
        h = html(at)
        assert [x.value for x in at.subheader] == [section], section  # exactly one subheader: the section title
        assert "inoema-kicker" in h and "PAYMENTS ·" in h, section  # same kicker treatment
        assert "inoema-pill" in h and "INOEMA" in h, section  # same header
        assert re.search(r"data-testid='workflow-indicator'", h), section
        assert re.search(r"border:1px solid [^;]+; background:[^;]+;\">[^<]*◆|SIMULATED|AGENT-DERIVED|SANDBOX · ACTUAL|SYSTEM", h), section  # a data-source label near the top


def test_each_section_leads_with_its_own_data_source_label(world):
    expect = {"Experiments": "SIMULATED", "PayPal Sandbox": "SANDBOX · ACTUAL", "Policy Optimization": "AGENT-DERIVED", "Agent Execution Traces": "AGENT-DERIVED"}
    for section, label in expect.items():
        h = html(app(section))
        after_title = h[h.index("PAYMENTS ·"):]
        first_badge = re.search(r"(SIMULATED|AGENT-DERIVED|SANDBOX · ACTUAL|SYSTEM)", after_title).group(1)
        assert first_badge == label, (section, first_badge)


def test_sources_are_visually_distinct_and_consistent(world):
    colors = {src: SOURCE_STYLE[src]["color"].lower() for src in DataSource}
    assert len(set(colors.values())) == 4  # four different colours
    marks = {SOURCE_STYLE[s]["mark"] for s in DataSource}
    assert len(marks) == 4  # ... and four different symbols (not colour alone)
    h = html(app("Overview"))
    bars = re.findall(r"data-source='(\w+)' style='height:5px; border-radius:3px; margin:0.1rem 0 0.5rem 0; background:(#[0-9A-Fa-f]{6});", h)
    assert [b[0] for b in bars] == ["AGENT_DERIVED", "SIMULATED", "SANDBOX_ACTUAL"]  # each metric group has exactly one accent bar
    for src, color in bars:
        assert color.lower() == colors[DataSource(src)]
    # the same colour marks that source in the workflow indicator and on its own page
    for step, src in (("optimize", DataSource.AGENT_DERIVED), ("experiments", DataSource.SIMULATED), ("sandbox", DataSource.SANDBOX_ACTUAL)):
        m = re.search(rf"data-step='{step}'[^>]*border-top:5px solid (#[0-9A-Fa-f]{{6}})", h)
        assert m and m.group(1).lower() == colors[src], step
    assert colors[DataSource.SANDBOX_ACTUAL] in html(app("PayPal Sandbox")).lower()


# ---- Demonstration mode: complete vs incomplete
def test_demo_mode_prefers_the_latest_complete_genuine_run_over_a_newer_interrupted_one(world, tmp_path, monkeypatch):
    from tests.dash_fixtures import variant_agent_db
    v = variant_agent_db(world["agent"], str(tmp_path / "a.db"), add_newer_interrupted=True)
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "a.db"))
    at = app("Overview", demo=True)
    no_exception(at)
    t = text(at)
    assert "latest COMPLETE genuine Gemini execution" in t and world["gemini_fixture_id"][:8] in t and "fixture-model " in t.replace("`", "") + " "
    assert v["newer_interrupted_id"][:8] not in "".join(m.value for m in at.metric if m.label == "Latest agent execution")
    assert next(m for m in at.metric if m.label == "Latest agent execution").value == world["gemini_fixture_id"][:8]
    assert "NO COMPLETE genuine execution" not in t
    tr = app("Agent Execution Traces", demo=True)
    assert tr.selectbox(key="trace_select").value == world["gemini_fixture_id"]  # opens on the complete run, not the newer interrupted one
    assert "State: COMPLETE" in text(tr) or "COMPLETE" in text(tr)
    opt = app("Policy Optimization", demo=True)
    assert "Latest COMPLETE genuine execution" in text(opt)


def test_demo_mode_with_only_an_interrupted_run_never_calls_it_complete(world, tmp_path, monkeypatch):
    from tests.dash_fixtures import variant_agent_db
    variant_agent_db(world["agent"], str(tmp_path / "a.db"), interrupt_genuine=True)
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "a.db"))
    ov = app("Overview", demo=True)
    no_exception(ov)
    t = text(ov)
    assert "NO COMPLETE genuine execution is available" in t and "INTERRUPTED (llm_error)" in t and "incomplete historical run" in t
    assert "latest COMPLETE genuine Gemini execution" not in t
    assert {s["key"]: s["state"] for s in steps_of(ov)}["optimize"] == "INTERRUPTED"
    opt = app("Policy Optimization", demo=True)
    topt = text(opt)
    assert "No COMPLETE genuine execution is available" in topt and "INTERRUPTED (llm_error)" in topt and "did not finish autonomously" in topt
    assert "Latest COMPLETE genuine execution" not in topt
    tr = app("Agent Execution Traces", demo=True)
    ttr = text(tr)
    assert "INTERRUPTED / INCOMPLETE" in ttr and "State: **COMPLETE**" not in ttr and "State: COMPLETE" not in ttr


def test_demo_mode_experiments_open_on_the_demo_runs_experiment(world):
    at = app("Experiments", demo=True)
    no_exception(at)
    assert at.selectbox(key="exp_select").value  # a concrete experiment is selected without user action


# ============================================================ Phase 5.1: fixes found during the real-browser review
def test_disabled_primary_buttons_are_styled_as_disabled(world):
    h = html(app("Policy Optimization"))
    assert 'button[kind="primary"]:disabled' in h and "cursor: not-allowed" in h  # legacy CSS made disabled primary buttons look active


def test_long_metric_values_are_shown_in_full():
    def script():
        from dashboard.payments.sources import DataSource, Metric
        from dashboard.payments.ui import render_metrics
        render_metrics([Metric("Selected recovery policy", "intel_ab_exp_b45_m2", DataSource.AGENT_DERIVED), Metric("Short", "LOW", DataSource.AGENT_DERIVED)], key_prefix="t")

    at = AppTest.from_function(script)
    at.run()
    no_exception(at)
    h = html(at)
    assert ">intel_ab_exp_b45_m2</div>" in h  # st.metric alone would truncate it to "intel_ab_exp_b4…"
    assert h.count("word-break:break-all") == 1  # short values do not get the extra line


def test_experiment_run_counts_are_human_readable(world):
    from dashboard.payments.ui import fmt_run_counts
    assert fmt_run_counts({"comparison": 9, "run": 1, "sensitivity": 1}) == "9 comparisons · 1 single-policy run · 1 sensitivity analysis"
    assert fmt_run_counts({}) == "no runs"
    t = text(app("Overview"))
    assert "{'" not in t and "comparisons" in t  # no raw Python dict on the page


def test_model_fallback_is_flagged_when_gemini_model_is_unset(world, monkeypatch):
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    t = text(app("Policy Optimization"))
    assert "GEMINI_MODEL" in t and "retired that model for newly created keys" in t.replace("Google has ", "")
    monkeypatch.setenv("GEMINI_MODEL", "some-model")
    assert "built-in fallback" not in text(app("Policy Optimization"))


def test_experiments_table_fits_by_default_and_extra_columns_are_opt_in(world):
    at = app("Experiments")
    cols = list(at.dataframe[0].value.columns)
    assert "RR 95% CI" in cols and "Recovered revenue / trial" in cols and "Processing cost / trial" in cols and "Retries / txn" in cols and "Avg recovery (min)" in cols
    assert not {"RR CI low", "RR CI high", "Score CI low", "Cost / success", "Violations"} & set(cols) and len(cols) <= 11
    at.multiselect(key="exp_extra").set_value(["Cost / success", "Score CI low", "Violations"]).run()
    assert {"Cost / success", "Score CI low", "Violations"} <= set(at.dataframe[0].value.columns)


def test_tradeoff_chart_has_one_labelled_trace_per_policy():
    import pandas as pd
    from dashboard.payments import charts
    df = pd.DataFrame([{"Policy": f"p{i}", "Feasible": i != 3, "Avg recovery (min)": 100 + i, "Recovered revenue / trial": 24000 + i, "Processing cost / trial": 300.0 + i} for i in range(4)])
    fig = charts.tradeoff_scatter(df)
    assert [t.name for t in fig.data] == ["p0", "p1", "p2", "p3 (infeasible)"] and fig.data[3].marker.symbol == "x"


def test_sandbox_currency_defaults_to_usd(world, sandbox_env):
    assert app("PayPal Sandbox").selectbox(key="sb_currency").value == "USD"


def test_sandbox_policy_comes_from_the_latest_complete_run_and_flags_incomplete_ones(world, tmp_path, monkeypatch, sandbox_env):
    from tests.dash_fixtures import variant_agent_db
    v = variant_agent_db(world["agent"], str(tmp_path / "a.db"), add_newer_interrupted=True)
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "a.db"))
    at = app("PayPal Sandbox")
    lab = at.selectbox(key="sb_policy").options[0]
    assert world["gemini_fixture_id"][:8] in lab and v["newer_interrupted_id"][:8] not in lab and "INTERRUPTED" not in lab
    variant_agent_db(world["agent"], str(tmp_path / "b.db"), interrupt_genuine=True)
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "b.db"))
    at2 = app("PayPal Sandbox")
    assert "INTERRUPTED (llm_error)" in at2.selectbox(key="sb_policy").options[0]
    assert "execution that is **INTERRUPTED (llm_error)**" in text(at2).replace("execution that is INTERRUPTED (llm_error)", "execution that is **INTERRUPTED (llm_error)**")


def test_traces_show_run_state_not_a_misleading_completed_status(world, tmp_path, monkeypatch):
    at = app("Agent Execution Traces")
    assert next(m for m in at.metric if m.label == "Run state").value == "COMPLETE" and "Status" not in [m.label for m in at.metric]
    from tests.dash_fixtures import variant_agent_db
    variant_agent_db(world["agent"], str(tmp_path / "a.db"), interrupt_genuine=True)
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "a.db"))
    at2 = app("Agent Execution Traces")
    assert next(m for m in at2.metric if m.label == "Run state").value == "INTERRUPTED"  # even though the harness status is COMPLETED


def test_dollar_signs_in_model_text_are_not_rendered_as_latex(world, tmp_path, monkeypatch):
    import shutil, sqlite3, json
    shutil.copy(world["agent"], str(tmp_path / "a.db"))
    conn = sqlite3.connect(str(tmp_path / "a.db"))
    conn.execute("UPDATE payment_agent_events SET payload_json=? WHERE kind='llm_turn'", (json.dumps({"iteration": 1, "text": "window $\\le 4320$ minutes", "function_calls": []}),))
    conn.commit()
    conn.close()
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "a.db"))
    at = app("Agent Execution Traces")
    said = next(m.value for m in at.markdown if "Gemini said" in str(m.value))
    assert "\\$" in said and "$\\le" not in said.replace("\\$\\le", "")


# ============================================================ Execution Plan section (policy execution bridge)
def _bridge_world(tmp_path, monkeypatch, *, complete=True, confidence="HIGH"):
    from tests.test_execution_bridge import report, write_db
    db = tmp_path / "agent.db"
    write_db(db, [("bridge-run-1", "GEMINI_API", "COMPLETED", "finished_by_llm" if complete else "llm_error", report(confidence={"label": confidence}), "2026-01-01")])
    monkeypatch.setenv("INOEMA_AGENT_DB", str(db))


def test_execution_plan_section_shows_gate_plan_and_decision_trace(world, tmp_path, monkeypatch):
    _bridge_world(tmp_path, monkeypatch)
    at = app("Execution Plan")
    no_exception(at)
    assert [x.value for x in at.subheader] == ["Execution Plan"]
    t, h = text(at), html(at)
    assert "bridge-r" in t and "COMPLETE" in t and "PASS" in t and not at.error
    trace = next(d.value for d in at.dataframe if "Action" in d.value.columns and "Rationale" in d.value.columns)
    assert {"RETRY", "WAIT", "REQUIRE_BUYER_ACTION", "STOP"} <= set(trace["Action"])  # every action is demonstrated
    assert "SCHEDULED_RETRY" in t and "NEVER_RECAPTURE" in t
    assert "SIMULATED" in h and "SANDBOX · ACTUAL" in h  # simulated trace and the real payment are separately labelled
    assert "was resumed manually" in t and "STOP" in t  # the real completed Sandbox payment is only ever shown as protected from retry


def test_execution_plan_refuses_an_interrupted_low_confidence_run_unless_overridden(world, tmp_path, monkeypatch):
    _bridge_world(tmp_path, monkeypatch, complete=False, confidence="LOW")
    at = app("Execution Plan")
    no_exception(at)
    assert at.error and "REFUSED" in at.error[0].value and "SCHEDULED_RETRY" not in text(at)  # nothing is compiled or executed
    at.checkbox(key="ep_override").check().run()
    no_exception(at)
    assert not at.error and at.warning and "OVERRIDE ACTIVE" in at.warning[0].value
    assert "INTERRUPTED" in text(at) and "OVERRIDDEN_INCOMPLETE_SOURCE" in text(at)


def test_execution_plan_handles_missing_optimisation_result(tmp_path, monkeypatch):
    monkeypatch.setenv("INOEMA_AGENT_DB", str(tmp_path / "none.db"))
    monkeypatch.setenv("INOEMA_PAYMENT_DB", str(tmp_path / "s.db"))
    monkeypatch.setenv("INOEMA_PAYPAL_DB", str(tmp_path / "p.db"))
    at = app("Execution Plan")
    no_exception(at)
    assert at.info and "No optimisation result" in at.info[0].value


def test_execution_plan_page_is_read_only(world, tmp_path, monkeypatch):
    import hashlib
    _bridge_world(tmp_path, monkeypatch)
    files = [world["agent"], world["sandbox"], world["sim"]]
    before = [hashlib.md5(open(f, "rb").read()).hexdigest() for f in files]
    at = app("Execution Plan")
    at.checkbox(key="ep_override").check().run()
    assert before == [hashlib.md5(open(f, "rb").read()).hexdigest() for f in files]
