"""Policy Optimization: configure the merchant objective and launch the REAL Gemini agent (via MCP) in the background.

No scripted/mock model is ever used here. Without credentials the start button is disabled and setup instructions are shown.
In Demonstration mode live execution is disabled and the last genuine persisted execution is displayed instead.
"""
from __future__ import annotations

import os
import time
from datetime import datetime

import streamlit as st

from agent.payments.models import DEFAULT_GEMINI_MODEL, MissingCredentialsError
from dashboard.payments import data, objective, status
from dashboard.payments.errors import classify_error
from dashboard.payments.reports import render_decision_report
from dashboard.payments.runner import JobBusy, get_manager
from dashboard.payments.sources import DataSource
from dashboard.payments.ui import demo_mode, fmt_ts, goto, page_intro, render_error, source_banner
from engine.payments.scenarios import PRESETS


def _form() -> objective.MerchantForm:
    st.markdown("#### 1 · Merchant objective")
    a, b = st.columns(2)
    with a:
        n = st.number_input("Transactions per simulated trial", 100, 200_000, 10_000, 500, key="opt_n", help="Workload size the agent will simulate.")
        preset = st.selectbox("Workload scenario", list(PRESETS), format_func=lambda k: f"{k} — {PRESETS[k]}", key="opt_preset")
        custom = st.checkbox("Customise order-value distribution (lognormal)", key="opt_custom_amount")
        med = sig = None
        if custom:
            c1, c2 = st.columns(2)
            med = c1.number_input("Median order value", 1.0, 100_000.0, 45.0, 5.0, key="opt_amount_median")
            sig = c2.number_input("Sigma (spread)", 0.05, 3.0, 0.9, 0.05, key="opt_amount_sigma")
    with b:
        max_r = st.slider("Maximum automatic retries", 0, 8, 2, key="opt_max_retries")
        window = st.number_input("Maximum recovery window (hours)", 0.5, 336.0, 72.0, 1.0, key="opt_window_h")
        lat = st.slider("Recovery latency weight", 0.0, 2.0, 0.1, 0.05, key="opt_w_latency", help="Higher = the merchant cares more about fast recovery (revenue stays the primary objective, weight 1.0).")
        cost = st.slider("Recovery cost weight", 0.0, 2.0, 0.2, 0.05, key="opt_w_cost")
        unn = st.slider("Unnecessary-attempts weight", 0.0, 2.0, 0.2, 0.05, key="opt_w_unn")
    st.markdown("**Hard constraints** (a policy violating any is rejected regardless of score)")
    h1, h2, h3 = st.columns(3)
    forbid = h1.checkbox("Never auto-retry terminal failures", True, key="opt_forbid_terminal")
    idem = h2.checkbox("Retries must be idempotent", True, key="opt_idem")
    buyer = h3.checkbox("Never auto-retry buyer-action failures", True, key="opt_buyer_approval")
    notes = st.text_area("Additional guidance for the agent (optional)", key="opt_notes", help="Free text for planning. Constraint-like phrases that the parser does not recognise are NOT enforced and are listed below.")
    return objective.MerchantForm(n_transactions=int(n), scenario_preset=preset, amount_median=med, amount_sigma=sig, max_retries=int(max_r), recovery_window_hours=float(window),
                                  forbid_terminal_retries=forbid, require_idempotency=idem, require_buyer_approval=buyer, latency_weight=float(lat), cost_weight=float(cost),
                                  unnecessary_weight=float(unn), extra_notes=notes)


def _job_panel(job_id: str) -> None:
    job = get_manager().get(job_id)
    if job is None:
        return
    elapsed = time.time() - datetime.fromisoformat(job.started_at).timestamp()
    if job.running:
        st.info(f"⏳ **Agent running** · {elapsed:.0f}s elapsed · job `{job.job_id}` — Gemini is planning experiments and calling MCP simulation tools. This usually takes 1–4 minutes.")
    live = [r for r in data.list_agent_runs() if r["started_at"] >= job.started_at]
    exec_id = job.execution_id or (live[0]["execution_id"] if live else None)
    if exec_id:
        detail = data.get_agent_run(exec_id)
        if detail:
            calls = detail["tool_calls"]
            st.markdown(f"**Live execution trace** · `{exec_id[:8]}` · {len(calls)} tool calls so far")
            rows = [{"#": c["seq"], "Turn": "setup" if c["iteration"] == 0 else "final" if c["iteration"] < 0 else c["iteration"], "By": c["source"], "MCP tool": c["tool"],
                     "What happened": data.describe_tool_call(c["tool"], c["args"] or {}, c["result"], c["ok"]), "Work units": c["work_units"]} for c in calls]
            st.dataframe(rows, hide_index=True, width="stretch", key=f"live_trace_{job_id}_{len(rows)}")
    with st.expander("Agent progress log (sanitised)", expanded=job.running):
        st.code("\n".join(job.lines()[-80:]) or "(waiting for the first message…)", language="text")
    if st.session_state.get("opt_last_state") == "RUNNING" and not job.running:
        st.session_state["opt_last_state"] = job.state
        st.rerun()
    st.session_state["opt_last_state"] = job.state


def render() -> None:
    page_intro("PAYMENTS · POLICY OPTIMIZATION", "Policy Optimization", "Set the merchant's objective and constraints, then let the real Gemini agent explore policies through MCP simulation tools; "
               "a deterministic evaluator makes the final selection.", [DataSource.AGENT_DERIVED], note="the agent's decisions and the evaluator's selection; the simulation results it produces are ◆ SIMULATED")
    mgr = get_manager()
    cur_job = mgr.get(st.session_state.get("opt_job_id", "")) or mgr.active()

    if demo_mode():
        st.info("**Demonstration mode is ON** — live optimisation is disabled. Showing persisted, genuine history. Switch it off in the sidebar to run the agent.")
        last = data.latest_genuine_agent_run()
        if not last:
            st.warning("No genuine Gemini execution has been persisted yet.")
            return
        if last["complete"]:
            st.success(f"Latest COMPLETE genuine execution: `{last['execution_id'][:8]}` · model `{last['model']}` · started {fmt_ts(last['started_at'])}.")
        else:
            st.warning(f"**No COMPLETE genuine execution is available.** Showing the latest genuine execution `{last['execution_id'][:8]}` · model `{last['model']}` · "
                       f"started {fmt_ts(last['started_at'])}, which is **{last['state']}** — it is an incomplete historical run, not a finished optimisation.")
        rep = data.get_agent_run(last["execution_id"])["run"]["report"]
        render_decision_report(rep, key="demo_rep", historical_ts=last["started_at"])
        return

    form = _form()
    problems = objective.validate(form)
    spec = objective.preview(form)
    st.markdown("#### 2 · What the evaluator will enforce")
    st.caption("Built by the agent's own parser from your inputs — these constraints are authoritative; the LLM cannot drop them.")
    c = spec.constraints_payload()
    st.markdown(" · ".join(f"**{k.replace('_', ' ')}** = {v}" for k, v in c.items()))
    st.caption(f"Objective weights → revenue 1.0 (primary), cost {spec.weight_cost}, latency {spec.weight_latency}, unnecessary attempts {spec.weight_unnecessary}")
    for w in spec.warnings:
        st.warning(w)
    for p in problems:
        st.error(p)
    with st.expander("Objective text sent to Gemini"):
        st.code(spec.text, language="text")

    st.markdown("#### 3 · Agent settings")
    a, b, c3, d = st.columns(4)
    iters = a.slider("Max Gemini turns", 3, 15, 8, key="opt_iters")
    trials = b.slider("Trials per experiment", 3, 30, 10, key="opt_trials")
    ftrials = c3.slider("Final-evaluation trials", 5, 40, 20, key="opt_final_trials")
    env_model = os.environ.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL
    model = d.text_input("Gemini model", env_model, key="opt_model", help="Defaults to GEMINI_MODEL. The dashboard never silently switches models.")
    if not os.environ.get("GEMINI_MODEL"):
        st.warning(f"`GEMINI_MODEL` is not set, so the field above shows the code's built-in fallback (`{DEFAULT_GEMINI_MODEL}`). Google has retired that model for newly created keys: "
                   "set `GEMINI_MODEL` to a model your key can use before starting. The dashboard never switches models for you.")
    st.caption(f"This uses about **{iters + 1} Gemini requests** (turns + one explanation). Free-tier keys allow only a few requests per day per model. "
               f"The agent talks to the simulation exclusively through the MCP server.")

    gem = status.gemini_status()
    can_run = gem.state != "NOT_CONFIGURED"
    if not can_run:
        render_error(classify_error(MissingCredentialsError("GEMINI_API_KEY is not set")), show_demo_button=True)
    busy = bool(cur_job and cur_job.running)
    confirm = st.checkbox("I understand this makes real Gemini API calls and spends API quota", key="opt_confirm")
    start = st.button("Start Gemini optimization agent", type="primary", key="opt_start", disabled=not (can_run and confirm and not problems and not busy))
    if busy:
        st.caption("An optimisation is already running.")
    if start:
        try:
            job = mgr.start(spec.text, objective.to_overrides(form), max_iterations=iters, exploration_trials=trials, final_trials=ftrials,
                            model=(model if model != env_model else None))
            st.session_state["opt_job_id"], st.session_state["opt_last_state"] = job.job_id, "RUNNING"
            cur_job = job
        except MissingCredentialsError as e:
            render_error(classify_error(e), show_demo_button=True)
        except JobBusy as e:
            st.warning(str(e))

    st.markdown("#### 4 · Execution")
    cur_job = mgr.get(st.session_state.get("opt_job_id", "")) or cur_job
    if cur_job is None:
        st.caption("No optimisation has been started in this session.")
    else:
        panel = st.fragment(run_every=2 if cur_job.running else None)(_job_panel)
        panel(cur_job.job_id)
        if not cur_job.running:
            if cur_job.error:
                render_error(cur_job.error, show_demo_button=cur_job.error.kind == "QUOTA_EXHAUSTED")
                if cur_job.error.kind == "QUOTA_EXHAUSTED":
                    last = data.latest_genuine_agent_run()
                    if last:
                        st.info(f"Last genuine execution: `{last['execution_id'][:8]}` · model `{last['model']}` · {fmt_ts(last['started_at'])} — available in Demonstration mode.")
            if cur_job.report:
                st.markdown("#### 5 · Result")
                render_decision_report(cur_job.report, key=f"job_{cur_job.job_id}")
