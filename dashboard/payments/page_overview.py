"""Overview: what the product is, how the pieces fit, where the user is in the end-to-end workflow, component status and headline
metrics. Every metric carries its data source and groups are never mixed."""
from __future__ import annotations

import os

import streamlit as st

from dashboard.payments import data, status
from dashboard.payments.sources import SOURCE_STYLE, DataSource
from dashboard.payments.ui import (BORDER, PRIMARY_TEXT, SURFACE, demo_mode, fmt_run_counts, fmt_ts, goto, page_intro, render_metrics, render_pipeline_diagram,
                                   render_source_legend, render_workflow_indicator, source_banner, status_pill)

CARDS = [
    ("What INOEMA Payments does", DataSource.SYSTEM,
     "When a payment fails a merchant must choose: retry now, retry later, ask the buyer to act, or give up. INOEMA Payments evaluates whole "
     "<b>recovery policies</b> across thousands of <i>hypothetical</i> transactions — before any real payment is touched — and recommends one."),
    ("What the AI agent optimizes", DataSource.AGENT_DERIVED,
     "A Gemini agent proposes and refines candidate policies to <b>maximize recovered revenue</b> while minimizing recovery cost, latency and "
     "unnecessary retries — subject to <b>hard constraints</b> (e.g. max retries, never auto-retry terminal failures) that are enforced in code, not by the LLM."),
    ("How simulation and MCP interact", DataSource.SIMULATED,
     "Gemini never touches the simulation directly: it calls tools on an <b>MCP server</b> that runs seeded, reproducible Monte Carlo simulations. "
     "A <b>deterministic evaluator</b> — not the LLM — scores candidates, rejects infeasible ones and selects the winner; sensitivity analysis shows how the choice changes if the assumptions are wrong."),
    ("How PayPal Sandbox validates it", DataSource.SANDBOX_ACTUAL,
     "The selected policy is applied to a <b>real PayPal Sandbox order</b>: create → buyer approves → capture → verify. The adapter never captures before approval, "
     "uses idempotency keys and never auto-retries buyer-action or terminal failures. Sandbox validates the <i>mechanics</i>, not the simulated success rates."),
]


def _autocheck() -> bool:
    return os.environ.get("INOEMA_DASHBOARD_AUTOCHECK", "1") != "0"


def _service_checks(force: bool = False) -> dict:
    ss = st.session_state
    if force or "svc" not in ss:
        ss["svc"] = {"engine": status.check_simulation_engine(), "gemini": status.gemini_status(), "paypal": status.paypal_status(), "mcp": None}
    if ss["svc"]["mcp"] is None and (_autocheck() or force):
        with st.spinner("Checking MCP server (starting it and discovering tools)…"):
            ss["svc"]["mcp"] = status.check_mcp_server()
    return ss["svc"]


def _explainer() -> None:
    cols = st.columns(4)
    for col, (title, src, body) in zip(cols, CARDS):
        c = SOURCE_STYLE[src]["color"]
        with col:
            st.markdown(f"<div data-card='{title}' style='height:100%; border:1px solid {BORDER}; border-top:5px solid {c}; border-radius:10px; background:{SURFACE}; padding:0.7rem 0.8rem;'>"
                        f"<div style='font-size:0.68rem; font-weight:700; letter-spacing:0.05em; color:{c};'>{SOURCE_STYLE[src]['mark']} {SOURCE_STYLE[src]['label']}</div>"
                        f"<div style='font-weight:700; color:{PRIMARY_TEXT}; margin:0.15rem 0 0.35rem 0;'>{title}</div>"
                        f"<div style='font-size:0.86rem; line-height:1.45;'>{body}</div></div>", unsafe_allow_html=True)


def render(steps: list[dict]) -> None:
    page_intro("PAYMENTS · OVERVIEW", "Overview", "Autonomous, simulation-backed optimization of merchant payment-recovery policies — validated against the PayPal Sandbox.", [DataSource.SYSTEM],
               note="Cards and the diagram describe the product; the metrics below carry their own source label.")
    _explainer()
    st.markdown("#### How the pieces fit")
    render_pipeline_diagram()
    st.caption("All simulated figures use ASSUMED distributions — they are not PayPal data and not real-world performance. Colours are consistent everywhere: "
               "teal = simulated, deep green = agent-derived, coral = actual Sandbox, grey = system.")
    st.markdown("#### End-to-end workflow")
    render_workflow_indicator(steps, "Overview", detailed=True)
    nav = st.columns(5)
    for col, step in zip(nav, steps):
        if step["section"] != "Overview":
            col.button(f"{step['title']} →", key=f"ov_go_{step['key']}", on_click=goto, args=(step["section"],))
    st.divider()

    last_genuine = data.latest_genuine_agent_run()
    pin = last_genuine["execution_id"] if (demo_mode() and last_genuine) else None
    if demo_mode():
        if last_genuine and last_genuine["complete"]:
            st.info(f"**Demonstration mode** — persisted history. Showing the latest COMPLETE genuine Gemini execution `{last_genuine['execution_id'][:8]}` · "
                    f"model `{last_genuine['model']}` · started {fmt_ts(last_genuine['started_at'])}.")
        elif last_genuine:
            st.warning(f"**Demonstration mode — NO COMPLETE genuine execution is available.** Showing the latest genuine execution `{last_genuine['execution_id'][:8]}` "
                       f"(model `{last_genuine['model']}`, started {fmt_ts(last_genuine['started_at'])}), which is **{last_genuine['state']}**: Gemini never called "
                       f"`finish_optimization`. Its numbers come from the deterministic evaluator over the candidates explored before it stopped. "
                       f"Treat it as an incomplete historical run, not a completed autonomous optimisation.")
        else:
            st.warning("Demonstration mode: no genuine Gemini execution has been persisted yet, so there is nothing to show.")
    hist = last_genuine["started_at"] if pin else None

    st.markdown("#### Component status")
    svc = _service_checks()
    cols = st.columns(4)
    mcp = svc["mcp"] or status.ServiceStatus("MCP server", "NOT_CHECKED", "Click “Check MCP server” (starts the server and discovers tools)")
    for col, s in zip(cols, (svc["engine"], mcp, svc["gemini"], svc["paypal"])):
        with col:
            status_pill(s.name, s.state, s.detail)
    b1, b2, b3, b4 = st.columns(4)
    if b1.button("Re-check services", key="ov_recheck"):
        _service_checks(force=True)
        st.rerun()
    if b2.button("Check MCP server", key="ov_check_mcp", disabled=svc["mcp"] is not None and _autocheck()):
        svc["mcp"] = status.check_mcp_server()
        st.rerun()
    if b3.button("Test Gemini (1 API request)", key="ov_probe_gemini", help="Spends one request of your Gemini quota."):
        svc["gemini_probe"] = status.probe_gemini()
    if b4.button("Test PayPal OAuth (Sandbox)", key="ov_probe_paypal", help="One real OAuth call to the PayPal Sandbox."):
        svc["paypal_probe"] = status.probe_paypal()
    for k in ("gemini_probe", "paypal_probe"):
        if k in svc:
            p = svc[k]
            (st.success if p.state == "OK" else st.error)(f"{p.name}: {p.detail}")
    st.caption("○ SYSTEM · statuses describe local components, not payment results. Gemini and PayPal are not probed automatically (quota / external calls).")
    st.divider()

    m = data.overview_metrics(execution_id=pin)
    exps = data.list_experiments()
    latest = exps[0] if exps else None
    sel = m["_context"][1].value
    st.markdown("#### Latest experiment")
    if latest:
        a = latest["agent_run"]
        st.markdown(f"**{latest['name']}** · `{latest['experiment_id'][:8]}` · created {fmt_ts(latest['created_at'])} · {latest['n_scenarios']} scenario(s) · "
                    f"{fmt_run_counts(latest['runs_by_kind'])} · " + (f"agent trace: `{a['execution_id'][:8]}` ({a['llm_provider']}, {a['termination_reason']})" if a else "no agent trace persisted"))
    else:
        st.info("No experiments yet. Start one in **Policy Optimization**.")

    st.markdown("#### Selected recovery policy & recommendation")
    source_banner(DataSource.AGENT_DERIVED, "chosen by the deterministic evaluator after the Gemini agent explored candidates", historical_ts=hist)
    render_metrics(m["agent"], key_prefix="ov_agent")
    if sel:
        with st.expander("Selected policy definition"):
            st.json(sel["definition"])

    st.markdown("#### Policy performance")
    source_banner(DataSource.SIMULATED, "simulated workloads under ASSUMED distributions — not real-world or PayPal performance", historical_ts=hist)
    render_metrics(m["simulated"], key_prefix="ov_sim")

    st.markdown("#### PayPal Sandbox (actual API observations)")
    source_banner(DataSource.SANDBOX_ACTUAL, "real Sandbox calls only; empty until you run the Sandbox workflow. Never combined with simulation.")
    render_metrics(m["sandbox"], key_prefix="ov_sb")
    render_source_legend()
