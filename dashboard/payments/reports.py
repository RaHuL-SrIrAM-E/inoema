"""Rendering of a persisted/returned agent DecisionReport (dict). Every number shown is read from the report; nothing is recomputed."""
from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import streamlit as st

from dashboard.payments import charts
from dashboard.payments.errors import classify_error
from dashboard.payments.redaction import sanitize
from dashboard.payments.sources import DataSource
from dashboard.payments.ui import fmt_ts, goto, json_expander, source_banner


def final_eval_frame(report: dict) -> pd.DataFrame:
    fe = (report.get("final_evaluation") or {}).get("ranking", [])
    return pd.DataFrame([{"Rank": r["rank"], "Policy": r["policy"], "Feasible": r["feasible"], "Score": r["score"], "Recovery rate": r["recovery_rate"],
                          "Recovery 95% CI": f"{r['recovery_rate_ci95'][0]:.3f}–{r['recovery_rate_ci95'][1]:.3f}", "Avg recovery (min)": r["avg_recovery_minutes"],
                          "Unnecessary retries / trial": r["unnecessary_retries"], "Violations": "; ".join(r["violations"][:2])} for r in fe])


def render_decision_report(report: dict, *, key: str, historical_ts: Optional[str] = None, model_line: str = "") -> None:
    if not report:
        st.info("No decision report is available for this execution.")
        return
    llm = report.get("llm", {})
    conf = report.get("confidence", {})
    sel = report.get("selected_policy")
    term = report.get("termination_reason")

    if term != "finished_by_llm":
        st.warning(f"**This execution did not finish autonomously** (termination: `{term}`). The deterministic evaluator still ran on the candidates "
                   f"explored so far, but the model never called `finish_optimization`. Treat it as an interrupted run.")
    source_banner(DataSource.AGENT_DERIVED, f"Execution `{report['execution_id'][:8]}` · {llm.get('provider')} · configured model `{llm.get('model')}` {model_line}", historical_ts=historical_ts)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Agent turns", report["llm_activity"]["turns"], help="Gemini reasoning turns")
    c2.metric("Gemini function calls", report["llm_activity"]["function_calls"])
    c3.metric("Simulation experiments", report["experiments"]["simulation_calls"], help=f"{report['experiments']['by_gemini']} requested by Gemini, {report['experiments']['by_harness']} by the harness")
    c4.metric("Candidate policies", len(report["candidates_explored"]))
    st.caption("◆ SIMULATED experiments requested by the agent · counts come from the persisted execution")

    st.markdown("##### Deterministic final selection")
    if sel:
        m = sel["metrics"]
        a, b, c = st.columns(3)
        a.metric("Selected policy", sel["name"])
        b.metric("Objective score", f"{m['score']:+.4f}")
        c.metric("Feasible", "Yes" if m["feasible"] else "No")
        st.caption("✦ AGENT-DERIVED · picked by the deterministic evaluator (top-scoring FEASIBLE candidate under a fixed seed), not by the LLM")
        with st.expander("Selected policy definition"):
            st.json(sel["definition"], expanded=True)
    else:
        st.error("No candidate satisfied the hard constraints, so no policy was selected.")

    st.markdown("##### Final evaluation (fixed seed, identical workloads) — ◆ SIMULATED")
    fe = final_eval_frame(report)
    if not fe.empty:
        fe_chart = fe.assign(**{"Score CI low": float("nan"), "Score CI high": float("nan")})  # CIs are not part of the report's ranking table
        st.dataframe(fe, hide_index=True, width="stretch", key=f"{key}_fe",
                     column_config={"Score": st.column_config.NumberColumn(format="%.4f"), "Recovery rate": st.column_config.NumberColumn(format="%.3f"),
                                    "Avg recovery (min)": st.column_config.NumberColumn(format="%.0f")})
        st.plotly_chart(charts.score_chart(fe_chart), width="stretch", key=f"{key}_score")

    rej = report.get("rejected_alternatives", [])
    if rej:
        st.markdown("##### Why the alternatives were rejected")
        for r in rej:
            if r["kind"] == "INFEASIBLE":
                st.markdown(f"- `{r['policy']}` — **infeasible**: " + "; ".join(r["reasons"][:3]) + (" …" if len(r["reasons"]) > 3 else ""))
            else:
                gap = r.get("score_gap_to_winner")
                sig = "statistically separable" if r.get("gap_significant") else "**not statistically separable** from the winner"
                st.markdown(f"- `{r['policy']}` — feasible but scored lower (gap {gap:+.4f}; {sig}).")

    st.markdown("##### Sensitivity & recommendation confidence")
    sf = report.get("sensitivity_findings", {})
    if sf.get("label") == "NOT_ASSESSED":
        st.info("Sensitivity analysis was not assessed for this execution.")
    else:
        st.markdown(f"Winner stable in **{sf['winner_stable_share']:.0%}** of {sf['n_points']} assumption variations → **{sf['label']}**. "
                    f"Most sensitive parameters: {', '.join(f'`{p}`' for p in sf.get('most_sensitive_parameters') or []) or 'none flip the winner'}.")
        for f in sf.get("flip_conditions", [])[:5]:
            st.markdown(f"- flips to `{f['new_winner']}` when **{f['parameter']} = {f['value']}**")
    st.markdown(f"Overall confidence: **{conf.get('label')}** — " + "; ".join(conf.get("reasons", [])))
    cc = st.columns(4)
    cc[0].metric("Repeatability", {True: "Verified", False: "FAILED", None: "n/a"}[conf["repeatability"].get("verified")])
    cc[1].metric("Statistical", conf["statistical_uncertainty"].get("label", "n/a"))
    cc[2].metric("Assumption sens.", conf["assumption_sensitivity"].get("label", "n/a"))
    cc[3].metric("Scenario robustness", conf["scenario_robustness"].get("label", "n/a"))
    st.warning(conf["disclaimer"])

    st.markdown("##### Candidate policies explored")
    cand = pd.DataFrame([{"Candidate": c["name"], "Origin": c["origin"], "Evaluations": c["n_observations"], "Ever feasible": c["ever_feasible"],
                          "Best feasible score": c["best_feasible_score"]} for c in report["candidates_explored"]])
    st.dataframe(cand, hide_index=True, width="stretch", key=f"{key}_cand",
                 column_config={"Best feasible score": st.column_config.NumberColumn(format="%.4f")})

    if report.get("llm_explanation"):
        st.markdown(f"##### Decision explanation — ✦ written by {llm.get('provider')} (not used for selection)")
        st.markdown(sanitize(report["llm_explanation"]).replace("$", "\\$"))  # literal dollars, not LaTeX
    for n in report.get("notes", []):
        if "error" in n.lower():
            info = classify_error(n)
            st.caption(f"Note: the agent's exploration ended with an API error — **{info.title}**." + (f" Suggested wait: {info.retry_after}." if info.retry_after else ""))
            with st.expander("Raw error text (sanitised)"):
                st.code(sanitize(n, max_str=1500), language="text")
    with st.expander("Assumptions and limitations"):
        for a in report.get("assumptions_and_limitations", []):
            st.markdown(f"- {a}")
    st.button("Open the full execution trace", key=f"{key}_trace", on_click=goto, args=("Agent Execution Traces",), kwargs={"trace_execution_id": report["execution_id"]})
