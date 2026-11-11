"""Experiments: detailed, filterable, sortable policy comparison + sensitivity views over persisted SIMULATION results.

Uses Streamlit's native sortable dataframe (click a header to sort) plus real filter/sort widgets; AG Grid is intentionally not
used (extra dependency, no capability we need). Everything shown is read from stored results, never recomputed.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.payments import charts, data
from dashboard.payments.sources import DataSource
from dashboard.payments.ui import fmt_ts, page_intro

COLUMN_CONFIG = {  # short headers + explicit pixel widths so every required metric fits at 1440 px without horizontal scrolling
    "Rank": st.column_config.NumberColumn("#", width=36),
    "Policy": st.column_config.TextColumn("Policy", width=150),
    "Feasible": st.column_config.CheckboxColumn("Feasible", width=66, help="Satisfies every hard constraint"),
    "Score": st.column_config.NumberColumn("Score", format="%.4f", width=66, help="Deterministic objective score (mean over trials)"),
    "Score CI low": st.column_config.NumberColumn(format="%.4f"), "Score CI high": st.column_config.NumberColumn(format="%.4f"),
    "Recovery rate (%)": st.column_config.ProgressColumn("Recovery rate", format="%.1f%%", min_value=0.0, max_value=100.0, width=100, help="recovered / initially failed (pooled over trials)"),
    "RR CI low": st.column_config.NumberColumn("RR 95% CI low", format="%.3f"), "RR CI high": st.column_config.NumberColumn("RR 95% CI high", format="%.3f"),
    "RR 95% CI": st.column_config.TextColumn("Rate 95% CI", width=104, help="Wilson 95% interval on the pooled recovery rate"),
    "Score 95% CI": st.column_config.TextColumn("Score 95% CI", width=110),
    "Recovered revenue / trial": st.column_config.NumberColumn("Revenue / trial", format="%.0f", width=96, help="Mean recovered revenue per simulated trial"),
    "Avg recovery (min)": st.column_config.NumberColumn("Avg recov. (min)", format="%.0f", width=92),
    "Retries / txn": st.column_config.NumberColumn("Retries / txn", format="%.3f", width=82, help="Retry overhead: automatic retries per transaction"),
    "Unnecessary retries / trial": st.column_config.NumberColumn("Unnec. retries", format="%.0f", width=82, help="Retries of failures that cannot be fixed by retrying"),
    "Processing cost / trial": st.column_config.NumberColumn("Cost / trial", format="%.1f", width=82, help="Total processing cost per simulated trial (assumed unit costs)"),
    "Cost / success": st.column_config.NumberColumn(format="%.4f"),
    "Buyer actions / trial": st.column_config.NumberColumn(format="%.0f"),
}
SORTABLE = ["Score", "Recovery rate", "Recovered revenue / trial", "Avg recovery (min)", "Retries / txn", "Processing cost / trial", "Unnecessary retries / trial", "Rank"]


def apply_filters(df: pd.DataFrame, *, feasibility: str, min_rr: float, max_minutes: float, name: str, sort_by: str, ascending: bool) -> pd.DataFrame:
    """Pure filtering/sorting logic (unit-tested)."""
    d = df
    if feasibility == "Feasible only":
        d = d[d["Feasible"]]
    elif feasibility == "Infeasible only":
        d = d[~d["Feasible"]]
    d = d[d["Recovery rate"] >= min_rr]
    d = d[d["Avg recovery (min)"] <= max_minutes]
    if name.strip():
        d = d[d["Policy"].str.contains(name.strip(), case=False, regex=False)]
    return d.sort_values(sort_by, ascending=ascending, kind="stable") if sort_by in d.columns and not d.empty else d


def _run_label(r: dict) -> str:
    res = r["result"]
    n = len(res.get("ranking", [])) or 1
    return f"{r['kind']} · {r['requested_by']} · {n} policies · {r['n_trials']} trials · seed {r['master_seed']} · {fmt_ts(r['created_at'])}"


def render() -> None:
    page_intro("PAYMENTS · EXPERIMENTS", "Experiments", "Compare candidate recovery policies on identical simulated workloads: recovery rate, revenue, latency, cost, feasibility and sensitivity.",
               [DataSource.SIMULATED], note="stored simulation experiments (assumed distributions). Not PayPal data. The Gemini agent may have requested them; the numbers are still simulation output.")
    exps = data.list_experiments()
    if not exps:
        st.info("No experiments have been stored yet. Run an optimization in **Policy Optimization**.")
        return

    def elabel(e: dict) -> str:
        a = e["agent_run"]
        return f"{e['name']} · {e['experiment_id'][:8]} · {fmt_ts(e['created_at'])} · " + (f"agent: {a['llm_provider']} ({a['termination_reason']})" if a else "no agent trace persisted")

    ids = [e["experiment_id"] for e in exps]
    pref = data.experiment_for_run((data.latest_genuine_agent_run() or {}).get("execution_id")) if st.session_state.get("payments_demo_mode") else None
    eid = st.selectbox("Experiment", ids, index=ids.index(pref) if pref in ids else 0, format_func=lambda i: elabel(next(e for e in exps if e["experiment_id"] == i)), key="exp_select")
    exp = next(e for e in exps if e["experiment_id"] == eid)
    agent_detail = data.get_agent_run(exp["agent_run"]["execution_id"]) if exp["agent_run"] else None
    runs = data.annotate_runs(data.get_experiment_runs(eid), agent_detail)
    c = (exp["config"] or {}).get("constraints", {})
    st.caption(f"Hard constraints for this experiment: max retries {c.get('max_retries')}, no terminal retries {c.get('forbid_terminal_retries')}, "
               f"idempotency required {c.get('require_idempotency')}, buyer approval required {c.get('require_buyer_approval')}, window ≤ {c.get('max_recovery_window_minutes')} min.")

    tab_cmp, tab_sens = st.tabs(["Policy comparison", "Sensitivity analysis"])
    with tab_cmp:
        cmp_runs = [r for r in runs if r["kind"] in ("comparison", "run")]
        if not cmp_runs:
            st.info("This experiment has no stored comparisons.")
        else:
            finals = [i for i, r in enumerate(cmp_runs) if r["requested_by"] == "deterministic final evaluation"]
            default = finals[0] if finals else len(cmp_runs) - 1
            ri = st.selectbox("Simulation run", range(len(cmp_runs)), index=default, format_func=lambda i: _run_label(cmp_runs[i]), key="exp_run")
            df = pd.DataFrame(data.comparison_rows(cmp_runs[ri]["result"]))
            f1, f2, f3, f4 = st.columns([1.3, 1.6, 1.6, 1.5])
            feas = f1.radio("Feasibility", ["All", "Feasible only", "Infeasible only"], key="exp_feas")
            min_rr = f2.slider("Min recovery rate", 0.0, 1.0, 0.0, 0.01, key="exp_minrr")
            hi = float(max(1.0, round(df["Avg recovery (min)"].max() + 1))) if len(df) else 1.0
            max_min = f3.slider("Max avg recovery (min)", 0.0, hi, hi, key="exp_maxmin")
            name = f4.text_input("Policy name contains", key="exp_name")
            s1, s2 = st.columns([2, 1])
            sort_by = s1.selectbox("Sort by", SORTABLE, index=SORTABLE.index("Score"), key="exp_sort")
            asc = s2.toggle("Ascending", key="exp_asc", value=False)
            view = apply_filters(df, feasibility=feas, min_rr=min_rr, max_minutes=max_min, name=name, sort_by=sort_by, ascending=asc)
            st.caption(f"Showing {len(view)} of {len(df)} policies · click any column header to sort · {int(df['Trials'].iloc[0])} trials × {int(df['Txns / trial'].iloc[0]):,} simulated transactions")
            view = view.assign(**{"Recovery rate (%)": view["Recovery rate"] * 100})
            view = view.assign(**{"RR 95% CI": view.apply(lambda r: f"{r['RR CI low']:.3f}–{r['RR CI high']:.3f}", axis=1),
                                  "Score 95% CI": view.apply(lambda r: "—" if pd.isna(r["Score CI low"]) else f"{r['Score CI low']:.3f}–{r['Score CI high']:.3f}", axis=1)})
            core = ["Rank", "Policy", "Feasible", "Score", "Recovery rate (%)", "RR 95% CI", "Recovered revenue / trial", "Avg recovery (min)", "Retries / txn",
                    "Unnecessary retries / trial", "Processing cost / trial"]
            extras = st.multiselect("Additional columns", ["Violations", "Score 95% CI", "Cost / success", "Buyer actions / trial", "Score CI low", "Score CI high", "RR CI low", "RR CI high"], key="exp_extra")
            shown = core + extras
            st.dataframe(view[shown], hide_index=True, width="stretch", column_config=COLUMN_CONFIG, key="exp_table")
            st.download_button("Download table (CSV, all columns)", view.drop(columns=["RR 95% CI", "Score 95% CI"]).to_csv(index=False).encode(), file_name=f"policy_comparison_{eid[:8]}.csv", mime="text/csv", key="exp_csv")
            if not view.empty:
                g1, g2 = st.columns(2)
                with g1:
                    st.plotly_chart(charts.recovery_ci_chart(view), width="stretch", key="exp_ch_rr")
                with g2:
                    st.plotly_chart(charts.score_chart(view), width="stretch", key="exp_ch_score")
                st.plotly_chart(charts.tradeoff_scatter(view), width="stretch", key="exp_ch_trade")
            else:
                st.warning("No policies match the current filters.")

    with tab_sens:
        sens = [r for r in runs if r["kind"] == "sensitivity"]
        if not sens:
            st.info("This experiment has no sensitivity analyses. The agent runs one in its deterministic final phase; you can also call `run_sensitivity_analysis` over MCP.")
            return
        si = st.selectbox("Sensitivity run", range(len(sens)), format_func=lambda i: f"{sens[i]['requested_by']} · {len(sens[i]['result']['policies'])} policies · {sens[i]['n_trials']} trials · {fmt_ts(sens[i]['created_at'])}", key="exp_sens_run")
        fr = data.sensitivity_frames(sens[si]["result"])
        rob = fr["robustness"]
        a, b, c3 = st.columns(3)
        a.metric("Baseline winner", fr["baseline_winner"])
        b.metric("Winner stable in", f"{rob['winner_stable_share']:.0%}", help=f"{rob['winner_stable_points']} of {rob['n_points']} sweep points")
        c3.metric("Robustness label", rob["label"], help=rob["note"])
        st.caption("◆ SIMULATED · one-at-a-time sweeps over ASSUMED ranges; same seed at every point; all policies see identical workloads. Describes the simulation, not the real world.")
        st.markdown("**Parameters ranked by how much they affect the recommendation**")
        summ = pd.DataFrame([{"Parameter": s["parameter"], "Winner flips": s["n_winner_changes"], "Ranking changes": s["n_ranking_changes"], "Smallest margin": s["min_margin"],
                              "Flip conditions": "; ".join(f"{f['value']:g}→{f['new_winner']}" for f in s["flip_conditions"]) or "—"} for s in fr["summary"]])
        st.dataframe(summ, hide_index=True, width="stretch", key="exp_sens_summary", column_config={"Smallest margin": st.column_config.NumberColumn(format="%.4f")})
        st.plotly_chart(charts.sensitivity_heatmap(fr), width="stretch", key="exp_sens_heat")
        params = [s["parameter"] for s in fr["summary"]]
        p = st.selectbox("Parameter to plot", params, key="exp_sens_param")
        st.plotly_chart(charts.sensitivity_line(fr, p), width="stretch", key="exp_sens_line")
        with st.expander("Recorded assumptions and parameter configurations"):
            st.json(fr["assumptions"], expanded=False)
