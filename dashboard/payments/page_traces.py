"""Agent Execution Traces: a judge-readable, sanitised record of what the agent did."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.payments import data
from dashboard.payments.redaction import sanitize
from dashboard.payments.sources import DataSource
from dashboard.payments.ui import demo_mode, fmt_ts, json_expander, page_intro


def render() -> None:
    page_intro("PAYMENTS · TRACES", "Agent Execution Traces", "A step-by-step record of an autonomous optimisation: what Gemini decided, which MCP simulation tools ran, what they returned and how the "
               "deterministic evaluator chose. Credentials are never stored or displayed; tool I/O is redacted and truncated.", [DataSource.AGENT_DERIVED],
               note="trace of agent behaviour; simulation numbers inside tool outputs are ◆ SIMULATED")
    runs = data.list_agent_runs(provider="GEMINI_API" if demo_mode() else None)
    if not runs:
        st.info("No agent executions have been persisted yet. Start one in **Policy Optimization**." + (" (Demonstration mode lists genuine Gemini executions only.)" if demo_mode() else ""))
        return
    ids = [r["execution_id"] for r in runs]
    want = st.session_state.get("trace_execution_id")
    if want is None and demo_mode():
        want = (data.latest_genuine_agent_run() or {}).get("execution_id")
    idx = ids.index(want) if want in ids else 0

    def label(i: str) -> str:
        r = next(x for x in runs if x["execution_id"] == i)
        return f"{i[:8]} · {r['llm_provider']} · {r['model']} · {fmt_ts(r['started_at'])} · {r['termination_reason'] or r['status']}"

    chosen = st.selectbox("Execution", ids, index=idx, format_func=label, key="trace_select")
    trace = data.build_trace(chosen)
    run = trace["run"]

    c = st.columns(4)
    c[0].metric("Execution ID", run["execution_id"][:8], help=run["execution_id"])
    state_short = "COMPLETE" if data.is_complete(run) else ("NOT A GEMINI RUN" if run["llm_provider"] != "GEMINI_API" else "INTERRUPTED")
    c[1].metric("Run state", state_short, help=f"{data.run_state_label(run)} · harness status {run['status']} (the harness always finishes its deterministic evaluation) · termination {run['termination_reason']}")
    c[2].metric("LLM provider", run["llm_provider"])
    c[3].metric("Iterations", sum(1 for p in trace["phases"] if p["title"].startswith("Gemini turn")))
    st.markdown(f"**Configured model:** `{run['model']}` · **Served model(s):** " + (", ".join(f"`{m}`" for m in trace["served_models"]) if trace["served_models"]
                else "_not recorded — this execution predates per-turn served-model capture_") +
                f" · started {fmt_ts(run['started_at'])} · finished {fmt_ts(run['finished_at'])}")
    if run["llm_provider"] != "GEMINI_API":
        st.error(f"This execution used **{run['llm_provider']}** — it is NOT a Gemini run.")
    if run["llm_provider"] == "GEMINI_API" and run["termination_reason"] == "finished_by_llm":
        st.success("State: **COMPLETE** — Gemini finished its exploration and the deterministic evaluator made the final selection.")
    if run["termination_reason"] != "finished_by_llm":
        st.warning(f"State: **INTERRUPTED / INCOMPLETE** — termination `{run['termination_reason']}`." + (f" Error: {sanitize(run['error'])}" if run.get("error") else ""))
    ids_ = trace["experiment_ids"]
    st.markdown(f"**Experiment ID:** `{ids_['experiment_id']}` · **{len(ids_['simulation_run_ids'])} simulation run IDs** recorded")

    obj = trace["objective"]
    with st.expander("Merchant objective and enforced constraints", expanded=False):
        st.markdown(f"> {obj['text']}")
        st.json({"constraints": obj["constraints"], "weights": obj["weights"], "n_transactions": obj["n_transactions"], "provenance": obj["provenance"], "warnings": obj["warnings"]})

    st.markdown("#### Iteration history")
    for i, ph in enumerate(trace["phases"]):
        if not ph["calls"] and not ph["gemini_text"] and not ph["requested"]:
            continue
        with st.expander(f"{ph['title']}  ·  {len(ph['calls'])} tool call(s)", expanded=(i in (1,))):
            if ph["gemini_text"]:
                st.markdown("**Gemini said:** " + ph["gemini_text"].replace("$", "\\$"))  # keep Gemini's "$...$" literal (not LaTeX)
            if ph["requested"]:
                st.markdown("**Gemini requested:** " + ", ".join(f"`{n}`" for n in ph["requested"]))
            if ph["response_meta"] and ph["response_meta"].get("model_version"):
                st.caption(f"served by `{ph['response_meta']['model_version']}` · tokens {ph['response_meta'].get('total_tokens')}")
            for call in ph["calls"]:
                who = "Gemini" if call["source"] == "gemini" else "harness"
                st.markdown(f"**#{call['seq']} `{call['tool']}`** ({who}) — {call['narrative']}")
                meta = f"{call['duration_ms']:.0f} ms · {call['work_units']:,} work units" + (f" · simulation run `{call['simulation_run_id'][:8]}`" if call["simulation_run_id"] else "")
                st.caption(meta + (" · ✗ failed" if not call["ok"] else ""))
                json_expander(f"input · #{call['seq']}", call["input"])
                json_expander(f"output · #{call['seq']}", call["output"])

    st.markdown("#### Candidate policies and how they changed")
    def short(changes: list[str], n: int = 3) -> str:
        if not changes:
            return "—"
        return "; ".join(changes[:n]) + (f"  (+{len(changes) - n} more)" if len(changes) > n else "")

    cand = pd.DataFrame([{"Candidate": c["name"], "Origin": c["origin"], "Evaluations": c["evaluations"], "Ever feasible": c["ever_feasible"],
                          "Best feasible score": c["best_feasible_score"], "Changes vs initial policy": short(c["changes_vs_initial"])} for c in trace["candidates"]])
    st.dataframe(cand, hide_index=True, width="stretch", key="trace_cands", column_config={"Best feasible score": st.column_config.NumberColumn(format="%.4f")})
    with st.expander("Full parameter changes vs parent policy, and why infeasible candidates were rejected"):
        st.dataframe(pd.DataFrame([{"Candidate": c["name"], "Changes vs parent": ("; ".join(c["changes_vs_parent"]) if c["changes_vs_parent"] else "—") if c["changes_vs_parent"] is not None else "n/a (not a variation)",
                                    "All changes vs initial": "; ".join(c["changes_vs_initial"]) or "—", "Why infeasible": "; ".join(c["first_violations"]) or "—"} for c in trace["candidates"]]),
                     hide_index=True, width="stretch", key="trace_cands_detail")

    st.markdown("#### Final evaluator decision")
    rep = trace["report"]
    if rep and rep.get("selected_policy"):
        sel = rep["selected_policy"]
        st.success(f"Selected **{sel['name']}** (score {sel['metrics']['score']:+.4f}) — top-scoring FEASIBLE candidate in the deterministic fixed-seed evaluation.")
        st.markdown(f"Confidence: **{rep['confidence']['label']}** — " + "; ".join(rep["confidence"]["reasons"]))
        for r in rep.get("rejected_alternatives", []):
            st.markdown(f"- `{r['policy']}`: {r['kind'].lower().replace('_', ' ')} — " + "; ".join(r["reasons"][:2]))
    elif rep:
        st.error("No policy was selected (no candidate satisfied the hard constraints).")
    else:
        st.info("The execution did not produce a final report.")
    with st.expander("Raw persisted identifiers"):
        st.json(sanitize({"execution_id": run["execution_id"], **ids_}))
