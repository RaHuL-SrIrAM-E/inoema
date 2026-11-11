"""Read-only data access for the Payments workspace (no Streamlit imports).

Reads three SEPARATE stores and keeps them separate:
  * simulation experiments  (payment_sim_*,      provenance SIMULATED)      -> storage.payment_experiments
  * agent executions        (payment_agent_*)                                -> storage.payment_agent_runs
  * PayPal Sandbox evidence (sandbox_*,          environment PAYPAL_SANDBOX) -> paypal.persistence
Databases are opened read-only and are never created by viewing the dashboard.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Any, Optional

from dashboard.payments import paths
from dashboard.payments.redaction import sanitize
from dashboard.payments.sources import DataSource, Metric
from storage import payment_experiments as sim_store


# ------------------------------------------------------------------ low level
def _ro(path: str) -> Optional[sqlite3.Connection]:
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _rows(conn: Optional[sqlite3.Connection], sql: str, args: tuple = ()) -> list[dict]:
    if conn is None:
        return []
    try:
        return [dict(r) for r in conn.execute(sql, args)]
    except sqlite3.OperationalError:  # table not created yet
        return []


def _j(s: Optional[str]) -> Any:
    try:
        return json.loads(s) if s else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------ agent runs / traces
def list_agent_runs(provider: Optional[str] = None, limit: int = 100) -> list[dict]:
    conn = _ro(paths.agent_db_path())
    try:
        rows = _rows(conn, "SELECT execution_id, started_at, finished_at, status, llm_provider, model, objective_text, experiment_id, "
                           "termination_reason, error FROM payment_agent_runs ORDER BY started_at DESC LIMIT ?", (limit,))
    finally:
        conn and conn.close()
    return [r for r in rows if provider is None or r["llm_provider"] == provider]


def get_agent_run(execution_id: str) -> Optional[dict]:
    conn = _ro(paths.agent_db_path())
    if conn is None:
        return None
    try:
        run = next(iter(_rows(conn, "SELECT * FROM payment_agent_runs WHERE execution_id=?", (execution_id,))), None)
        if run is None:
            return None
        calls = _rows(conn, "SELECT * FROM payment_agent_tool_calls WHERE execution_id=? ORDER BY seq", (execution_id,))
        events = _rows(conn, "SELECT * FROM payment_agent_events WHERE execution_id=? ORDER BY seq", (execution_id,))
        cands = _rows(conn, "SELECT * FROM payment_agent_candidates WHERE execution_id=? ORDER BY name", (execution_id,))
    finally:
        conn.close()
    run["objective"] = _j(run.pop("objective_json"))
    run["config"] = _j(run.pop("config_json"))
    run["report"] = _j(run.pop("final_report_json"))
    for c in calls:
        c["args"], c["result"], c["ok"] = _j(c.pop("args_json")), _j(c.pop("result_json")), bool(c["ok"])
    for e in events:
        e["payload"] = _j(e.pop("payload_json"))
    for c in cands:
        c["definition"], c["observations"] = _j(c.pop("definition_json")), _j(c.pop("observations_json")) or []
    return {"run": run, "tool_calls": calls, "events": events, "candidates": cands}


def served_models(detail: dict) -> list[str]:
    """Models that actually served the Gemini turns (per-turn `model_version`); empty if the run predates that capture."""
    out = {((e["payload"] or {}).get("response") or {}).get("model_version") for e in detail["events"] if e["kind"] == "llm_turn"}
    return sorted(m for m in out if m)


def definition_diff(base: dict, other: dict) -> list[str]:
    keys = ("retry_limit", "retry_intervals", "exponential_backoff", "backoff_base_minutes", "backoff_multiplier", "backoff_max_interval_minutes",
            "maximum_recovery_window", "buyer_action_timeout", "request_buyer_action", "alternate_funding_action", "terminal_failure_handling",
            "classification_aware", "idempotent_retries")
    return [f"{k}: {base.get(k)} → {other.get(k)}" for k in keys if base.get(k) != other.get(k)]


def describe_tool_call(tool: str, args: dict, result: Any, ok: bool) -> str:
    """One plain-English line per tool call, for a reader who has never seen the code."""
    if not ok:
        return f"`{tool}` failed: {str(result)[:160]}"
    r = result if isinstance(result, dict) else {}
    if tool == "create_payment_experiment":
        c = (r.get("config") or {}).get("constraints", {})
        return f"Created the experiment and enforced the merchant's hard constraints (max retries {c.get('max_retries')}, no terminal retries: {c.get('forbid_terminal_retries')})."
    if tool == "generate_payment_scenarios":
        return "Generated simulation scenario(s): " + ", ".join(f"{s['name']} ({s['n_transactions']:,} txns)" for s in r.get("scenarios", []))
    if tool == "run_payment_simulation":
        e = r.get("evaluation", {})
        return f"Simulated policy `{(r.get('policy_definition') or {}).get('name')}`: feasible={e.get('feasible')}, score {e.get('score_mean', 0):+.3f}."
    if tool == "compare_recovery_policies":
        rk = r.get("ranking", [])
        top = f"; best `{rk[0]['policy']}` ({'feasible' if rk[0]['feasible'] else 'INFEASIBLE'}, score {rk[0]['score_mean']:+.3f})" if rk else ""
        n_inf = sum(not x["feasible"] for x in rk)
        return f"Compared {len(rk)} policies on identical simulated workloads ({r.get('n_trials')} trials × {r.get('n_transactions'):,} txns){top}; {n_inf} rejected as infeasible."
    if tool == "run_sensitivity_analysis":
        ro = r.get("robustness", {})
        return (f"Sensitivity analysis over {ro.get('n_points')} assumption variations: winner stable in {ro.get('winner_stable_share', 0):.0%} "
                f"({ro.get('label')}); most sensitive: {', '.join(r.get('most_sensitive_parameters') or []) or 'none'}.")
    if tool == "propose_policy_variations":
        return f"Asked the deterministic neighbour generator for variations of `{args.get('policy_name')}` → {len(result or [])} candidates."
    if tool == "finish_optimization":
        return f"Declared exploration finished; shortlist {args.get('shortlist')}."
    return f"Called `{tool}`."


def build_trace(execution_id: str) -> Optional[dict]:
    """Structured, judge-readable trace of one agent execution (already sanitised for display)."""
    detail = get_agent_run(execution_id)
    if detail is None:
        return None
    run, calls, events = detail["run"], detail["tool_calls"], detail["events"]
    turns = []
    llm_events = {e["payload"]["iteration"]: e["payload"] for e in events if e["kind"] == "llm_turn" and e["payload"]}
    by_iter: dict[int, list[dict]] = {}
    for c in calls:
        by_iter.setdefault(c["iteration"], []).append(c)

    def view(c: dict) -> dict:
        run_id = c["result"].get("run_id") if isinstance(c["result"], dict) else None
        return {"seq": c["seq"], "source": c["source"], "tool": c["tool"], "ok": c["ok"], "duration_ms": c["duration_ms"], "work_units": c["work_units"],
                "narrative": describe_tool_call(c["tool"], c["args"] or {}, c["result"], c["ok"]), "simulation_run_id": run_id,
                "input": sanitize(c["args"]), "output": sanitize(c["result"])}

    phases = [("Harness setup (deterministic code)", by_iter.get(0, []), None)]
    for it in sorted(k for k in llm_events):
        ev = llm_events[it]
        phases.append((f"Gemini turn {it}", by_iter.get(it, []), ev))
    phases.append(("Deterministic final evaluation (fixed seed)", by_iter.get(-1, []), None))
    for title, cs, ev in phases:
        turns.append({"title": title, "gemini_text": sanitize(ev.get("text")) if ev else None,
                      "requested": [f["name"] for f in (ev or {}).get("function_calls", [])], "response_meta": (ev or {}).get("response"),
                      "calls": [view(c) for c in cs]})
    cands = {c["name"]: c for c in detail["candidates"]}
    init = next((c for c in cands.values() if c["origin"] == "baseline" and (run.get("report") or {}).get("initial_policy", {}).get("name") == c["name"]), None)
    cand_rows = []
    for c in cands.values():
        parent = cands.get(c["name"].split("~")[0]) if "~" in c["name"] else None
        best = max((o["score_mean"] for o in c["observations"] if o["feasible"]), default=None)
        cand_rows.append({"name": c["name"], "origin": c["origin"], "evaluations": len(c["observations"]),
                          "ever_feasible": any(o["feasible"] for o in c["observations"]), "best_feasible_score": best,
                          "changes_vs_initial": definition_diff(init["definition"], c["definition"]) if init else [],
                          "changes_vs_parent": definition_diff(parent["definition"], c["definition"]) if parent else None,
                          "first_violations": next((o["violations"][:2] for o in c["observations"] if not o["feasible"]), [])})
    return {"run": {k: run[k] for k in ("execution_id", "started_at", "finished_at", "status", "llm_provider", "model", "experiment_id",
                                        "termination_reason", "error")},
            "served_models": served_models(detail), "objective": sanitize(run["objective"]), "report": sanitize(run["report"]),
            "phases": turns, "candidates": cand_rows,
            "experiment_ids": {"experiment_id": run["experiment_id"], "simulation_run_ids": [c["result"]["run_id"] for c in calls
                                                                                         if isinstance(c["result"], dict) and c["result"].get("run_id")]}}


def is_complete(run: dict) -> bool:
    """A COMPLETE genuine execution: real Gemini provider, finished normally, and the model itself called finish_optimization.
    Anything else (llm_error, max_iterations, stopped early, failed, scripted double) is NOT complete."""
    return (run.get("llm_provider") == "GEMINI_API" and run.get("status") == "COMPLETED"
            and run.get("termination_reason") == "finished_by_llm")


def run_state_label(run: dict) -> str:
    if run.get("llm_provider") != "GEMINI_API":
        return f"NOT A GEMINI RUN ({run.get('llm_provider')})"
    if is_complete(run):
        return "COMPLETE"
    return f"INTERRUPTED ({run.get('termination_reason') or run.get('status')})"


def latest_genuine_agent_run() -> Optional[dict]:
    """Demonstration mode: the most recent COMPLETE genuine Gemini execution; if none exists, the most recent genuine execution,
    explicitly flagged INCOMPLETE. A scripted double or an interrupted run is never reported as complete."""
    runs = list_agent_runs(provider="GEMINI_API")
    if not runs:
        return None
    done = [r for r in runs if is_complete(r)]
    pick = (done or runs)[0]
    return {**pick, "complete": bool(done), "completed_autonomously": bool(done), "state": run_state_label(pick)}


def experiment_for_run(execution_id: Optional[str]) -> Optional[str]:
    run = next((r for r in list_agent_runs() if r["execution_id"] == execution_id), None)
    return run["experiment_id"] if run else None


# ------------------------------------------------------------------ simulation experiments
def list_experiments() -> list[dict]:
    conn = _ro(paths.sim_db_path())
    exps = _rows(conn, "SELECT experiment_id, name, config_json, created_at FROM payment_sim_experiments ORDER BY created_at DESC")
    counts = {}
    for r in _rows(conn, "SELECT experiment_id, kind, COUNT(*) n FROM payment_sim_runs GROUP BY experiment_id, kind"):
        counts.setdefault(r["experiment_id"], {})[r["kind"]] = r["n"]
    nscen = {r["experiment_id"]: r["n"] for r in _rows(conn, "SELECT experiment_id, COUNT(*) n FROM payment_sim_scenarios GROUP BY experiment_id")}
    conn and conn.close()
    linked = {r["experiment_id"]: r for r in reversed(list_agent_runs()) if r["experiment_id"]}  # newest run wins
    out = []
    for e in exps:
        a = linked.get(e["experiment_id"])
        out.append({"experiment_id": e["experiment_id"], "name": e["name"], "created_at": e["created_at"], "config": _j(e["config_json"]),
                    "n_scenarios": nscen.get(e["experiment_id"], 0), "runs_by_kind": counts.get(e["experiment_id"], {}),
                    "agent_run": ({k: a[k] for k in ("execution_id", "llm_provider", "model", "status", "termination_reason", "started_at")} if a else None)})
    return out


def get_experiment_runs(experiment_id: str) -> list[dict]:
    conn = _ro(paths.sim_db_path())
    if conn is None:
        return []
    try:
        return sim_store.list_runs(conn, experiment_id)
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def annotate_runs(runs: list[dict], agent_detail: Optional[dict]) -> list[dict]:
    """Label each stored simulation run with who requested it (Gemini / harness / final evaluation) when an agent trace is linked."""
    meta = {}
    if agent_detail:
        for c in agent_detail["tool_calls"]:
            rid = c["result"].get("run_id") if isinstance(c["result"], dict) else None
            if rid:
                meta[rid] = c
    out = []
    for r in runs:
        c = meta.get(r["run_id"])
        if c is None:
            who = "no agent trace persisted" if agent_detail is None else "unlinked"
        elif c["iteration"] < 0:
            who = "deterministic final evaluation"
        elif c["source"] == "gemini":
            who = f"Gemini turn {c['iteration']}"
        else:
            who = "harness setup"
        out.append({**r, "requested_by": who})
    return out


def comparison_rows(result: dict) -> list[dict]:
    """Flatten a stored `compare_recovery_policies` (or single `run_payment_simulation`) result into one row per policy.
    All numbers are copied from the stored result; nothing is recomputed."""
    rows: list[dict] = []
    if "ranking" in result:
        ranking = {r["policy"]: r for r in result["ranking"]}
        pols = result["policies"]
        n_tx = result.get("n_transactions")
    else:  # single run
        name = result["policy_definition"]["name"]
        e = result["evaluation"]
        ranking = {name: {"rank": 1, "policy": name, "feasible": e["feasible"], "score_mean": e["score_mean"], "score_ci95": e["score_ci95"],
                          "violations": e["violations"]}}
        pols = {name: {"definition": result["policy_definition"], "summary": result["summary"]}}
        n_tx = result["summary"]["per_trial_means"]["total_transactions"]
    for name, rk in ranking.items():
        s = pols[name]["summary"]
        pm, rr = s["per_trial_means"], s["recovery_rate"]
        ci = rk.get("score_ci95") or [None, None]
        rows.append({
            "Rank": rk.get("rank"), "Policy": name, "Feasible": bool(rk["feasible"]), "Score": rk["score_mean"], "Score CI low": ci[0], "Score CI high": ci[1],
            "Recovery rate": rr["value"], "RR CI low": rr["ci95_wilson"][0], "RR CI high": rr["ci95_wilson"][1],
            "Recovered revenue / trial": pm["recovered_revenue"], "Avg recovery (min)": s["avg_recovery_time_minutes"]["value"],
            "Retries / txn": s["avg_retries_per_transaction"]["mean"], "Unnecessary retries / trial": pm["unnecessary_retries"],
            "Processing cost / trial": pm["total_processing_cost"], "Cost / success": s["cost_per_successful_transaction"]["value"],
            "Buyer actions / trial": pm["buyer_actions_requested"], "Violations": "; ".join(rk.get("violations", [])[:3]) + (" …" if len(rk.get("violations", [])) > 3 else ""),
            "n_violations": len(rk.get("violations", [])), "Trials": s["n_trials"], "Txns / trial": n_tx})
    return rows


def sensitivity_frames(result: dict) -> dict[str, Any]:
    """Tabular views of a stored sensitivity result."""
    pts = []
    for p in result["points"]:
        for r in p["ranking"]:
            pts.append({"parameter": p["parameter"], "value": p["value"], "policy": r["policy"], "score": r["score_mean"], "feasible": r["feasible"],
                        "recovery_rate": r["recovery_rate"], "rr_low": r["recovery_rate_ci95"][0], "rr_high": r["recovery_rate_ci95"][1],
                        "winner": p["winner"], "winner_changed": p["winner_changed"]})
    base = [{"parameter": "(baseline)", "value": None, "policy": r["policy"], "score": r["score_mean"], "feasible": r["feasible"],
             "recovery_rate": r["recovery_rate"], "rr_low": r["recovery_rate_ci95"][0], "rr_high": r["recovery_rate_ci95"][1],
             "winner": result["baseline"]["winner"], "winner_changed": False} for r in result["baseline"]["ranking"]]
    return {"points": pts, "baseline": base, "summary": result["parameter_summary"], "robustness": result["robustness"],
            "assumptions": result.get("assumptions", {}), "baseline_winner": result["baseline"]["winner"]}


# ------------------------------------------------------------------ selected-policy metrics (from the agent's own stored results)
def selected_policy_view(detail: dict) -> Optional[dict]:
    """Metrics for the agent's selected policy, taken from the stored FINAL deterministic comparison (never recomputed)."""
    report = detail["run"].get("report")
    sel = (report or {}).get("selected_policy")
    if not sel:
        return None
    final = next((c for c in detail["tool_calls"] if c["tool"] == "compare_recovery_policies" and c["iteration"] < 0 and c["ok"]), None)
    view = {"name": sel["name"], "definition": sel["definition"], "feasible": True, "score": None, "recovery_rate": None, "rr_ci": None,
            "recovered_revenue": None, "avg_recovery_min": None, "n_trials": None, "n_transactions": None, "scenario": "baseline",
            "confidence": (report.get("confidence") or {}).get("label")}
    if final and sel["name"] in final["result"].get("policies", {}):
        r = final["result"]
        s = r["policies"][sel["name"]]["summary"]
        row = next(x for x in r["ranking"] if x["policy"] == sel["name"])
        view.update(feasible=row["feasible"], score=row["score_mean"], recovery_rate=s["recovery_rate"]["value"], rr_ci=s["recovery_rate"]["ci95_wilson"],
                    recovered_revenue=s["per_trial_means"]["recovered_revenue"], avg_recovery_min=s["avg_recovery_time_minutes"]["value"],
                    n_trials=r["n_trials"], n_transactions=r["n_transactions"])
    else:
        row = next((x for x in (report.get("final_evaluation") or {}).get("ranking", []) if x["policy"] == sel["name"]), None)
        if row:
            view.update(feasible=row["feasible"], score=row["score"], recovery_rate=row["recovery_rate"], rr_ci=row["recovery_rate_ci95"],
                        avg_recovery_min=row["avg_recovery_minutes"], n_trials=(report["final_evaluation"] or {}).get("n_trials"),
                        n_transactions=(report["final_evaluation"] or {}).get("n_transactions"))
    return view


# ------------------------------------------------------------------ sandbox
def list_sandbox_executions() -> list[dict]:
    conn = _ro(paths.sandbox_db_path())
    try:
        return _rows(conn, "SELECT execution_id, experiment_id, policy_name, reference_id, status, order_id, capture_id, approval_url, amount, currency, "
                           "created_at, updated_at FROM sandbox_executions ORDER BY created_at DESC")
    finally:
        conn and conn.close()


def sandbox_detail(execution_id: str) -> Optional[dict]:
    conn = _ro(paths.sandbox_db_path())
    if conn is None:
        return None
    try:
        ex = next(iter(_rows(conn, "SELECT * FROM sandbox_executions WHERE execution_id=?", (execution_id,))), None)
        if ex is None:
            return None
        ex.pop("policy_json", None)
        trans = _rows(conn, "SELECT seq, ts, from_state, to_state, trigger, detail_json FROM sandbox_transitions WHERE execution_id=? ORDER BY seq", (execution_id,))
        calls = _rows(conn, "SELECT ts, operation, method, url, request_id, status, paypal_debug_id, latency_ms, error FROM sandbox_api_calls WHERE execution_id=? ORDER BY id", (execution_id,))
        recs = _rows(conn, "SELECT * FROM sandbox_validation_records WHERE execution_id=? ORDER BY rowid", (execution_id,))
    finally:
        conn.close()
    for t in trans:
        t["detail"] = sanitize(_j(t.pop("detail_json")))
    for r in recs:
        r["predicted_vs_observed"] = sanitize(_j(r.pop("predicted_vs_observed_json")))
    ex["last_error"] = sanitize(_j(ex.pop("last_error_json", None)))
    return {"execution": ex, "transitions": trans, "api_calls": calls, "validation_records": recs}


def sandbox_summary() -> dict:
    conn = _ro(paths.sandbox_db_path())
    try:
        ex = _rows(conn, "SELECT status, COUNT(*) n FROM sandbox_executions GROUP BY status")
        calls = _rows(conn, "SELECT COUNT(*) n, MAX(ts) last_ts FROM sandbox_api_calls")
        last = _rows(conn, "SELECT order_id, capture_id, updated_at FROM sandbox_executions ORDER BY updated_at DESC LIMIT 1")
        dbg = _rows(conn, "SELECT paypal_debug_id FROM sandbox_api_calls WHERE paypal_debug_id IS NOT NULL ORDER BY id DESC LIMIT 1")
    finally:
        conn and conn.close()
    return {"by_status": {r["status"]: r["n"] for r in ex}, "n_api_calls": (calls[0]["n"] if calls else 0), "last_call": (calls[0]["last_ts"] if calls else None),
            "last_execution": (last[0] if last else None), "last_debug_id": (dbg[0]["paypal_debug_id"] if dbg else None)}


# ------------------------------------------------------------------ overview (every metric carries its source)
def overview_metrics(execution_id: Optional[str] = None) -> dict[str, list[Metric]]:
    """Grouped metrics. Groups are never merged and no metric mixes sources. `execution_id` pins the agent execution the
    agent-derived and per-policy simulated metrics are taken from (Demonstration mode passes the latest COMPLETE genuine run);
    without it the newest execution of any kind is used and its state (COMPLETE / INTERRUPTED / not Gemini) is shown."""
    exps = list_experiments()
    runs = list_agent_runs()
    latest_run = next((r for r in runs if r["execution_id"] == execution_id), None) if execution_id else (runs[0] if runs else None)
    detail = get_agent_run(latest_run["execution_id"]) if latest_run else None
    sel = selected_policy_view(detail) if detail else None
    n_runs = sum(sum(e["runs_by_kind"].values()) for e in exps)
    sim: list[Metric] = [
        Metric("Total experiments", len(exps), DataSource.SIMULATED, f"{n_runs} stored simulation runs"),
        Metric("Recovery rate (selected policy)", None if not sel or sel["recovery_rate"] is None else f"{sel['recovery_rate']:.1%}", DataSource.SIMULATED,
               "" if not sel or not sel["rr_ci"] else f"95% CI [{sel['rr_ci'][0]:.1%}, {sel['rr_ci'][1]:.1%}] · simulation sampling error only"),
        Metric("Recovered revenue / trial", None if not sel or sel["recovered_revenue"] is None else f"{sel['recovered_revenue']:,.0f}", DataSource.SIMULATED,
               "" if not sel or not sel["n_transactions"] else f"mean per trial of {sel['n_transactions']:,} simulated txns (currency units)"),
        Metric("Avg recovery latency", None if not sel or sel["avg_recovery_min"] is None else f"{sel['avg_recovery_min']:.0f} min", DataSource.SIMULATED,
               "mean over recovered transactions"),
    ]
    agent: list[Metric] = [
        Metric("Latest agent execution", None if not latest_run else latest_run["execution_id"][:8], DataSource.AGENT_DERIVED,
               "" if not latest_run else f"{latest_run['llm_provider']} · {latest_run['model']} · {run_state_label(latest_run)}"),
        Metric("Selected recovery policy", None if not sel else sel["name"], DataSource.AGENT_DERIVED, "chosen by the deterministic evaluator"),
        Metric("Policy feasibility", None if not sel else ("Feasible" if sel["feasible"] else "INFEASIBLE"), DataSource.AGENT_DERIVED, "under the merchant's hard constraints"),
        Metric("Recommendation confidence", None if not sel else sel["confidence"], DataSource.AGENT_DERIVED, "simulation-only; not real-world confidence"),
    ]
    sb = sandbox_summary()
    sandbox: list[Metric] = [
        Metric("Sandbox executions", sum(sb["by_status"].values()), DataSource.SANDBOX_ACTUAL, ", ".join(f"{k}: {v}" for k, v in sb["by_status"].items()) or "none recorded"),
        Metric("Captures completed", sb["by_status"].get("COMPLETED", 0), DataSource.SANDBOX_ACTUAL, "verified by GET capture"),
        Metric("Last PayPal order id", (sb["last_execution"] or {}).get("order_id"), DataSource.SANDBOX_ACTUAL, "real Sandbox order"),
        Metric("Last PayPal debug id", sb["last_debug_id"], DataSource.SANDBOX_ACTUAL, f"{sb['n_api_calls']} recorded API calls"),
    ]
    return {"simulated": sim, "agent": agent, "sandbox": sandbox,
            "_context": [Metric("latest_experiment", exps[0] if exps else None, DataSource.SIMULATED), Metric("selected", sel, DataSource.AGENT_DERIVED)]}


# ------------------------------------------------------------------ end-to-end workflow indicator
def workflow_steps(*, gemini_configured: bool, paypal_configured: bool) -> list[dict]:
    """The five sections as an end-to-end pipeline, each with a status derived from persisted data (never invented)."""
    runs = list_agent_runs(provider="GEMINI_API")
    genuine = latest_genuine_agent_run()
    exps = list_experiments()
    n_cmp = sum(e["runs_by_kind"].get("comparison", 0) for e in exps)
    sb = sandbox_summary()
    if genuine is None:
        agent_state = ("NOT_STARTED", "no genuine Gemini execution yet" if gemini_configured else "Gemini not configured")
    elif genuine["complete"]:
        agent_state = ("COMPLETE", f"complete run {genuine['execution_id'][:8]}")
    else:
        agent_state = ("INTERRUPTED", f"run {genuine['execution_id'][:8]} interrupted ({genuine['termination_reason']})")
    verified = sb["by_status"].get("COMPLETED", 0)
    awaiting = sb["by_status"].get("AWAITING_BUYER", 0)
    sandbox_state = (("VERIFIED", f"{verified} capture(s) verified") if verified else ("IN_PROGRESS", f"{awaiting} awaiting buyer approval") if awaiting
                     else ("NOT_STARTED", "no Sandbox execution yet" if paypal_configured else "PayPal Sandbox not configured"))
    return [
        {"key": "overview", "section": "Overview", "title": "Understand", "does": "What the product does and what is running", "source": DataSource.SYSTEM,
         "state": ("DONE", "")},
        {"key": "optimize", "section": "Policy Optimization", "title": "Optimize", "does": "Merchant objective → Gemini plans experiments → MCP → simulation → deterministic selection",
         "source": DataSource.AGENT_DERIVED, "state": agent_state},
        {"key": "experiments", "section": "Experiments", "title": "Compare", "does": "Inspect simulated policies, scores, feasibility and sensitivity",
         "source": DataSource.SIMULATED, "state": (("DONE", f"{len(exps)} experiment(s), {n_cmp} comparison(s)") if n_cmp else ("NOT_STARTED", "no comparisons stored"))},
        {"key": "sandbox", "section": "PayPal Sandbox", "title": "Validate", "does": "Run the selected policy against a real Sandbox order: approve → capture → verify",
         "source": DataSource.SANDBOX_ACTUAL, "state": sandbox_state},
        {"key": "traces", "section": "Agent Execution Traces", "title": "Audit", "does": "Every agent decision, MCP call and evaluator verdict",
         "source": DataSource.AGENT_DERIVED, "state": (("DONE", f"{len(runs)} genuine execution trace(s)") if runs else ("NOT_STARTED", "no genuine execution traces"))},
    ]
