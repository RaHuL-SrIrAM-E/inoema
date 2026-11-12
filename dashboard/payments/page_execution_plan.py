"""Execution Plan: the selected policy flowing into a deterministic, auditable decision trace. Read-only: no PayPal, Gemini or MCP call, nothing written."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.payments import data, paths
from dashboard.payments.sources import DataSource
from dashboard.payments.ui import page_intro, source_banner
from engine.payments.models import FailureCategory
from execution_bridge.executor import REAL_SANDBOX_READONLY, FailureEvent, PaymentState, PolicyExecutor
from execution_bridge.plan import BridgeError, PlanRefused, compile_plan, load_persisted_result, validate_result
from execution_bridge.scenarios import SCENARIOS


def render() -> None:
    page_intro("PAYMENTS · EXECUTION PLAN", "Execution Plan",
               "The agent's selected policy is gated, compiled into a versioned execution plan, and applied by a deterministic executor to payment states. "
               "No model decides here.", [DataSource.AGENT_DERIVED],
               note="gate and plan come from the persisted optimisation result; the executor is deterministic code")
    try:
        res = load_persisted_result(db_path=paths.agent_db_path())
    except BridgeError as e:
        st.info(f"No optimisation result is available to plan from: {e}. Run **Policy Optimization** first.")
        return
    st.markdown(f"**Source optimisation run:** `{res.execution_id[:8]}` · {res.provider} · " + ("**COMPLETE**" if res.is_complete else f"**INTERRUPTED / INCOMPLETE** (termination `{res.termination_reason}`)"))
    override = st.checkbox("Demonstration override: allow an incomplete or low-confidence source run", value=False, key="ep_override",
                           help="Waives ONLY the completeness and confidence checks. Feasibility and constraint checks can never be waived. The plan and every trace line are labelled as an override.")
    gate = validate_result(res, allow_incomplete_source=override)
    st.markdown("**Gate: is this result safe to execute?**")
    st.dataframe(pd.DataFrame([{"Check": c.name, "Result": "PASS" if c.ok else ("WAIVED (override)" if gate.override and c.waivable else "FAIL"), "Detail": c.detail} for c in gate.checks]),
                 hide_index=True, width="stretch", key="ep_gate")
    try:
        plan = compile_plan(res, allow_incomplete_source=override)
    except PlanRefused as e:
        st.error("**Execution plan REFUSED.** " + " · ".join(e.problems) + " — nothing is compiled or executed. Tick the demonstration override to see what the plan would look like.")
        return
    if plan.status != "APPROVED":
        st.warning("**OVERRIDE ACTIVE — demonstration only.** " + " ".join(plan.warnings))
    st.markdown(f"**Plan** `{plan.plan_id}` · `{plan.plan_version}` · policy `{plan.policy_name}` (hash `{plan.policy_hash}`) · source run `{plan.source_run_id[:8]}` · status **{plan.status}**")
    c = st.columns(4)
    c[0].metric("Retry limit", plan.retry_limit)
    c[1].metric("Retry delays (min)", ", ".join(f"{d:g}" for d in plan.retry_delays_minutes) or "—")
    c[2].metric("Recovery window (min)", f"{plan.maximum_recovery_window_minutes:g}")
    c[3].metric("Buyer action", "Required" if plan.buyer_action_required else "Not used", help=f"timeout {plan.buyer_action_timeout_minutes:g} min · alternate funding {plan.alternate_funding}")
    with st.expander("Failure-category rules compiled from the policy"):
        st.dataframe(pd.DataFrame([{"Category": k, "Class": v["recovery_class"], "First action": v["first_action"], "Reason": v["reason"], "First delay (min)": v["first_delay_minutes"]}
                                   for k, v in plan.failure_rules.items()]), hide_index=True, width="stretch", key="ep_rules")

    source_banner(DataSource.SIMULATED, "SIMULATED payment states chosen for this demonstration, not outcomes produced by PayPal")
    ex = PolicyExecutor(plan)
    for _, state, event in SCENARIOS:
        ex.step(state, event)
    st.markdown("**Decision trace** (deterministic; the same inputs always give the same decisions)")
    st.dataframe(pd.DataFrame([{"#": s.index, "Scenario": SCENARIOS[s.index - 1][0], "Order": s.order_status, "Retries": s.retries_made, "Event": s.event or "—", "Action": s.action,
                                "Delay (min)": s.delay_minutes or None, "Rule": s.rule, "Rationale": s.rationale} for s in ex.trace]), hide_index=True, width="stretch", key="ep_trace")
    st.caption(f"Each line carries source run `{plan.source_run_id[:8]}`, policy `{plan.policy_name}` (`{plan.policy_hash}`), plan `{plan.plan_id}`. Persist with `python -m scripts.execution_bridge_demo --persist`.")

    source_banner(DataSource.SANDBOX_ACTUAL, "the real manually-resumed Sandbox payment, read from the evidence database; no PayPal call is made")
    done = next((e for e in data.list_sandbox_executions() if e["status"] == "COMPLETED" and e["capture_id"]), None)
    if done is None:
        st.info("No completed Sandbox payment is recorded, so there is nothing to protect from a retry.")
        return
    real = PaymentState(done["execution_id"][:8], "COMPLETED", capture_id=done["capture_id"], source=REAL_SANDBOX_READONLY)
    d = PolicyExecutor(plan).decide(real, FailureEvent(FailureCategory.TRANSIENT_NETWORK_ERROR))
    st.markdown(f"Real payment `{done['order_id']}` (capture `{done['capture_id']}`, {done['amount']} {done['currency']}) was resumed manually, **not** by this plan. If a retry event arrived for it, the executor decides "
                f"**{d.action.value}** (`{d.rule}`): {d.rationale}.")
