"""INOEMA — Simulation Infrastructure for Autonomous Systems.

Streamlit dashboard for the end-to-end demo: agent proposal -> Monte Carlo
simulation -> adjustment -> mock execution -> independent verification.

Run with:
    streamlit run dashboard/app.py

This module is presentation + sequencing only. It does not duplicate any
simulation or orchestration logic -- it calls agent.demo_agent.run_agent
and pipeline.run_pipeline exactly as they are, and only renders what they
return. All data produced anywhere in this demo is SYNTHETIC / MOCKED.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from agent.demo_agent import AgentConfig, BATCH_SIZE, run_agent
from dashboard.components import (
    render_agent_control,
    render_execution,
    render_header,
    render_simulation_results,
    render_timeline,
    render_verification,
)
from dashboard.developer_integration import render_developer_integration
from dashboard.payments.workspace import render_payments_workspace
from execution.mock_paypal import get_batch
from pipeline import run_pipeline
from storage.database import fetch_actual_outcome, init_db

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "inoema.db")


def _run_full_demo(agent_config: AgentConfig, batch_size: int, reality_seed: int, db_path: str) -> dict:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    with st.status("Running INOEMA simulation...", expanded=True) as status:
        st.write("**PROPOSE** — building initial proposal (full batch, 18:30 release).")
        st.write("**SIMULATE** — running Monte Carlo trials through the discrete-event engine...")
        result = run_agent(agent_config, batch_size=batch_size)

        initial_verdict = result.trace[0].verdict
        st.write(
            f"&nbsp;&nbsp;↳ initial risk_score = `{initial_verdict.risk_score:.3f}`, "
            f"decision = `{initial_verdict.decision.value}`"
        )

        if len(result.trace) > 1:
            st.write("**ADJUST** — splitting the batch into two earlier releases (16:00 / 17:15).")
            st.write("**RE-SIMULATE** — running Monte Carlo trials on the revised proposal...")
            revised_verdict = result.trace[1].verdict
            st.write(
                f"&nbsp;&nbsp;↳ revised risk_score = `{revised_verdict.risk_score:.3f}`, "
                f"decision = `{revised_verdict.decision.value}`"
            )

        batch = None
        comparison = None
        actual_outcome_row = None
        predicted_impact_count = result.trace[-1].verdict.expected_impact_count

        if result.final_decision.approved_for_execution:
            st.write("**APPROVE** — proposal cleared the agent's risk/confidence thresholds.")
            st.write("**EXECUTE** — submitting approved batch to the mock PayPal gateway...")
            comparison = run_pipeline(result, reality_seed=reality_seed, db_path=db_path)
            batch = get_batch(comparison.batch_id)
            st.write(f"&nbsp;&nbsp;↳ mock batch `{comparison.batch_id[:12]}…` submitted.")
            st.write("**VERIFY** — independent reality simulator + collector comparing outcomes...")

            conn = init_db(db_path)
            actual_outcome_row = fetch_actual_outcome(conn, comparison.batch_id)
            conn.close()
            actual_missed_count = batch.batch_size - actual_outcome_row["n_completed_by_close"]
            st.write(
                f"&nbsp;&nbsp;↳ actual unreconciled = `{actual_missed_count}` payments "
                f"(predicted `{round(predicted_impact_count)}`)"
            )
            status.update(label="INOEMA simulation complete", state="complete")
        else:
            st.write(f"**{result.final_decision.rationale}**")
            status.update(label="Simulation halted — human intervention required", state="error")

    return {
        "result": result,
        "batch_size": batch_size,
        "batch": batch,
        "comparison": comparison,
        "actual_outcome_row": actual_outcome_row,
        "predicted_impact_count": predicted_impact_count,
    }


PAYMENTS, LEGACY = "INOEMA Payments", "Simulation Control Center"
WORKSPACES = [PAYMENTS, LEGACY]  # Payments first: it is the default workspace


def _default_workspace() -> str:
    """INOEMA Payments is the default workspace. The original Simulation Control Center remains available from the sidebar, and can be
    opened directly with ?workspace=simulation or INOEMA_DEFAULT_WORKSPACE=simulation."""
    q = st.query_params.get("workspace", os.environ.get("INOEMA_DEFAULT_WORKSPACE", "payments"))
    return LEGACY if str(q).lower().startswith("sim") else PAYMENTS


def main() -> None:
    st.set_page_config(
        page_title="INOEMA",
        layout="wide",
    )
    if "workspace" not in st.session_state:
        st.session_state["workspace"] = _default_workspace()
    with st.sidebar:
        st.radio("Workspace", WORKSPACES, key="workspace")
    if st.session_state["workspace"] == PAYMENTS:
        render_payments_workspace()
        return
    render_header()

    with st.sidebar:
        st.markdown("### Run Configuration")
        st.caption("SCENARIO")
        batch_size = st.number_input("Batch size", min_value=10, max_value=50_000, value=BATCH_SIZE, step=100)
        n_trials = st.slider("Monte Carlo trials", min_value=50, max_value=500, value=200, step=10)
        st.divider()
        st.caption("REPRODUCIBILITY")
        master_seed = st.number_input("Agent master seed", min_value=0, value=0, step=1)
        reality_seed = st.number_input("Reality seed (independent)", min_value=0, value=11, step=1)
        st.divider()
        st.caption("APPROVAL GATE")
        risk_threshold = st.slider("Execution risk threshold", 0.0, 0.5, 0.05, 0.01)
        min_confidence = st.slider("Minimum confidence", 0.0, 1.0, 0.8, 0.05)
        st.caption("All runs are fully reproducible for a given seed.")

    tab_control, tab_dev = st.tabs(["Simulation Control Center", "Developer Integration"])

    with tab_control:
        st.markdown("")
        run_clicked = st.button("Run INOEMA Simulation", type="primary", use_container_width=True)

        if run_clicked:
            agent_config = AgentConfig(
                risk_threshold=risk_threshold,
                min_confidence=min_confidence,
                n_trials=n_trials,
                master_seed=int(master_seed),
            )
            st.session_state["demo"] = _run_full_demo(agent_config, int(batch_size), int(reality_seed), DB_PATH)

        demo = st.session_state.get("demo")
        if demo is None:
            st.info("Configure parameters in the sidebar and click **Run INOEMA Simulation** to begin.")
        else:
            st.divider()
            render_agent_control(demo["result"], demo["batch_size"])
            st.divider()
            render_simulation_results(demo["result"])
            st.divider()
            render_execution(demo["batch"])
            st.divider()
            render_verification(
                demo["comparison"],
                demo["actual_outcome_row"],
                batch=demo["batch"],
                predicted_impact_count=demo["predicted_impact_count"],
            )
            st.divider()
            render_timeline(demo["result"], executed=demo["batch"] is not None, verified=demo["comparison"] is not None)

    with tab_dev:
        render_developer_integration(st.session_state.get("demo"))


main()
