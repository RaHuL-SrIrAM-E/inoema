"""Streamlit/Plotly rendering helpers for the INOEMA dashboard.

Pure presentation layer: every function here only renders data that was
already computed by agent.demo_agent / pipeline.py / the storage layer.
Nothing here runs a simulation, calls the Monte Carlo engine, or decides
anything -- it only displays what those modules already produced.

Every section is explicitly labelled as synthetic/mocked, per the
project's constraint that nothing here represents a real financial
system.
"""

from __future__ import annotations

from typing import Optional

import plotly.graph_objects as go
import streamlit as st

from agent.demo_agent import AgentRunResult
from collector.collector import OutcomeComparison
from execution.mock_paypal import MockPayoutBatch

SYNTHETIC_NOTICE = (
    "🔧 **SYNTHETIC / MOCKED ENVIRONMENT** — no real PayPal, bank, or ledger "
    "system is contacted anywhere in this demonstration."
)

RISK_COLORS = {
    "LOW": "#34D399",
    "MEDIUM": "#FBBF24",
    "HIGH": "#F87171",
    "UNKNOWN": "#9CA3AF",
}
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#E6E8EE"),
    margin=dict(l=40, r=20, t=40, b=40),
)


def _minutes_to_clock(minutes: float) -> str:
    total = round(minutes)
    hour = 16 + total // 60
    minute = total % 60
    return f"{int(hour):02d}:{int(minute):02d}"


def render_header() -> None:
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #262B3A; margin-bottom: 1.2rem;">
          <div style="display:flex; align-items:baseline; gap: 0.75rem;">
            <span style="font-size: 2.1rem; font-weight: 800; letter-spacing: 0.04em; color:#E6E8EE;">INOEMA</span>
            <span style="font-size: 0.95rem; color:#8B93A7;">Simulation Infrastructure for Autonomous Systems</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([2, 1, 1])
    with cols[0]:
        st.markdown(
            f"<span style='color:#FBBF24; font-weight:600;'>{SYNTHETIC_NOTICE}</span>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            "<span style='color:#34D399;'>&#9679;</span>&nbsp;**Engine** — Online",
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            "<span style='color:#34D399;'>&#9679;</span>&nbsp;**Mode** — Mocked",
            unsafe_allow_html=True,
        )


def render_agent_control(result: AgentRunResult, batch_size: int) -> None:
    st.subheader("Agent Control")
    initial_action = result.trace[0].action

    cols = st.columns(4)
    cols[0].metric("Objective", "Release batch safely")
    cols[1].metric("Batch Size", f"{batch_size:,}")
    release_time, release_count = initial_action.release_schedule[0]
    cols[2].metric("Initial Release Time", _minutes_to_clock(release_time))
    cols[3].metric("Initial Proposal", initial_action.label.replace("_", " ").title())

    st.caption(initial_action.rationale)

    st.markdown("**Agent Execution Trace**")
    for step in result.trace:
        decision = step.agent_decision.decision.value
        with st.expander(
            f"Step {step.step_index}: {step.action.label.replace('_', ' ').title()} — "
            f"verdict: {decision}",
            expanded=(step.step_index == result.trace[-1].step_index),
        ):
            st.write(f"**Rationale:** {step.action.rationale}")
            st.write(
                f"**Release schedule:** "
                + ", ".join(
                    f"{_minutes_to_clock(t)} × {c:,}" for t, c in step.action.release_schedule
                )
            )
            st.write(f"**Agent decision:** {step.agent_decision.rationale}")


def render_simulation_results(result: AgentRunResult) -> None:
    st.subheader("Simulation Results")
    initial_step = result.trace[0]
    revised_step = result.trace[1] if len(result.trace) > 1 else None

    st.markdown("**Initial Proposal**")
    _render_verdict_metrics(initial_step)

    if revised_step is not None:
        st.markdown("---")
        st.markdown("**Revised Proposal (agent-adjusted)**")
        st.caption(
            "Release schedule: "
            + ", ".join(
                f"{_minutes_to_clock(t)} × {c:,}"
                for t, c in revised_step.action.release_schedule
            )
        )
        _render_verdict_metrics(revised_step)

    st.markdown("**Initial vs. Revised Risk**")
    _render_risk_comparison_chart(initial_step, revised_step)


def _render_verdict_metrics(step) -> None:
    verdict = step.verdict
    mc = step.mc_result
    cols = st.columns(5)
    cols[0].metric("Monte Carlo Trials", f"{mc.n_trials:,}")
    cols[1].metric("Risk Score", f"{verdict.risk_score:.1%}")
    cols[2].metric("Confidence", f"{verdict.confidence:.1%}")
    cols[3].metric("Expected Impact", f"{verdict.expected_impact_count:,.0f} payments")
    risk_color = RISK_COLORS.get(verdict.risk_level.value, "#9CA3AF")
    cols[4].markdown(
        f"<div style='text-align:center;'>"
        f"<div style='color:#8B93A7; font-size:0.8rem;'>Risk Level / Recommendation</div>"
        f"<div style='font-size:1.3rem; font-weight:700; color:{risk_color};'>{verdict.risk_level.value}</div>"
        f"<div style='font-size:0.85rem; color:#E6E8EE;'>{verdict.decision.value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_risk_comparison_chart(initial_step, revised_step) -> None:
    labels = ["Initial Proposal"]
    risk_scores = [initial_step.verdict.risk_score]
    impact_fractions = [initial_step.verdict.expected_impact_fraction]
    if revised_step is not None:
        labels.append("Revised Proposal")
        risk_scores.append(revised_step.verdict.risk_score)
        impact_fractions.append(revised_step.verdict.expected_impact_fraction)

    fig = go.Figure()
    fig.add_bar(name="Risk Score — P(material shortfall)", x=labels, y=risk_scores, marker_color="#F87171")
    fig.add_bar(name="Expected Impact Fraction", x=labels, y=impact_fractions, marker_color="#6C8CFF")
    fig.update_layout(
        barmode="group",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_execution(batch: Optional[MockPayoutBatch]) -> None:
    st.subheader("Execution — Mock PayPal Gateway")
    if batch is None:
        st.info("No execution yet — the proposal has not been approved for execution.")
        return

    cols = st.columns(4)
    cols[0].metric("Batch ID", batch.batch_id[:12] + "…")
    cols[1].metric("Payout Items", f"{batch.batch_size:,}")
    cols[2].metric("Status", batch.status)
    cols[3].metric("Submitted At (UTC)", batch.submitted_at.split("T")[1][:8])
    st.caption(f"🔧 {batch.label} — submission label: `{batch.label}`")


def render_verification(
    comparison: Optional[OutcomeComparison], actual_outcome_row: Optional[dict]
) -> None:
    st.subheader("Independent Verification")
    if comparison is None:
        st.info("No verification yet — nothing has been executed.")
        return

    st.success("✅ Collector status: outcome independently retrieved and compared.")

    cols = st.columns(4)
    cols[0].metric("Predicted Impact (fraction)", f"{comparison.predicted_expected_impact_fraction:.1%}")
    cols[1].metric("Actual Missed Fraction", f"{comparison.actual_missed_fraction:.1%}")
    cols[2].metric("Prediction Error", f"{comparison.prediction_error:+.1%}")
    cols[3].metric("Brier Component (this run)", f"{comparison.brier_component:.4f}")
    st.caption(
        "Brier component shown is a single-sample term "
        "(predicted_risk_score − actual_shortfall_indicator)² for this one run — "
        "not a calibrated Brier score, which needs many runs. See Phase 5 (evaluation)."
    )

    if actual_outcome_row is not None:
        st.markdown("**Reality Simulator Details (hidden from the predictor)**")
        rcols = st.columns(3)
        rcols[0].metric("Day Slowdown Multiplier", f"{actual_outcome_row['day_slowdown_multiplier']:.2f}×")
        rcols[1].metric("Reality Seed", actual_outcome_row["reality_seed"])
        rcols[2].metric("Completed by Close", f"{actual_outcome_row['n_completed_by_close']:,}")

    st.markdown("**Predicted vs. Actual**")
    fig = go.Figure()
    fig.add_bar(
        name="PREDICTED (expected impact fraction)",
        x=["Executed Batch"],
        y=[comparison.predicted_expected_impact_fraction],
        marker_color="#6C8CFF",
    )
    fig.add_bar(
        name="ACTUAL (reality simulator, independent)",
        x=["Executed Batch"],
        y=[comparison.actual_missed_fraction],
        marker_color="#F87171",
    )
    fig.update_layout(
        barmode="group",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


_TIMELINE_STEPS = ["PROPOSE", "SIMULATE", "ADJUST", "RE-SIMULATE", "APPROVE", "EXECUTE", "VERIFY"]


def render_timeline(result: AgentRunResult, executed: bool, verified: bool) -> None:
    st.subheader("Agent Execution Timeline")

    was_adjusted = len(result.trace) > 1
    approved = result.final_decision.approved_for_execution

    completed = {"PROPOSE", "SIMULATE"}
    if was_adjusted:
        completed |= {"ADJUST", "RE-SIMULATE"}
    if approved:
        completed.add("APPROVE")
    if executed:
        completed.add("EXECUTE")
    if verified:
        completed.add("VERIFY")

    steps = _TIMELINE_STEPS if was_adjusted else [s for s in _TIMELINE_STEPS if s not in ("ADJUST", "RE-SIMULATE")]

    chips = []
    for step in steps:
        if step in completed:
            chips.append(f"<span style='color:#34D399; font-weight:700;'>✔ {step}</span>")
        else:
            chips.append(f"<span style='color:#4B5263;'>{step}</span>")
    st.markdown(" &nbsp;→&nbsp; ".join(chips), unsafe_allow_html=True)

    if not approved:
        st.error(
            "⛔ HUMAN INTERVENTION REQUIRED — the agent stopped before execution "
            f"({result.final_decision.rationale})"
        )
