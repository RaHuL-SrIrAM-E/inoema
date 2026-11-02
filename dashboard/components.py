"""Streamlit/Plotly rendering helpers for the INOEMA dashboard.

Pure presentation layer: every function here only renders data that was
already computed by agent.demo_agent / pipeline.py / the storage layer.
Nothing here runs a simulation, calls the Monte Carlo engine, or decides
anything -- it only displays what those modules already produced, and
only ever reformats existing numbers (e.g. frequency -> trial count);
it never recomputes or alters a result.

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

# --- INOEMA palette (light) -------------------------------------------------
BG = "#FFFFFF"
PRIMARY_TEXT = "#102C2B"
TEAL = "#2CB6A4"
CORAL = "#F07868"
SURFACE = "#F3F7F5"
BORDER = "#E4E9E7"
MUTED = "#65716F"

TEXT = PRIMARY_TEXT  # kept for readability at call sites below

RISK_COLORS = {"LOW": TEAL, "MEDIUM": CORAL, "HIGH": CORAL, "UNKNOWN": CORAL}
DECISION_COLORS = {"PROCEED": TEAL, "ADJUST": CORAL, "ABSTAIN": CORAL}

SYNTHETIC_NOTICE = (
    "SYNTHETIC / MOCKED ENVIRONMENT — no real PayPal, bank, or ledger "
    "system is contacted anywhere in this demonstration."
)

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color=PRIMARY_TEXT, family="sans-serif"),
    margin=dict(l=40, r=20, t=30, b=40),
    xaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
    yaxis=dict(gridcolor=BORDER, linecolor=BORDER, zerolinecolor=BORDER),
)


def inject_base_style() -> None:
    """Small, targeted CSS overrides -- restrained cards, no glow/gradients,
    muted secondary text, a teal (not blue) primary button. Applied once."""
    st.markdown(
        f"""
        <style>
          div[data-testid="stMetric"] {{
              background: {SURFACE};
              border: 1px solid {BORDER};
              border-radius: 4px;
              padding: 0.55rem 0.75rem;
          }}
          div[data-testid="stMetricValue"] {{
              font-size: 1.35rem;
              font-weight: 600;
              color: {PRIMARY_TEXT};
          }}
          div[data-testid="stMetricLabel"] {{
              color: {MUTED};
              font-size: 0.75rem;
              text-transform: uppercase;
              letter-spacing: 0.03em;
          }}
          .stButton > button[kind="primary"] {{
              background-color: {TEAL};
              border: 1px solid {TEAL};
              color: #06211D;
              border-radius: 4px;
              font-weight: 600;
              box-shadow: none;
          }}
          .stButton > button[kind="primary"]:hover {{
              background-color: #26A294;
              border-color: #26A294;
              color: #06211D;
          }}
          .stButton > button[kind="secondary"] {{
              border-radius: 4px;
              border-color: {BORDER};
              color: {PRIMARY_TEXT};
          }}
          hr {{ border-color: {BORDER}; }}
          [data-testid="stExpander"] {{
              border: 1px solid {BORDER};
              border-radius: 4px;
          }}
          code {{
              color: {PRIMARY_TEXT};
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def minutes_to_clock(minutes: float) -> str:
    total = round(minutes)
    hour = 16 + total // 60
    minute = total % 60
    return f"{int(hour):02d}:{int(minute):02d}"


def shortfall_trial_count(mc_result) -> int:
    """Recover the integer trial count behind mc_result.p_shortfall, for
    display -- this reformats an existing value, it does not recompute it."""
    return round(mc_result.p_shortfall * mc_result.n_trials)


def render_header() -> None:
    inject_base_style()
    st.markdown(
        f"""
        <div style="padding: 0.4rem 0 1rem 0; border-bottom: 1px solid {BORDER}; margin-bottom: 1rem;">
          <div style="display:flex; align-items:baseline; gap: 0.75rem;">
            <span style="font-size: 1.9rem; font-weight: 700; letter-spacing: 0.03em; color:{PRIMARY_TEXT};">INOEMA</span>
            <span style="font-size: 0.92rem; color:{MUTED};">Simulation Infrastructure for Autonomous Systems</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="border-left: 3px solid {CORAL}; background:{SURFACE}; padding: 0.5rem 0.9rem;
                    border-radius: 2px; margin-bottom: 0.9rem; color:{PRIMARY_TEXT}; font-size:0.88rem;">
          <strong>{SYNTHETIC_NOTICE}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1, 1, 6])
    with cols[0]:
        st.markdown(
            f"<span style='color:{TEAL};'>&#9679;</span>&nbsp;<span style='color:{MUTED};'>Engine</span> "
            f"<span style='color:{PRIMARY_TEXT};'>Online</span>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f"<span style='color:{TEAL};'>&#9679;</span>&nbsp;<span style='color:{MUTED};'>Mode</span> "
            f"<span style='color:{PRIMARY_TEXT};'>Mocked</span>",
            unsafe_allow_html=True,
        )


def render_agent_control(result: AgentRunResult, batch_size: int) -> None:
    st.subheader("Agent Control")
    initial_action = result.trace[0].action

    cols = st.columns(4)
    cols[0].metric("Objective", "Release batch safely")
    cols[1].metric("Batch Size", f"{batch_size:,}")
    release_time, release_count = initial_action.release_schedule[0]
    cols[2].metric("Initial Release Time", minutes_to_clock(release_time))
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
                    f"{minutes_to_clock(t)} × {c:,}" for t, c in step.action.release_schedule
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
                f"{minutes_to_clock(t)} × {c:,}"
                for t, c in revised_step.action.release_schedule
            )
        )
        _render_verdict_metrics(revised_step)

    st.markdown("**Initial vs. Revised — Monte Carlo Outcomes**")
    _render_risk_comparison_charts(initial_step, revised_step)


def _render_verdict_metrics(step) -> None:
    verdict = step.verdict
    mc = step.mc_result
    n_shortfall = shortfall_trial_count(mc)

    cols = st.columns(5)
    cols[0].metric("Monte Carlo Trials", f"{mc.n_trials:,}")
    cols[1].metric("Trials with Shortfall", f"{n_shortfall} / {mc.n_trials}")
    cols[1].caption(f"{verdict.risk_score:.1%} of trials")
    cols[2].metric("Confidence", f"{verdict.confidence:.1%}")
    cols[3].metric("Expected Impact", f"{round(verdict.expected_impact_count):,} payments")
    cols[3].caption(f"{verdict.expected_impact_fraction:.1%} of batch")

    risk_color = RISK_COLORS.get(verdict.risk_level.value, MUTED)
    cols[4].markdown(
        f"<div style='text-align:center;'>"
        f"<div style='color:{MUTED}; font-size:0.75rem; text-transform:uppercase; letter-spacing:0.03em;'>"
        f"Risk Level / Recommendation</div>"
        f"<div style='font-size:1.15rem; font-weight:600; color:{risk_color};'>{verdict.risk_level.value}</div>"
        f"<div style='font-size:0.82rem; color:{PRIMARY_TEXT};'>{verdict.decision.value}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_risk_comparison_charts(initial_step, revised_step) -> None:
    steps = [("Initial", initial_step)]
    if revised_step is not None:
        steps.append(("Revised", revised_step))

    labels = [label for label, _ in steps]
    bar_colors = [RISK_COLORS.get(s.verdict.risk_level.value, MUTED) for _, s in steps]

    shortfall_counts = [shortfall_trial_count(s.mc_result) for _, s in steps]
    trial_totals = [s.mc_result.n_trials for _, s in steps]
    impact_counts = [round(s.verdict.expected_impact_count) for _, s in steps]

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_bar(x=labels, y=shortfall_counts, marker_color=bar_colors, showlegend=False)
        fig.update_layout(
            title=dict(text="Trials with Shortfall", font=dict(size=13, color=MUTED)),
            yaxis_title="Trials",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            " · ".join(
                f"{label}: {count}/{total} ({count / total:.1%})"
                for label, count, total in zip(labels, shortfall_counts, trial_totals)
            )
        )

    with col2:
        fig = go.Figure()
        fig.add_bar(x=labels, y=impact_counts, marker_color=bar_colors, showlegend=False)
        fig.update_layout(
            title=dict(text="Expected Impact (payments)", font=dict(size=13, color=MUTED)),
            yaxis_title="Payments",
            **PLOTLY_LAYOUT,
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            " · ".join(
                f"{label}: {count:,} ({s.verdict.expected_impact_fraction:.1%} of batch)"
                for label, count, (_, s) in zip(labels, impact_counts, steps)
            )
        )


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
    st.caption(f"Submission label: `{batch.label}`")


def render_verification(
    comparison: Optional[OutcomeComparison],
    actual_outcome_row: Optional[dict],
    batch: Optional[MockPayoutBatch] = None,
    predicted_impact_count: Optional[float] = None,
) -> None:
    st.subheader("Independent Verification")
    if comparison is None:
        st.info("No verification yet — nothing has been executed.")
        return

    st.markdown(
        f"<span style='color:{TEAL}; font-weight:600;'>Collector status</span>&nbsp;"
        f"<span style='color:{PRIMARY_TEXT};'>— outcome independently retrieved and compared.</span>",
        unsafe_allow_html=True,
    )

    batch_size = batch.batch_size if batch is not None else None
    actual_missed_count: Optional[int] = None
    if actual_outcome_row is not None and batch_size is not None:
        actual_missed_count = batch_size - actual_outcome_row["n_completed_by_close"]
    elif batch_size is not None:
        actual_missed_count = round(comparison.actual_missed_fraction * batch_size)

    cols = st.columns(4)
    if predicted_impact_count is not None:
        cols[0].metric("Predicted Impact", f"{round(predicted_impact_count):,} payments")
    else:
        cols[0].metric("Predicted Impact", f"{comparison.predicted_expected_impact_fraction:.1%} of batch")
    cols[0].caption(f"{comparison.predicted_expected_impact_fraction:.1%} of batch")

    if actual_missed_count is not None:
        cols[1].metric("Actual Unreconciled", f"{actual_missed_count:,} payments")
    else:
        cols[1].metric("Actual Unreconciled", f"{comparison.actual_missed_fraction:.1%} of batch")
    cols[1].caption(f"{comparison.actual_missed_fraction:.1%} of batch")

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
    if predicted_impact_count is not None and actual_missed_count is not None:
        fig.add_bar(name="PREDICTED", x=["Executed Batch"], y=[round(predicted_impact_count)], marker_color=TEAL)
        fig.add_bar(name="ACTUAL (independent)", x=["Executed Batch"], y=[actual_missed_count], marker_color=CORAL)
        y_title = "Unreconciled payments"
    else:
        fig.add_bar(
            name="PREDICTED",
            x=["Executed Batch"],
            y=[comparison.predicted_expected_impact_fraction],
            marker_color=TEAL,
        )
        fig.add_bar(
            name="ACTUAL (independent)",
            x=["Executed Batch"],
            y=[comparison.actual_missed_fraction],
            marker_color=CORAL,
        )
        y_title = "Fraction of batch"
    fig.update_layout(
        barmode="group",
        yaxis_title=y_title,
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

    pending_color = CORAL if not approved else MUTED

    chips = []
    for step in steps:
        if step in completed:
            chips.append(f"<span style='color:{TEAL}; font-weight:600;'>{step}</span>")
        else:
            chips.append(f"<span style='color:{pending_color};'>{step}</span>")
    st.markdown(" &nbsp;→&nbsp; ".join(chips), unsafe_allow_html=True)

    if not approved:
        st.error(
            "HUMAN INTERVENTION REQUIRED — the agent stopped before execution "
            f"({result.final_decision.rationale})"
        )
