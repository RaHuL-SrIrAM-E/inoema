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
    """Targeted CSS overrides: soft elevation, a teal gradient primary
    button, a numbered-stepper timeline, section "kickers", and polished
    code/expander/container treatments -- all within the INOEMA palette
    (no electric blue, no emoji, no AI-chatbot styling)."""
    st.markdown(
        f"""
        <style>
          html, body, [class*="css"] {{
              font-family: -apple-system, "Segoe UI", Inter, sans-serif;
          }}

          /* Metric cards -------------------------------------------------- */
          div[data-testid="stMetric"] {{
              background: {SURFACE};
              border: 1px solid {BORDER};
              border-radius: 10px;
              padding: 0.7rem 0.85rem 0.6rem 0.85rem;
              box-shadow: 0 1px 2px rgba(16, 44, 43, 0.04);
              transition: box-shadow 0.15s ease, transform 0.15s ease;
          }}
          div[data-testid="stMetric"]:hover {{
              box-shadow: 0 4px 14px rgba(16, 44, 43, 0.08);
              transform: translateY(-1px);
          }}
          div[data-testid="stMetricValue"] {{
              font-size: 1.45rem;
              font-weight: 650;
              color: {PRIMARY_TEXT};
              letter-spacing: -0.01em;
          }}
          div[data-testid="stMetricLabel"] {{
              color: {MUTED};
              font-size: 0.72rem;
              text-transform: uppercase;
              letter-spacing: 0.05em;
              font-weight: 600;
          }}

          /* Bordered containers used as cards ------------------------------ */
          div[data-testid="stVerticalBlockBorderWrapper"] {{
              border-radius: 12px !important;
              border-color: {BORDER} !important;
              box-shadow: 0 1px 3px rgba(16, 44, 43, 0.05);
          }}

          /* Buttons --------------------------------------------------------- */
          .stButton > button[kind="primary"] {{
              background: linear-gradient(135deg, {TEAL} 0%, #209483 100%);
              border: 1px solid #1E8A7A;
              color: #FFFFFF;
              border-radius: 8px;
              font-weight: 650;
              letter-spacing: 0.01em;
              box-shadow: 0 2px 8px rgba(44, 182, 164, 0.28);
              transition: box-shadow 0.15s ease, transform 0.15s ease;
          }}
          .stButton > button[kind="primary"]:hover {{
              box-shadow: 0 4px 14px rgba(44, 182, 164, 0.4);
              transform: translateY(-1px);
          }}
          .stButton > button[kind="secondary"] {{
              border-radius: 8px;
              border-color: {BORDER};
              color: {PRIMARY_TEXT};
              transition: border-color 0.15s ease;
          }}
          .stButton > button[kind="secondary"]:hover {{
              border-color: {TEAL};
              color: {TEAL};
          }}

          hr {{ border-color: {BORDER}; }}

          /* Expanders ---------------------------------------------------- */
          [data-testid="stExpander"] {{
              border: 1px solid {BORDER};
              border-radius: 10px;
              box-shadow: 0 1px 2px rgba(16, 44, 43, 0.03);
          }}
          [data-testid="stExpander"] summary:hover {{
              color: {TEAL};
          }}

          /* Code blocks -- quiet "console" treatment ----------------------- */
          [data-testid="stCodeBlock"] pre {{
              border-radius: 10px !important;
              border: 1px solid {BORDER} !important;
              box-shadow: 0 1px 2px rgba(16, 44, 43, 0.04);
          }}
          code {{ color: {PRIMARY_TEXT}; }}

          /* Section kicker label ------------------------------------------- */
          .inoema-kicker {{
              display: inline-flex;
              align-items: center;
              gap: 0.4rem;
              font-size: 0.72rem;
              font-weight: 700;
              letter-spacing: 0.09em;
              text-transform: uppercase;
              color: var(--kicker-color, {TEAL});
              margin-bottom: 0.15rem;
          }}
          .inoema-kicker::before {{
              content: "";
              width: 7px;
              height: 7px;
              border-radius: 50%;
              background: var(--kicker-color, {TEAL});
              display: inline-block;
          }}
          .inoema-rule {{
              height: 2px;
              width: 46px;
              border-radius: 2px;
              background: linear-gradient(90deg, {TEAL}, {CORAL});
              margin: 0.3rem 0 0.9rem 0;
          }}

          /* Pill badges ------------------------------------------------------ */
          .inoema-pill {{
              display: inline-flex;
              align-items: center;
              gap: 0.35rem;
              padding: 0.28rem 0.7rem;
              border-radius: 999px;
              background: {SURFACE};
              border: 1px solid {BORDER};
              font-size: 0.78rem;
              color: {PRIMARY_TEXT};
          }}
          .inoema-dot {{
              width: 7px; height: 7px; border-radius: 50%;
              background: {TEAL};
              box-shadow: 0 0 0 3px rgba(44,182,164,0.18);
          }}

          /* Stepper timeline -------------------------------------------------- */
          .inoema-stepper {{ display:flex; align-items:flex-start; width:100%; overflow-x:auto; padding: 0.4rem 0 0.2rem 0; }}
          .inoema-step {{ display:flex; flex-direction:column; align-items:center; flex:1; min-width:92px; position:relative; }}
          .inoema-step-circle {{
              width: 30px; height: 30px; border-radius: 50%;
              display:flex; align-items:center; justify-content:center;
              font-size: 0.75rem; font-weight:700;
              border: 2px solid {BORDER};
              background: #FFFFFF;
              color: {MUTED};
              z-index: 2;
          }}
          .inoema-step-circle.done {{
              background: {TEAL}; border-color: {TEAL}; color: #FFFFFF;
              box-shadow: 0 0 0 4px rgba(44,182,164,0.15);
          }}
          .inoema-step-circle.blocked {{
              background: #FFFFFF; border-color: {CORAL}; color: {CORAL};
          }}
          .inoema-step-line {{
              position:absolute; top:15px; left:-50%; width:100%; height:2px;
              background: {BORDER}; z-index:1;
          }}
          .inoema-step-line.done {{ background: {TEAL}; }}
          .inoema-step:first-child .inoema-step-line {{ display:none; }}
          .inoema-step-label {{
              margin-top: 0.4rem; font-size: 0.68rem; font-weight:700;
              letter-spacing: 0.04em; text-transform:uppercase; text-align:center;
              color: {PRIMARY_TEXT};
          }}
          .inoema-step-label.pending {{ color: {MUTED}; }}
          .inoema-step-label.blocked {{ color: {CORAL}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kicker(label: str, color: str = TEAL) -> None:
    """A small colored label + gradient rule rendered above a subheader --
    purely decorative, never changes the subheader text itself."""
    st.markdown(
        f"<div class='inoema-kicker' style='--kicker-color:{color};'>{label}</div>"
        f"<div class='inoema-rule'></div>",
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
        <div style="padding: 0.5rem 0 1.1rem 0; margin-bottom: 1.1rem;
                    border-bottom: 1px solid {BORDER};
                    background: linear-gradient(180deg, {SURFACE} 0%, rgba(255,255,255,0) 100%);">
          <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.6rem;">
            <div style="display:flex; align-items:baseline; gap: 0.85rem;">
              <span style="font-size: 2.1rem; font-weight: 800; letter-spacing: 0.01em; color:{PRIMARY_TEXT};">INOEMA</span>
              <span style="font-size: 0.95rem; color:{MUTED};">Simulation Infrastructure for Autonomous Systems</span>
            </div>
            <div style="display:flex; gap:0.5rem;">
              <span class="inoema-pill"><span class="inoema-dot"></span>Engine&nbsp;Online</span>
              <span class="inoema-pill"><span class="inoema-dot"></span>Mode&nbsp;Mocked</span>
            </div>
          </div>
          <div style="height:3px; width:72px; margin-top:0.75rem; border-radius:2px;
                      background: linear-gradient(90deg, {TEAL}, {CORAL});"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_agent_control(result: AgentRunResult, batch_size: int) -> None:
    render_kicker("Step 01 · Agent")
    st.subheader("Agent Control")
    initial_action = result.trace[0].action

    with st.container(border=True):
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
    render_kicker("Step 02 · Monte Carlo Engine")
    st.subheader("Simulation Results")
    initial_step = result.trace[0]
    revised_step = result.trace[1] if len(result.trace) > 1 else None

    st.markdown("**Initial Proposal**")
    with st.container(border=True):
        _render_verdict_metrics(initial_step)

    if revised_step is not None:
        st.markdown("**Revised Proposal (agent-adjusted)**")
        st.caption(
            "Release schedule: "
            + ", ".join(
                f"{minutes_to_clock(t)} × {c:,}"
                for t, c in revised_step.action.release_schedule
            )
        )
        with st.container(border=True):
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
        fig.add_bar(
            x=labels,
            y=shortfall_counts,
            marker_color=bar_colors,
            marker_line_width=0,
            text=shortfall_counts,
            textposition="outside",
            textfont=dict(color=PRIMARY_TEXT, size=12),
            showlegend=False,
        )
        fig.update_layout(
            title=dict(text="Trials with Shortfall", font=dict(size=13, color=MUTED)),
            yaxis_title="Trials",
            bargap=0.45,
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
        fig.add_bar(
            x=labels,
            y=impact_counts,
            marker_color=bar_colors,
            marker_line_width=0,
            text=[f"{v:,}" for v in impact_counts],
            textposition="outside",
            textfont=dict(color=PRIMARY_TEXT, size=12),
            showlegend=False,
        )
        fig.update_layout(
            title=dict(text="Expected Impact (payments)", font=dict(size=13, color=MUTED)),
            yaxis_title="Payments",
            bargap=0.45,
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
    render_kicker("Step 03 · Mock Gateway")
    st.subheader("Execution — Mock PayPal Gateway")
    if batch is None:
        st.info("No execution yet — the proposal has not been approved for execution.")
        return

    with st.container(border=True):
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
    render_kicker("Step 04 · Independent Collector", color=CORAL)
    st.subheader("Independent Verification")
    if comparison is None:
        st.info("No verification yet — nothing has been executed.")
        return

    st.markdown(
        "<span class='inoema-pill'><span class='inoema-dot'></span>"
        f"<span style='color:{PRIMARY_TEXT}; font-weight:600;'>Collector status</span>&nbsp;"
        f"<span style='color:{MUTED};'>outcome independently retrieved and compared</span></span>",
        unsafe_allow_html=True,
    )
    st.write("")

    batch_size = batch.batch_size if batch is not None else None
    actual_missed_count: Optional[int] = None
    if actual_outcome_row is not None and batch_size is not None:
        actual_missed_count = batch_size - actual_outcome_row["n_completed_by_close"]
    elif batch_size is not None:
        actual_missed_count = round(comparison.actual_missed_fraction * batch_size)

    with st.container(border=True):
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
        with st.container(border=True):
            rcols = st.columns(3)
            rcols[0].metric("Day Slowdown Multiplier", f"{actual_outcome_row['day_slowdown_multiplier']:.2f}×")
            rcols[1].metric("Reality Seed", actual_outcome_row["reality_seed"])
            rcols[2].metric("Completed by Close", f"{actual_outcome_row['n_completed_by_close']:,}")

    st.markdown("**Predicted vs. Actual**")
    fig = go.Figure()
    if predicted_impact_count is not None and actual_missed_count is not None:
        fig.add_bar(
            name="PREDICTED", x=["Executed Batch"], y=[round(predicted_impact_count)],
            marker_color=TEAL, marker_line_width=0,
            text=[f"{round(predicted_impact_count):,}"], textposition="outside",
        )
        fig.add_bar(
            name="ACTUAL (independent)", x=["Executed Batch"], y=[actual_missed_count],
            marker_color=CORAL, marker_line_width=0,
            text=[f"{actual_missed_count:,}"], textposition="outside",
        )
        y_title = "Unreconciled payments"
    else:
        fig.add_bar(
            name="PREDICTED",
            x=["Executed Batch"],
            y=[comparison.predicted_expected_impact_fraction],
            marker_color=TEAL,
            marker_line_width=0,
        )
        fig.add_bar(
            name="ACTUAL (independent)",
            x=["Executed Batch"],
            y=[comparison.actual_missed_fraction],
            marker_color=CORAL,
            marker_line_width=0,
        )
        y_title = "Fraction of batch"
    fig.update_layout(
        barmode="group",
        yaxis_title=y_title,
        bargap=0.4,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        **PLOTLY_LAYOUT,
    )
    st.plotly_chart(fig, use_container_width=True)


_TIMELINE_STEPS = ["PROPOSE", "SIMULATE", "ADJUST", "RE-SIMULATE", "APPROVE", "EXECUTE", "VERIFY"]


def render_timeline(result: AgentRunResult, executed: bool, verified: bool) -> None:
    render_kicker("Step 05 · Full Lifecycle")
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

    # The first not-yet-completed step is where the run actually stopped --
    # only that one gets the coral "blocked" treatment; later steps stay a
    # neutral pending grey rather than all turning red.
    blocked_step = None
    if not approved:
        for step in steps:
            if step not in completed:
                blocked_step = step
                break

    nodes = []
    for i, step in enumerate(steps, start=1):
        if step in completed:
            circle_class, label_class, marker = "done", "", "&#10003;"
        elif step == blocked_step:
            circle_class, label_class, marker = "blocked", "blocked", "!"
        else:
            circle_class, label_class, marker = "", "pending", str(i)
        line_class = "done" if (i > 1 and steps[i - 2] in completed and step in completed) else ""
        nodes.append(
            f"<div class='inoema-step'>"
            f"<div class='inoema-step-line {line_class}'></div>"
            f"<div class='inoema-step-circle {circle_class}'>{marker}</div>"
            f"<div class='inoema-step-label {label_class}'>{step}</div>"
            f"</div>"
        )
    st.markdown(f"<div class='inoema-stepper'>{''.join(nodes)}</div>", unsafe_allow_html=True)

    if not approved:
        st.error(
            "HUMAN INTERVENTION REQUIRED — the agent stopped before execution "
            f"({result.final_decision.rationale})"
        )
