"""Developer Integration view for the INOEMA dashboard.

Demonstrates how an external autonomous agent *would* call INOEMA over a
REST API or an MCP adapter. This is a prototype/mock integration surface
only: there is no running HTTP server and no real MCP client connected
anywhere in this module. Every example below is either:

  - REAL: built directly from this session's actual AgentRunResult /
    OutcomeComparison objects (the same ones the main Simulation Control
    Center tab renders), reformatted as a JSON-ish payload -- never
    recomputed, never invented; or
  - ILLUSTRATIVE: a clearly labelled, static mock payload used only when
    no matching real data exists yet in this session.

Nothing here duplicates simulation, agent, or pipeline logic -- it only
reads whatever the existing run already produced.
"""

from __future__ import annotations

import json
from typing import Optional

import streamlit as st

from dashboard.components import BORDER, CORAL, MUTED, PRIMARY_TEXT, SURFACE, TEAL, minutes_to_clock, shortfall_trial_count

ILLUSTRATIVE_RESPONSE = {
    "simulation_id": "sim-illustrative-0001",
    "status": "ILLUSTRATIVE_EXAMPLE",
    "risk_score": 0.18,
    "expected_impact_count": 42,
    "shortfall_trial_count": 36,
    "total_trials": 200,
    "recommendation": "ADJUST",
    "confidence": 0.91,
    "risk_level": "MEDIUM",
    "synthetic": True,
}


def _simulation_id(result, comparison) -> str:
    if comparison is not None:
        return comparison.batch_id
    final_step = result.trace[-1]
    return f"sim-{final_step.mc_result.master_seed}-{final_step.mc_result.n_trials}-{len(result.trace)}"


def _verdict_payload(step) -> dict:
    verdict = step.verdict
    mc = step.mc_result
    return {
        "risk_score": round(verdict.risk_score, 4),
        "expected_impact_count": round(verdict.expected_impact_count),
        "shortfall_trial_count": shortfall_trial_count(mc),
        "total_trials": mc.n_trials,
        "recommendation": verdict.decision.value,
        "confidence": round(verdict.confidence, 4),
        "risk_level": verdict.risk_level.value,
        "synthetic": True,
    }


def _simulation_response(result, comparison) -> dict:
    final_step = result.trace[-1]
    return {
        "simulation_id": _simulation_id(result, comparison),
        "status": "COMPLETED",
        **_verdict_payload(final_step),
    }


def _release_schedule_payload(action) -> list[dict]:
    return [{"time": minutes_to_clock(t), "count": c} for t, c in action.release_schedule]


def _label(is_real: bool) -> str:
    return "REAL — from this session's run" if is_real else "ILLUSTRATIVE — mock payload"


def _render_label(is_real: bool) -> None:
    color = TEAL if is_real else CORAL
    st.markdown(
        f"<span style='color:{color}; font-weight:600; font-size:0.78rem; "
        f"text-transform:uppercase; letter-spacing:0.03em;'>{_label(is_real)}</span>",
        unsafe_allow_html=True,
    )


# --- 1. REST API panel ------------------------------------------------------

def render_rest_api_panel(demo: Optional[dict]) -> None:
    st.markdown("#### REST API")
    st.caption(
        "Prototype / mock interface — no server is actually running. This "
        "shows the contract INOEMA would expose to external agents."
    )

    st.markdown("**POST** `/api/v1/simulations`")

    example_request = {
        "scenario": "payment_release",
        "batch_size": 2000,
        "release_schedule": [{"time": "18:30", "count": 2000}],
        "trials": 200,
    }
    st.markdown("Example request")
    st.code(json.dumps(example_request, indent=2), language="json")

    curl_cmd = (
        "curl -X POST https://api.inoema.example/v1/simulations \\\n"
        '  -H "Content-Type: application/json" \\\n'
        f"  -d '{json.dumps(example_request)}'"
    )
    st.markdown("curl")
    st.code(curl_cmd, language="bash")

    result = demo.get("result") if demo else None
    comparison = demo.get("comparison") if demo else None

    st.markdown("Example response")
    if result is not None:
        response = _simulation_response(result, comparison)
        is_real = True
    else:
        response = dict(ILLUSTRATIVE_RESPONSE)
        is_real = False
    _render_label(is_real)
    st.code(json.dumps(response, indent=2), language="json")


# --- 2. MCP tool catalogue ---------------------------------------------------

def _simulate_scenario_example(demo):
    result = demo.get("result") if demo else None
    if result is not None:
        return _simulation_response(result, demo.get("comparison")), True
    return dict(ILLUSTRATIVE_RESPONSE), False


def _compare_scenarios_example(demo):
    result = demo.get("result") if demo else None
    if result is not None and len(result.trace) > 1:
        steps = result.trace
        return (
            {
                "scenarios": [
                    {"label": "Initial Proposal", **_verdict_payload(steps[0])},
                    {"label": "Revised Proposal", **_verdict_payload(steps[1])},
                ],
                "synthetic": True,
            },
            True,
        )
    return (
        {
            "scenarios": [
                {"label": "Scenario A (illustrative)", "risk_score": 0.21, "recommendation": "ADJUST", "synthetic": True},
                {"label": "Scenario B (illustrative)", "risk_score": 0.02, "recommendation": "PROCEED", "synthetic": True},
            ],
            "synthetic": True,
        },
        False,
    )


def _get_simulation_result_example(demo):
    result = demo.get("result") if demo else None
    if result is not None:
        payload = _simulation_response(result, demo.get("comparison"))
        payload["decision_rationale"] = result.trace[-1].agent_decision.rationale
        return payload, True
    payload = dict(ILLUSTRATIVE_RESPONSE)
    payload["decision_rationale"] = "Illustrative rationale — no run yet in this session."
    return payload, False


def _verify_actual_outcome_example(demo):
    comparison = demo.get("comparison") if demo else None
    if comparison is not None:
        return (
            {
                "batch_id": comparison.batch_id,
                "predicted_expected_impact_fraction": round(comparison.predicted_expected_impact_fraction, 4),
                "actual_missed_fraction": round(comparison.actual_missed_fraction, 4),
                "prediction_error": round(comparison.prediction_error, 4),
                "brier_component": round(comparison.brier_component, 6),
                "synthetic": True,
            },
            True,
        )
    return (
        {
            "batch_id": "batch-illustrative-0001",
            "predicted_expected_impact_fraction": 0.02,
            "actual_missed_fraction": 0.015,
            "prediction_error": -0.005,
            "brier_component": 0.0003,
            "synthetic": True,
        },
        False,
    )


_TOOL_DEFS = [
    (
        "simulate_scenario",
        "Run Monte Carlo simulations against a proposed action.",
        {"batch_size": 2000, "release_schedule": [{"time": "18:30", "count": 2000}], "trials": 200, "seed": 0},
        _simulate_scenario_example,
    ),
    (
        "compare_scenarios",
        "Compare multiple proposed actions and their predicted outcomes.",
        {
            "scenarios": [
                {"label": "Initial", "release_schedule": [{"time": "18:30", "count": 2000}]},
                {
                    "label": "Revised",
                    "release_schedule": [{"time": "16:00", "count": 1000}, {"time": "17:15", "count": 1000}],
                },
            ]
        },
        _compare_scenarios_example,
    ),
    (
        "get_simulation_result",
        "Retrieve a simulation result and its decision rationale.",
        {"simulation_id": "sim-..."},
        _get_simulation_result_example,
    ),
    (
        "verify_actual_outcome",
        "Retrieve independently collected execution outcomes.",
        {"batch_id": "batch-..."},
        _verify_actual_outcome_example,
    ),
]


def render_mcp_catalogue(demo: Optional[dict]) -> None:
    st.markdown("#### MCP Tool Catalogue")
    st.caption(
        "Prototype tool definitions INOEMA would expose over an MCP adapter. "
        "No live MCP server is running."
    )
    for name, description, input_example, builder in _TOOL_DEFS:
        with st.expander(name, expanded=False):
            st.write(description)
            st.markdown("Input arguments (representative)")
            st.code(json.dumps(input_example, indent=2), language="json")
            response, is_real = builder(demo)
            st.markdown("Example response")
            _render_label(is_real)
            st.code(json.dumps(response, indent=2), language="json")


# --- 3. MCP invocation trace -------------------------------------------------

def render_mcp_trace(demo: Optional[dict]) -> None:
    st.markdown("#### MCP Invocation Trace")
    st.caption(
        "Simulated MCP interaction — no real external MCP client is connected. "
        "Replays a believable tool-call sequence, using real data from the "
        "latest run in this session wherever it exists."
    )

    if st.button("Run example MCP interaction"):
        st.session_state["show_mcp_trace"] = True

    if not st.session_state.get("show_mcp_trace"):
        return

    result = demo.get("result") if demo else None
    comparison = demo.get("comparison") if demo else None
    has_run = result is not None
    has_adjustment = has_run and len(result.trace) > 1

    steps: list[tuple[str, Optional[dict], Optional[dict], bool]] = []

    steps.append(
        (
            "1. Agent discovers available INOEMA tools",
            None,
            {"tools": [t[0] for t in _TOOL_DEFS]},
            True,
        )
    )

    if has_run:
        initial_action = result.trace[0].action
        args1 = {
            "batch_size": sum(c for _, c in initial_action.release_schedule),
            "release_schedule": _release_schedule_payload(initial_action),
            "trials": result.trace[0].mc_result.n_trials,
        }
        response1 = _simulation_response(result, None)
        real1 = True
    else:
        args1 = {"batch_size": 2000, "release_schedule": [{"time": "18:30", "count": 2000}], "trials": 200}
        response1 = dict(ILLUSTRATIVE_RESPONSE)
        real1 = False
    steps.append(("2. Agent invokes simulate_scenario", args1, response1, real1))
    steps.append(("3. INOEMA returns the simulation result", None, response1, real1))

    if has_run and result.trace[0].verdict.decision.value == "ADJUST":
        adjust_info = {"decision": "ADJUST", "rationale": result.trace[0].agent_decision.rationale}
        real_adjust = True
    else:
        adjust_info = {
            "decision": "ADJUST",
            "rationale": "Illustrative — this session's initial proposal did not require adjustment.",
        }
        real_adjust = False
    steps.append(("4. Agent receives ADJUST and revises its proposal", None, adjust_info, real_adjust))

    if has_adjustment:
        revised_action = result.trace[1].action
        args2 = {
            "batch_size": sum(c for _, c in revised_action.release_schedule),
            "release_schedule": _release_schedule_payload(revised_action),
            "trials": result.trace[1].mc_result.n_trials,
        }
        response2 = _simulation_response(result, comparison)
        real2 = True
    else:
        args2 = {
            "batch_size": 2000,
            "release_schedule": [{"time": "16:00", "count": 1000}, {"time": "17:15", "count": 1000}],
            "trials": 200,
        }
        response2 = {**ILLUSTRATIVE_RESPONSE, "recommendation": "PROCEED", "risk_score": 0.0, "shortfall_trial_count": 0}
        real2 = False
    steps.append(("5. Agent invokes simulate_scenario again (revised proposal)", args2, response2, real2))
    steps.append(("6. INOEMA returns PROCEED", None, response2, real2))

    if has_adjustment:
        approval_info = {
            "approved_for_execution": result.final_decision.approved_for_execution,
            "rationale": result.final_decision.rationale,
        }
        real_approval = True
    else:
        approval_info = {
            "approved_for_execution": True,
            "rationale": "Illustrative — proposal cleared the agent's risk/confidence thresholds.",
        }
        real_approval = False
    steps.append(("7. Agent receives approval information", None, approval_info, real_approval))

    for title, args, response, is_real in steps:
        with st.expander(title, expanded=False):
            _render_label(is_real)
            if args is not None:
                st.markdown("Arguments")
                st.code(json.dumps(args, indent=2), language="json")
            if response is not None:
                st.markdown("Response")
                st.code(json.dumps(response, indent=2), language="json")


# --- 4. Architecture diagram -------------------------------------------------

_ARCH_STAGES = [
    "External AI Agent",
    "REST API / MCP Adapter",
    "INOEMA Simulation Engine",
    "Structured Risk Result",
    "Agent Decision",
    "Execution",
    "Independent Verification",
]


def render_architecture_diagram() -> None:
    st.markdown("#### Architecture")
    st.caption("How an external autonomous agent integrates with INOEMA.")

    boxes = [
        f"<div style='border:1px solid {BORDER}; border-radius:4px; background:{SURFACE}; "
        f"padding:0.55rem 0.6rem; font-size:0.76rem; color:{PRIMARY_TEXT}; text-align:center; "
        f"flex:1; min-width:120px;'>{stage}</div>"
        for stage in _ARCH_STAGES
    ]
    arrow = f"<div style='color:{MUTED}; padding:0 0.35rem; display:flex; align-items:center; font-size:0.9rem;'>&rarr;</div>"
    row = (
        "<div style='display:flex; align-items:stretch; gap:0.15rem; overflow-x:auto; padding-bottom:0.3rem;'>"
        + arrow.join(boxes)
        + "</div>"
    )
    st.markdown(row, unsafe_allow_html=True)


# --- Top-level section --------------------------------------------------------

def render_developer_integration(demo: Optional[dict]) -> None:
    st.subheader("Developer Integration")
    st.caption(
        "Prototype interface demonstrating how an external autonomous agent "
        "could call INOEMA. No production MCP server or external API is "
        "running — everything below is a mocked integration surface, "
        "populated from real session data wherever one exists."
    )

    render_architecture_diagram()
    st.divider()
    render_rest_api_panel(demo)
    st.divider()
    render_mcp_catalogue(demo)
    st.divider()
    render_mcp_trace(demo)
