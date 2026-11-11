"""Payments workspace router: five sections, session-state navigation (so buttons can jump between sections)."""
from __future__ import annotations

import os

import streamlit as st

from dashboard.payments import page_experiments, page_optimization, page_overview, page_sandbox, page_traces
from dashboard.payments import data, sandbox_service, status
from dashboard.payments.ui import render_payments_header, render_workflow_indicator

SECTIONS = ["Overview", "Policy Optimization", "Experiments", "PayPal Sandbox", "Agent Execution Traces"]
PAGES = {"Overview": page_overview.render, "Policy Optimization": page_optimization.render, "Experiments": page_experiments.render,
         "PayPal Sandbox": page_sandbox.render, "Agent Execution Traces": page_traces.render}


def render_payments_workspace() -> None:
    st.session_state.setdefault("payments_demo_mode", os.environ.get("INOEMA_DEMO_MODE", "0") == "1")
    with st.sidebar:
        st.markdown("### Payments")
        st.toggle("Demonstration mode", key="payments_demo_mode",
                  help="Loads previously persisted, genuine results (labelled HISTORICAL) and disables live agent / Sandbox actions. Never substitutes a scripted model.")
        if st.session_state["payments_demo_mode"]:
            st.caption("Historical, persisted data only.")
    render_payments_header()
    if st.session_state.get("payments_section") not in SECTIONS:
        st.session_state["payments_section"] = SECTIONS[0]
    st.radio("Section", SECTIONS, horizontal=True, key="payments_section", label_visibility="collapsed")
    steps = data.workflow_steps(gemini_configured=status.gemini_status().state != "NOT_CONFIGURED", paypal_configured=sandbox_service.credentials_configured())
    current = st.session_state["payments_section"]
    if current != "Overview":
        render_workflow_indicator(steps, current)  # compact: where am I in the end-to-end workflow
        PAGES[current]()
    else:
        PAGES[current](steps)
