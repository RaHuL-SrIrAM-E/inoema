"""Shared Streamlit rendering helpers for the Payments workspace. Reuses the existing INOEMA palette, CSS and Plotly layout from
dashboard.components (no new design system)."""
from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd
import streamlit as st

from dashboard.components import BORDER, CORAL, MUTED, PLOTLY_LAYOUT, PRIMARY_TEXT, SURFACE, TEAL, inject_base_style, render_kicker
from dashboard.payments.errors import ErrorInfo
from dashboard.payments.sources import SOURCE_STYLE, DataSource, Metric, badge_html

STATE_COLORS = {"OK": TEAL, "DEGRADED": CORAL, "DOWN": CORAL, "NOT_CONFIGURED": MUTED, "NOT_CHECKED": MUTED}


def goto(section: str, **session: Any) -> None:
    """Navigation callback (use as on_click): switch the Payments section and optionally set session values."""
    st.session_state["payments_section"] = section
    for k, v in session.items():
        st.session_state[k] = v


def demo_mode() -> bool:
    return bool(st.session_state.get("payments_demo_mode", False))


DISABLED_BUTTON_CSS = """<style>
.stButton > button[kind="primary"]:disabled, .stButton > button[kind="primary"][disabled] {
    background: #E4E9E7 !important; border: 1px solid #D5DCDA !important; color: #8A9694 !important;
    box-shadow: none !important; transform: none !important; cursor: not-allowed !important; }
.stButton > button:disabled { cursor: not-allowed !important; }
</style>"""


def render_payments_header() -> None:
    inject_base_style()
    st.markdown(DISABLED_BUTTON_CSS, unsafe_allow_html=True)
    mode = ("Demonstration&nbsp;mode&nbsp;·&nbsp;historical&nbsp;data" if demo_mode() else "Live&nbsp;workspace")
    st.markdown(
        f"""
        <div style="padding:0.5rem 0 1.0rem 0; margin-bottom:0.9rem; border-bottom:1px solid {BORDER};
                    background:linear-gradient(180deg,{SURFACE} 0%,rgba(255,255,255,0) 100%);">
          <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:0.6rem;">
            <div style="display:flex; align-items:baseline; gap:0.85rem;">
              <span style="font-size:2.1rem; font-weight:800; letter-spacing:0.01em; color:{PRIMARY_TEXT};">INOEMA</span>
              <span style="font-size:0.95rem; color:{MUTED};">Payments · autonomous recovery-policy optimisation</span>
            </div>
            <div style="display:flex; gap:0.5rem;"><span class="inoema-pill"><span class="inoema-dot"></span>{mode}</span></div>
          </div>
          <div style="height:3px; width:72px; margin-top:0.7rem; border-radius:2px; background:linear-gradient(90deg,{TEAL},{CORAL});"></div>
        </div>""", unsafe_allow_html=True)


def render_source_legend() -> None:
    cols = st.columns(4)
    for col, src in zip(cols, DataSource):
        with col:
            st.markdown(badge_html(src), unsafe_allow_html=True)
            st.caption(SOURCE_STYLE[src]["hint"])


def source_banner(source: DataSource, text: str, *, historical_ts: Optional[str] = None) -> None:
    extra = f"HISTORICAL · {fmt_ts(historical_ts)}" if historical_ts else ""
    st.markdown(f"<div style='margin:0.2rem 0 0.4rem 0;'>{badge_html(source, extra)} <span style='color:{MUTED}; font-size:0.85rem;'>{text}</span></div>", unsafe_allow_html=True)


def section_header(kicker: str, title: str, color: str = TEAL) -> None:
    render_kicker(kicker, color)
    st.subheader(title)


def render_metrics(metrics: list[Metric], *, key_prefix: str) -> None:
    """One st.metric per metric, each tagged with its data source; sources are never merged within a call."""
    srcs = {m.source for m in metrics}
    assert len(srcs) == 1, "a metric group must have exactly one data source"
    src = next(iter(srcs))
    color = SOURCE_STYLE[src]["color"]
    st.markdown(f"<div data-source='{src.value}' style='height:5px; border-radius:3px; margin:0.1rem 0 0.5rem 0; "
                f"background:{color};'></div>", unsafe_allow_html=True)  # colour-coded accent: the same colour is used for this source everywhere
    cols = st.columns(len(metrics))
    for col, m in zip(cols, metrics):
        with col:
            st.metric(m.label, m.display, help=SOURCE_STYLE[m.source]["hint"])
            if len(m.display) > 14:  # st.metric truncates long values with an ellipsis: always show the full value
                st.markdown(f"<div style='font-family:monospace; font-size:0.78rem; word-break:break-all; margin:-0.3rem 0 0.2rem 0;'>{m.display}</div>", unsafe_allow_html=True)
            st.caption(f"{SOURCE_STYLE[m.source]['mark']} {SOURCE_STYLE[m.source]['label']}" + (f" · {m.caption}" if m.caption else ""))


def status_pill(name: str, state: str, detail: str) -> None:
    color = STATE_COLORS.get(state, MUTED)
    st.markdown(f"<div style='border:1px solid {BORDER}; border-radius:10px; padding:0.6rem 0.8rem; background:{SURFACE};'>"
                f"<div style='font-size:0.7rem; font-weight:700; letter-spacing:0.05em; color:{MUTED}; text-transform:uppercase;'>{name}</div>"
                f"<div style='font-size:1.05rem; font-weight:700; color:{color};'>● {state.replace('_', ' ')}</div></div>", unsafe_allow_html=True)
    st.caption(detail)


def render_error(info: ErrorInfo, *, show_demo_button: bool = False) -> None:
    st.error(f"**{info.title}**")
    st.markdown(info.advice)
    if info.retry_after:
        st.info(f"The API suggests waiting {info.retry_after} before trying again.")
    with st.expander("Technical detail (sanitised)"):
        from dashboard.payments.redaction import sanitize
        st.code(sanitize(info.detail), language="text")
    if show_demo_button:
        st.button("Switch to Demonstration mode", key=f"err_demo_{info.kind}", on_click=lambda: st.session_state.update(payments_demo_mode=True))


def fmt_run_counts(counts: dict) -> str:
    names = {"comparison": ("comparison", "comparisons"), "run": ("single-policy run", "single-policy runs"), "sensitivity": ("sensitivity analysis", "sensitivity analyses")}
    parts = [f"{n} {names.get(k, (k, k + 's'))[0 if n == 1 else 1]}" for k, n in counts.items() if n]
    return " · ".join(parts) or "no runs"


def fmt_ts(ts: Optional[str]) -> str:
    return (ts or "—").replace("T", " ")[:19] + (" UTC" if ts else "")


def df_from_rows(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def json_expander(label: str, obj: Any, *, expanded: bool = False) -> None:
    with st.expander(label, expanded=expanded):
        st.json(obj if isinstance(obj, (dict, list)) else {"value": obj}, expanded=False)


def plotly_layout(**kw: Any) -> dict:
    return {**PLOTLY_LAYOUT, **kw}


STATE_STYLE = {"COMPLETE": (TEAL, "✔"), "DONE": (TEAL, "✔"), "VERIFIED": (TEAL, "✔"), "IN_PROGRESS": (CORAL, "◐"), "INTERRUPTED": (CORAL, "⚠"),
               "NOT_STARTED": (MUTED, "○")}


def page_intro(kicker: str, title: str, description: str, sources: list[DataSource], note: str = "") -> None:
    """Consistent top-of-page block for every section: kicker, title, one-line purpose, then the data-source label(s)."""
    section_header(kicker, title, color=SOURCE_STYLE[sources[0]]["color"] if sources and sources[0] != DataSource.SYSTEM else TEAL)
    st.markdown(f"<div style='color:{MUTED}; margin:-0.4rem 0 0.5rem 0;'>{description}</div>", unsafe_allow_html=True)
    for src in sources:
        source_banner(src, note if src == sources[0] else "")


def render_workflow_indicator(steps: list[dict], current: str, *, detailed: bool = False) -> None:
    """End-to-end workflow strip: the five sections as pipeline stages, each coloured by the data source it produces and showing
    a status derived from persisted data. `current` highlights where the user is."""
    boxes = []
    for i, step in enumerate(steps, 1):
        state, note = step["state"]
        color, mark = STATE_STYLE.get(state, (MUTED, "○"))
        src = SOURCE_STYLE[step["source"]]
        is_cur = step["section"] == current
        border = f"2px solid {src['color']}" if is_cur else f"1px solid {BORDER}"
        bg = f"{src['color']}14" if is_cur else SURFACE
        desc = f"<div style='font-size:0.74rem; color:{MUTED}; margin-top:0.25rem;'>{step['does']}</div>" if detailed else ""
        boxes.append(
            f"<div data-step='{step['key']}' data-state='{state}' data-current='{str(is_cur).lower()}' style='flex:1; min-width:150px; border:{border}; "
            f"border-top:5px solid {src['color']}; border-radius:10px; background:{bg}; padding:0.55rem 0.7rem;'>"
            f"<div style='font-size:0.68rem; font-weight:700; letter-spacing:0.05em; color:{src['color']};'>{i} · {src['mark']} {src['label']}</div>"
            f"<div style='font-size:1.0rem; font-weight:700; color:{PRIMARY_TEXT};'>{step['title']}</div>"
            f"<div style='font-size:0.78rem; color:{color}; font-weight:650;'>{mark} {state.replace('_', ' ')}"
            f"{(' · ' + note) if note else ''}</div>{desc}</div>")
    arrow = f"<div style='align-self:center; color:{MUTED}; font-size:1.2rem; padding:0 0.15rem;'>→</div>"
    st.markdown("<div data-testid='workflow-indicator' style='display:flex; gap:0.3rem; align-items:stretch; flex-wrap:wrap; margin:0.2rem 0 0.9rem 0;'>"
                + arrow.join(boxes) + "</div>", unsafe_allow_html=True)


def render_pipeline_diagram() -> None:
    """Static architecture strip: who does what, coloured by data source."""
    def box(label: str, sub: str, src: DataSource | None) -> str:
        c = SOURCE_STYLE[src]["color"] if src else MUTED
        tag = f"<div style='font-size:0.62rem; font-weight:700; color:{c}; letter-spacing:0.05em;'>{SOURCE_STYLE[src]['mark']} {SOURCE_STYLE[src]['label']}</div>" if src else \
              f"<div style='font-size:0.62rem; font-weight:700; color:{MUTED}; letter-spacing:0.05em;'>INPUT / OUTPUT</div>"
        return (f"<div style='flex:1; min-width:120px; border:1px solid {c}66; border-left:4px solid {c}; border-radius:8px; background:{SURFACE}; padding:0.45rem 0.6rem;'>"
                f"{tag}<div style='font-weight:700; color:{PRIMARY_TEXT}; font-size:0.9rem;'>{label}</div><div style='font-size:0.72rem; color:{MUTED};'>{sub}</div></div>")
    arrow = f"<div style='align-self:center; color:{MUTED}; padding:0 0.1rem;'>→</div>"
    parts = [box("Merchant objective", "goals + hard constraints", None),
             box("Gemini agent", "plans experiments, proposes policies", DataSource.AGENT_DERIVED),
             box("MCP tools", "the only interface to the simulation", DataSource.SYSTEM),
             box("Simulation engine", "seeded Monte Carlo of payment lifecycles", DataSource.SIMULATED),
             box("Deterministic evaluator", "scores · rejects infeasible · selects", DataSource.AGENT_DERIVED),
             box("PayPal Sandbox", "real order → approve → capture → verify", DataSource.SANDBOX_ACTUAL)]
    st.markdown("<div data-testid='pipeline-diagram' style='display:flex; gap:0.2rem; align-items:stretch; flex-wrap:wrap; margin:0.3rem 0 0.8rem 0;'>"
                + arrow.join(parts) + "</div>", unsafe_allow_html=True)
