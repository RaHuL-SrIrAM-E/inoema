"""PayPal Sandbox workspace: REAL Sandbox API operations only, behind explicit user actions.

Business rules (state checks, idempotency, never capturing an unapproved order, retry rules) live in paypal.execution; this page only
collects explicit confirmations and displays the persisted evidence. Nothing here is simulation, and nothing from the simulation is shown here.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.payments import data, sandbox_service as svc, status
from dashboard.payments.redaction import sanitize
from dashboard.payments.sources import DataSource
from dashboard.payments.ui import demo_mode, fmt_ts, page_intro, status_pill
from paypal.models import ALLOWED_CURRENCIES, PayPalApiError, PayPalConfigError, SandboxViolation


def _evidence(execution_id: str) -> None:
    det = data.sandbox_detail(execution_id)
    if det is None:
        st.info("No evidence recorded for this execution.")
        return
    ex = det["execution"]
    last_dbg = next((c["paypal_debug_id"] for c in reversed(det["api_calls"]) if c["paypal_debug_id"]), None)
    verified = any(t["trigger"] == "CAPTURE_VERIFIED" for t in det["transitions"])
    a, b, c, d = st.columns(4)
    a.metric("Execution status", ex["status"])
    b.metric("PayPal order ID", ex["order_id"] or "—")
    c.metric("Capture ID", ex["capture_id"] or "—")
    d.metric("Capture verified", "Yes" if verified else "No" if ex["status"] == "COMPLETED" else "—")
    st.caption(f"● SANDBOX · ACTUAL · amount {ex['amount']} {ex['currency']} · reference `{ex['reference_id']}` · policy `{ex['policy_name']}` · last PayPal debug id `{last_dbg or '—'}`")
    if ex["status"] == "AWAITING_BUYER" and ex["approval_url"]:
        st.warning("**Buyer action required.** The order has NOT been captured. Open the approval page, log in with a Sandbox *personal* (buyer) account and approve, then use step 3.")
        st.link_button("Open PayPal Sandbox approval page ↗", ex["approval_url"], key=f"sb_link_{execution_id}")
        st.code(ex["approval_url"], language="text")
    if ex["status"] == "TERMINATED":
        st.error(f"Execution terminated: {(ex.get('last_error') or {}).get('reason', 'see history')}")
    st.markdown("**Execution history (state transitions)**")
    st.dataframe(pd.DataFrame([{"#": t["seq"], "Time": fmt_ts(t["ts"]), "From": t["from_state"], "To": t["to_state"], "Trigger": t["trigger"],
                                "Detail": str(t["detail"])[:200]} for t in det["transitions"]]), hide_index=True, width="stretch", key=f"sb_trans_{execution_id}")
    st.markdown("**PayPal API calls (evidence; secrets redacted)**")
    st.dataframe(pd.DataFrame([{"Time": fmt_ts(c["ts"]), "Operation": c["operation"], "HTTP": c["status"], "PayPal debug ID": c["paypal_debug_id"],
                                "Idempotency key": (c["request_id"] or "")[:13] + ("…" if c["request_id"] else ""), "Latency ms": round(c["latency_ms"] or 0)}
                               for c in det["api_calls"]]), hide_index=True, width="stretch", key=f"sb_calls_{execution_id}")
    if det["validation_records"]:
        st.markdown("**Predicted vs observed** — single Sandbox observations, never rates and never combined with simulation")
        st.dataframe(pd.DataFrame([{"Operation": r["api_operation"], "Observed": r["actual_api_status"], "Error category": r["observed_error_category"],
                                    "Predicted action": r["predicted_action"], "Actual action": r["actual_action"], "Outcome": r["validation_outcome"]} for r in det["validation_records"]]),
                     hide_index=True, width="stretch", key=f"sb_val_{execution_id}")


def render() -> None:
    page_intro("PAYMENTS · PAYPAL SANDBOX", "PayPal Sandbox", "Create a genuine Sandbox order, stop for buyer approval, then capture and verify. Each step is an explicit action; "
               "a capture is never attempted on an unapproved order.", [DataSource.SANDBOX_ACTUAL],
               note="REAL calls to the PayPal Sandbox API. Not simulation: hypothetical results live in Experiments and never appear here.")

    ps = status.paypal_status()
    c1, c2 = st.columns([1, 2])
    with c1:
        status_pill("PayPal Sandbox", ps.state, ps.detail)
    with c2:
        if st.button("Test OAuth connection (Sandbox)", key="sb_probe", disabled=ps.state == "NOT_CONFIGURED"):
            p = status.probe_paypal()
            (st.success if p.state == "OK" else st.error)(f"{p.detail}" + (f" · debug id `{(p.extra or {}).get('debug_id')}`" if p.extra else ""))
    configured = ps.state == "OK"
    if not configured:
        st.warning("**PayPal Sandbox credentials are not configured**, so no Sandbox operation can run. Set `PAYPAL_CLIENT_ID` and `PAYPAL_CLIENT_SECRET` "
                   "(Sandbox REST app) in `.env`, create a Sandbox *business* and *personal* account, and restart. See docs/PAYPAL_SANDBOX_INTEGRATION.md §7. "
                   "Nothing is simulated or faked here.")
    if demo_mode():
        st.info("Demonstration mode is ON: Sandbox actions are disabled. Existing Sandbox evidence (if any) is shown below.")

    st.markdown("#### 1 · Policy and payment")
    pols = svc.available_policies()
    pick = st.selectbox("Recovery policy to apply", [p["key"] for p in pols], format_func=lambda k: next(p["label"] for p in pols if p["key"] == k), key="sb_policy")
    chosen = next(p for p in pols if p["key"] == pick)
    if chosen["origin"] == "AGENT_SELECTED" and not chosen.get("complete"):
        st.warning(f"This policy comes from an execution that is **{chosen['state']}**, not a complete autonomous run. It satisfied the hard constraints in the evaluator, "
                   "but treat it as a demonstration input.")
    if chosen["origin"] == "DEMO":
        st.caption("⚠ No agent-selected policy is available (or you picked the demo one): this built-in demo policy is NOT the result of an optimisation.")
    a, b, c = st.columns(3)
    amount = a.text_input("Amount", "25.00", key="sb_amount")
    currencies = sorted(ALLOWED_CURRENCIES)
    currency = b.selectbox("Currency", currencies, index=currencies.index("USD"), key="sb_currency")
    if "sb_reference" not in st.session_state:
        st.session_state["sb_reference"] = svc.new_reference_id()
    ref = c.text_input("Reference ID (idempotency key for this payment)", key="sb_reference")

    st.markdown("#### 2 · Create the Sandbox order")
    ok1 = st.checkbox("I understand this calls the real PayPal **Sandbox** API and creates a Sandbox order", key="sb_confirm_create")
    if st.button("Create Sandbox order", type="primary", key="sb_create", disabled=not (configured and ok1) or demo_mode()):
        try:
            res = svc.create_order(chosen["policy"], chosen["experiment_id"], amount, currency, ref)
            st.session_state["sb_exec_id"] = res.execution_id
            st.success(f"Execution {res.status.value}: order `{res.order_id}`")
        except ValueError as e:
            st.error(f"Invalid input: {e}")
        except (PayPalConfigError, SandboxViolation) as e:
            st.error(f"Configuration error: {e}")
        except PayPalApiError as e:
            st.error(f"PayPal API error: HTTP {e.status} {e.name} (debug id `{e.debug_id}`)")

    ex_id = st.session_state.get("sb_exec_id")
    execs = svc.data.list_sandbox_executions()
    cur = next((e for e in execs if e["execution_id"] == ex_id), None)
    st.markdown("#### 3 · Buyer approval → capture → verification")
    awaiting = bool(cur and svc.status_is_awaiting_buyer(cur["status"]))
    ok2 = st.checkbox("I approved this order in PayPal Sandbox as the buyer", key="sb_confirm_approved", disabled=not awaiting)
    if st.button("Check order state and capture", key="sb_capture", disabled=not (configured and awaiting and ok2) or demo_mode(),
                 help="Re-reads the real order from PayPal. Capture only happens if PayPal reports it APPROVED; otherwise the order stays awaiting buyer."):
        try:
            res = svc.check_and_capture(chosen["policy"], chosen["experiment_id"], cur["reference_id"])
            (st.success if res.status.value == "COMPLETED" else st.info)(f"Execution {res.status.value}" + (f" · capture `{res.capture_id}`" if res.capture_id else " · not captured (buyer has not approved or a failure occurred)"))
        except (PayPalConfigError, SandboxViolation) as e:
            st.error(f"Configuration error: {e}")
        except PayPalApiError as e:
            st.error(f"PayPal API error: HTTP {e.status} {e.name} (debug id `{e.debug_id}`)")
    if not awaiting:
        st.caption("Step 3 unlocks when the current execution is awaiting buyer approval." if cur is None or cur["status"] != "COMPLETED" else "This execution is complete and verified.")

    st.markdown("#### Execution details and history")
    if not execs:
        st.info("No Sandbox executions have been recorded. Nothing on this page is simulated.")
        return
    sel = st.selectbox("Sandbox execution", [e["execution_id"] for e in execs], index=([e["execution_id"] for e in execs].index(ex_id) if ex_id in [e["execution_id"] for e in execs] else 0),
                       format_func=lambda i: next(f"{e['reference_id']} · {e['status']} · {e['amount']} {e['currency']} · {fmt_ts(e['created_at'])}" for e in execs if e["execution_id"] == i), key="sb_select")
    _evidence(sel)
