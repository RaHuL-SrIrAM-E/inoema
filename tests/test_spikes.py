"""Offline tests for Phase 1 spike plumbing.

NOTE: PayPal tests use httpx.MockTransport purely to test OUR client logic
(redaction, error mapping, verification). They are NOT evidence that PayPal
behaves this way -- live behaviour is documented in docs/PHASE1_SPIKE_RESULTS.md.
"""
import ast
import asyncio
import json
import os

import httpx
import pytest

from spikes.paypal.client import ApiLog, PayPalError, PayPalSandbox, approval_link, redact, verify_capture

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _pp(handler):
    return PayPalSandbox("id", "secret", log=ApiLog(":memory:"), transport=httpx.MockTransport(handler))


def test_refuses_live_base_url():
    with pytest.raises(ValueError):
        PayPalSandbox("a", "b", base_url="https://api-m.paypal.com")


def test_redact_and_logging_excludes_secrets():
    def handler(req):
        if req.url.path == "/v1/oauth2/token":
            return httpx.Response(200, json={"access_token": "TOPSECRET", "expires_in": 300})
        return httpx.Response(201, json={"id": "O1", "status": "PAYER_ACTION_REQUIRED",
                                         "links": [{"rel": "payer-action", "href": "https://x/approve"}]})
    pp = _pp(handler)
    order = pp.create_order()
    assert approval_link(order) == "https://x/approve"
    dump = json.dumps(pp.log.conn.execute("select * from paypal_api_calls").fetchall())
    assert "TOPSECRET" not in dump and "secret" not in dump.replace("REDACTED", "")
    assert redact({"a": {"access_token": "x"}}) == {"a": {"access_token": "***REDACTED***"}}


def test_api_error_mapping():
    def handler(req):
        return httpx.Response(401, json={"error": "invalid_client", "error_description": "bad"},
                              headers={"paypal-debug-id": "abc"})
    with pytest.raises(PayPalError) as e:
        _pp(handler).get_access_token()
    assert e.value.status == 401 and e.value.debug_id == "abc"


def test_network_error_mapping():
    def handler(req):
        raise httpx.ConnectTimeout("boom")
    with pytest.raises(PayPalError) as e:
        _pp(handler).get_access_token()
    assert e.value.name == "NETWORK_ERROR"


def test_verify_capture():
    cap = {"id": "O1", "status": "COMPLETED",
           "purchase_units": [{"payments": {"captures": [{"id": "C1"}]}}]}
    det = {"id": "C1", "status": "COMPLETED", "amount": {"value": "10.00", "currency_code": "USD"}}
    assert verify_capture("O1", cap, det, "10.00", "USD") == []
    assert verify_capture("O1", cap, {**det, "amount": {"value": "9.00", "currency_code": "USD"}}, "10.00", "USD")


def test_mcp_tool_discovery_and_real_execution():
    from engine.model import SimConfig
    from engine.montecarlo import run_monte_carlo
    from spikes.mcp_sim.client import ARGS, make_client

    async def go():
        async with make_client() as c:
            tools = await c.list_tools()
            res = await c.call_tool("simulate_release_schedule", ARGS)
            return tools, res.structured_content

    tools, out = asyncio.run(go())
    # Phase 2 extended the same server with payment tools; the legacy tool must still be present.
    assert "simulate_release_schedule" in [t.name for t in tools]
    cfg = SimConfig(batch_size=2000, release_schedule=[(90, 2000)], n_workers=8,
                    reconciliation_close=210, initial_queue_depth=350, posting_time_median=0.25)
    assert out["p_shortfall"] == run_monte_carlo(cfg, 100, 0).p_shortfall
    assert out["synthetic"] is True


def test_llm_agent_does_not_import_engine():
    src = open(os.path.join(ROOT, "spikes", "llm_agent", "agent.py")).read()
    mods = [n.module or "" for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ImportFrom)]
    mods += [a.name for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Import) for a in n.names]
    assert not any(m == "engine" or m.startswith("engine.") for m in mods)
