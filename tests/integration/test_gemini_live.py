"""LIVE Gemini integration tests -- these call the real Gemini API and spend quota.

OPT-IN ONLY: having a key (in the environment or .env) is NOT enough; you must also set INOEMA_ALLOW_LIVE_GEMINI=1.
Run explicitly:   INOEMA_ALLOW_LIVE_GEMINI=1 python -m pytest tests/integration -m gemini_live -v -s
Without the opt-in flag (or without a key) they are SKIPPED (a skip is NOT a pass: it means the live API was not exercised).
Evidence of a genuine run: provider == 'GEMINI_API', at least one function call chosen by the model and
executed through MCP, request ids / traces persisted in the agent DB.
"""
import asyncio
import os

import pytest

from spikes.envutil import load_env

load_env()
KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
OPT_IN = os.environ.get("INOEMA_ALLOW_LIVE_GEMINI") == "1"
pytestmark = [pytest.mark.gemini_live,
              pytest.mark.skipif(not OPT_IN, reason="live Gemini tests are opt-in: set INOEMA_ALLOW_LIVE_GEMINI=1 (they call the real API and spend quota)"),
              pytest.mark.skipif(not KEY, reason="GEMINI_API_KEY not set: live Gemini API NOT exercised")]

from google.genai import types  # noqa: E402

from agent.payments.agent import HIDDEN_PARAMS, HIDDEN_TOOLS, PaymentAgent  # noqa: E402
from agent.payments.llm import GeminiBackend  # noqa: E402
from agent.payments.models import AgentConfig  # noqa: E402
from agent.payments.tool_adapter import McpToolAdapter, make_stdio_client  # noqa: E402

OBJECTIVE = ("Optimize payment recovery for a merchant processing 1,000 transactions. Maximize recovered revenue while minimizing "
             "recovery cost and latency. Do not exceed two automatic retries. Never automatically retry terminal failures.")


def test_gemini_accepts_converted_mcp_declarations_and_requests_a_function_call(tmp_path):
    cfg = AgentConfig.from_env()

    async def go():
        ad = McpToolAdapter(lambda: make_stdio_client(cfg.server_module, str(tmp_path / "sim.db")), HIDDEN_TOOLS, HIDDEN_PARAMS)
        async with ad:
            await ad.discover()
            decls = ad.declarations()
            resp = await GeminiBackend(cfg).generate(
                [types.Content(role="user", parts=[types.Part(text="Call generate_payment_scenarios with experiment_id 'demo' and one baseline scenario of 1000 transactions.")])],
                types.GenerateContentConfig(tools=[types.Tool(function_declarations=decls)],
                                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)))
            return resp, {d.name for d in decls}

    resp, names = asyncio.run(go())
    calls = resp.function_calls or []
    assert calls, "Gemini returned no function call"
    assert calls[0].name in names


def test_full_autonomous_run_with_real_gemini(tmp_path):
    cfg = AgentConfig.from_env(max_iterations=8, exploration_trials=4, final_trials=6, robustness_n_transactions=500,
                               robustness_trials=3, db_path=str(tmp_path / "sim.db"), agent_db_path=str(tmp_path / "agent.db"))
    lines: list[str] = []
    agent = PaymentAgent(cfg, progress=lambda m: (lines.append(m), print(m)))
    report = asyncio.run(agent.run(OBJECTIVE))
    print(report.to_markdown())
    assert report.llm["provider"] == "GEMINI_API"
    assert report.llm_activity["function_calls"] >= 1 and report.experiments["by_gemini"] >= 1  # Gemini itself drove MCP simulations
    assert report.selected_policy is not None and report.selected_policy["definition"]["retry_limit"] <= 2
    assert report.confidence["repeatability"]["verified"] is True
    assert agent.store.get_run(report.execution_id)["status"] == "COMPLETED"
