"""Service status checks (no Streamlit imports). Cheap checks run automatically; anything that spends API quota or hits an external
service only runs on an explicit probe_* call."""
from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from agent.payments.models import SERVER_MODULE, AgentConfig, MissingCredentialsError
from dashboard.payments import data, paths
from dashboard.payments.errors import classify_error


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ServiceStatus:
    name: str
    state: str  # OK | DEGRADED | DOWN | NOT_CONFIGURED | NOT_CHECKED
    detail: str
    checked_at: Optional[str] = None
    extra: Optional[dict] = None


def check_simulation_engine() -> ServiceStatus:
    """Run a tiny REAL Monte Carlo (300 txns x 1 trial) through the engine's public API."""
    try:
        from engine.payments.recovery_policy import intelligent_recovery
        from engine.payments.scenarios import build_scenario
        from engine.payments.simulator import run_payment_monte_carlo
        t0 = time.monotonic()
        r = run_payment_monte_carlo(build_scenario("baseline", 300), intelligent_recovery(), 1, 0)
        ms = (time.monotonic() - t0) * 1000
        return ServiceStatus("Simulation engine", "OK", f"300-transaction Monte Carlo completed in {ms:.0f} ms (seeded, reproducible)", _now(),
                             {"label": r.label})
    except Exception as e:  # noqa: BLE001
        return ServiceStatus("Simulation engine", "DOWN", f"{type(e).__name__}: {e}", _now())


def check_mcp_server(server_module: Optional[str] = None, timeout_s: float = 60.0) -> ServiceStatus:
    """Start the FastMCP server as a subprocess, discover its tools through an independent MCP client, and stop it."""
    from agent.payments.agent import HIDDEN_TOOLS, REQUIRED_TOOLS
    from agent.payments.tool_adapter import McpToolAdapter, make_stdio_client
    mod = server_module or os.environ.get("INOEMA_MCP_SERVER_MODULE") or SERVER_MODULE

    async def go() -> list[str]:
        ad = McpToolAdapter(lambda: make_stdio_client(mod, paths.sim_db_path(), timeout_s), HIDDEN_TOOLS, frozenset({"max_work_units"}), timeout_s)
        async with ad:
            return [t.name for t in await ad.discover()]

    t0 = time.monotonic()
    try:
        names = asyncio.run(asyncio.wait_for(go(), timeout_s + 5))
        missing = sorted(REQUIRED_TOOLS - set(names))
        ms = (time.monotonic() - t0) * 1000
        if missing:
            return ServiceStatus("MCP server", "DEGRADED", f"started, but missing tools: {missing}", _now(), {"tools": names})
        return ServiceStatus("MCP server", "OK", f"{len(names)} tools discovered via an independent MCP client in {ms / 1000:.1f}s", _now(), {"tools": names, "module": mod})
    except Exception as e:  # noqa: BLE001
        return ServiceStatus("MCP server", "DOWN", f"{type(e).__name__}: {str(e)[:200]}", _now())


def gemini_status() -> ServiceStatus:
    """Configuration + last genuine execution. Makes NO API call (a probe would spend quota)."""
    try:
        cfg = AgentConfig.from_env()
    except MissingCredentialsError:
        last = data.list_agent_runs(provider="GEMINI_API")
        return ServiceStatus("Gemini integration", "NOT_CONFIGURED", "GEMINI_API_KEY is not set", _now(),
                             {"last_genuine_run": last[0] if last else None})
    last = data.list_agent_runs(provider="GEMINI_API")
    detail = f"API key configured; model `{cfg.model}` (set GEMINI_MODEL to change). Not probed automatically (probing spends quota)."
    state = "OK"
    if last and last[0]["termination_reason"] == "llm_error":
        state, detail = "DEGRADED", detail + " Last genuine run ended with an LLM/API error (see Traces)."
    return ServiceStatus("Gemini integration", state, detail, _now(), {"model": cfg.model, "last_genuine_run": last[0] if last else None})


def probe_gemini() -> ServiceStatus:
    """ONE real request to Gemini with the configured model. Explicit user action only."""
    try:
        cfg = AgentConfig.from_env()
    except MissingCredentialsError as e:
        return ServiceStatus("Gemini integration", "NOT_CONFIGURED", str(e), _now())
    from google.genai import types
    from agent.payments.llm import GeminiBackend

    async def go():
        t0 = time.monotonic()
        r = await GeminiBackend(cfg).generate([types.Content(role="user", parts=[types.Part(text="Reply with exactly: PONG")])], types.GenerateContentConfig())
        return r, (time.monotonic() - t0) * 1000

    try:
        r, ms = asyncio.run(go())
        text = "".join(p.text or "" for p in r.candidates[0].content.parts) if r.candidates else ""
        return ServiceStatus("Gemini integration", "OK", f"responded in {ms:.0f} ms; configured `{cfg.model}`, served `{r.model_version}`", _now(),
                             {"configured_model": cfg.model, "served_model": r.model_version, "reply": text.strip()[:40]})
    except Exception as e:  # noqa: BLE001
        info = classify_error(e)
        return ServiceStatus("Gemini integration", "DOWN", f"{info.title}: {info.detail[:300]}", _now(), {"error_kind": info.kind, "retry_after": info.retry_after})


def paypal_status() -> ServiceStatus:
    """Configuration only (no network). Reports the Sandbox-only guard state."""
    from paypal.models import PayPalConfig, PayPalConfigError, SandboxViolation
    try:
        cfg = PayPalConfig.from_env()
    except PayPalConfigError:
        return ServiceStatus("PayPal Sandbox", "NOT_CONFIGURED", "PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET are not set", _now())
    except SandboxViolation as e:
        return ServiceStatus("PayPal Sandbox", "DOWN", f"refused: {e}", _now())
    sb = data.sandbox_summary()
    return ServiceStatus("PayPal Sandbox", "OK", f"credentials configured for {cfg.base_url} (Sandbox-only guard active); not probed automatically", _now(),
                         {"api_calls_recorded": sb["n_api_calls"]})


def probe_paypal() -> ServiceStatus:
    """ONE real OAuth request to the PayPal Sandbox. Explicit user action only; the exchange is recorded as Sandbox evidence."""
    from paypal.models import PayPalApiError, PayPalConfig, PayPalConfigError, SandboxViolation
    from paypal.persistence import SandboxStore
    from dashboard.payments import sandbox_service
    try:
        cfg = PayPalConfig.from_env()
        store = SandboxStore(paths.sandbox_db_path(), secrets=cfg.secrets())
        client = sandbox_service.make_client(cfg, store)
        client.tokens.get_token()
        row = store.api_calls()[-1]
        return ServiceStatus("PayPal Sandbox", "OK", f"OAuth succeeded against {client.base} (token never shown)", _now(),
                             {"debug_id": row["paypal_debug_id"], "status": row["status"]})
    except (PayPalConfigError, SandboxViolation) as e:
        return ServiceStatus("PayPal Sandbox", "NOT_CONFIGURED", str(e), _now())
    except PayPalApiError as e:
        return ServiceStatus("PayPal Sandbox", "DOWN", f"OAuth failed: HTTP {e.status} {e.name}", _now(), {"debug_id": e.debug_id})
