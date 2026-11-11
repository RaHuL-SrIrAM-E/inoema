"""Spike 2: independent MCP client. Spawns the server as a separate stdio
subprocess, discovers tools, calls one, and cross-checks against a direct
engine call. Run:  python -m spikes.mcp_sim.client"""
from __future__ import annotations

import asyncio
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

TIMEOUT_S = 120

ARGS = {"params": {
    "release_schedule": [{"time_minutes": 90, "count": 2000}],
    "n_workers": 8, "reconciliation_close": 210, "initial_queue_depth": 350,
    "posting_time_median": 0.25, "n_trials": 100, "master_seed": 0}}


def make_client() -> Client:
    return Client(StdioTransport(command=sys.executable, args=["-m", "spikes.mcp_sim.server"],
                                 cwd=ROOT), timeout=TIMEOUT_S)


async def main() -> int:
    async with make_client() as c:
        tools = await c.list_tools()
        print("DISCOVERED TOOLS:", [t.name for t in tools])
        res = await asyncio.wait_for(c.call_tool("simulate_release_schedule", ARGS), TIMEOUT_S)
        print("is_error:", res.is_error)
        print("structured:", json.dumps(res.structured_content, indent=2))
        res2 = await c.call_tool("simulate_release_schedule", ARGS)
        print("deterministic repeat:", res2.structured_content == res.structured_content)
        try:
            bad = {"params": {**ARGS["params"], "n_workers": 0}}
            await c.call_tool("simulate_release_schedule", bad)
            print("validation: NOT REJECTED (unexpected)")
        except Exception as e:  # noqa: BLE001
            print("validation rejects n_workers=0:", type(e).__name__)
    # cross-check against direct engine call (client-side check only)
    sys.path.insert(0, ROOT)
    from engine.model import SimConfig
    from engine.montecarlo import run_monte_carlo
    p = ARGS["params"]
    cfg = SimConfig(batch_size=2000, release_schedule=[(90, 2000)], n_workers=8,
                    reconciliation_close=210, initial_queue_depth=350, posting_time_median=0.25)
    direct = run_monte_carlo(cfg, 100, 0)
    match = abs(direct.p_shortfall - res.structured_content["p_shortfall"]) < 1e-12
    print("matches direct engine call:", match, direct.p_shortfall)
    return 0 if match else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
