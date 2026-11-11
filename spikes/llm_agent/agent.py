"""Spike 3: minimal runtime agent. The LLM (Anthropic API) is given the MCP
tools discovered from the Spike 2 server and decides what to call. The agent
imports NOTHING from engine/; all simulation happens behind MCP.

Run:  python -m spikes.llm_agent.agent
Env:  ANTHROPIC_API_KEY (see .env.example); optional SPIKE_LLM_MODEL.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone

import anthropic
from fastmcp import Client

from spikes.envutil import ROOT, load_env
from spikes.mcp_sim.client import make_client

MAX_TURNS = 8
TRANSCRIPT = os.path.join(ROOT, "data", "spike_llm_transcript.jsonl")

SYSTEM = (
    "You are a payments-operations analyst. You may ONLY learn about risk by calling the provided "
    "simulation tools; never guess numbers. Use few tool calls (at most 4). Finish with a short "
    "decision stating which schedule to use, citing the simulated p_shortfall for each option you tested."
)
TASK = (
    "A merchant must settle a batch of 2000 payments before a deadline at t=210 minutes. There are "
    "8 workers, an existing backlog of 350 items, and posting_time_median=0.25. The current plan "
    "releases the whole batch at t=150. Decide whether to keep it or change it. Policy: release "
    "schedules are acceptable only if simulated p_shortfall < 0.05. Use n_trials=100 and master_seed=0."
)


def to_anthropic_tools(mcp_tools) -> list[dict]:
    return [{"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
            for t in mcp_tools]


def log(event: dict) -> None:
    os.makedirs(os.path.dirname(TRANSCRIPT), exist_ok=True)
    with open(TRANSCRIPT, "a") as f:
        f.write(json.dumps({"ts": datetime.now(timezone.utc).isoformat(), **event}, default=str) + "\n")


async def run_agent(model: str) -> int:
    llm = anthropic.AsyncAnthropic(timeout=60.0, max_retries=2)  # reads ANTHROPIC_API_KEY
    tool_calls = 0
    async with make_client() as mcp:
        tools = to_anthropic_tools(await mcp.list_tools())
        print("MCP tools offered to LLM:", [t["name"] for t in tools])
        messages = [{"role": "user", "content": TASK}]
        for turn in range(MAX_TURNS):
            resp = await llm.messages.create(model=model, max_tokens=2000, system=SYSTEM,
                                             tools=tools, messages=messages)
            log({"turn": turn, "stop_reason": resp.stop_reason, "request_id": getattr(resp, "_request_id", None),
                 "content": [b.model_dump() for b in resp.content]})
            messages.append({"role": "assistant", "content": resp.content})
            for b in resp.content:
                if b.type == "text" and b.text.strip():
                    print(f"[assistant] {b.text.strip()}")
            if resp.stop_reason != "tool_use":
                print(f"\nDONE after {turn + 1} LLM turn(s); LLM-initiated MCP tool calls: {tool_calls}")
                return 0 if tool_calls > 0 else 1
            results = []
            for b in resp.content:
                if b.type != "tool_use":
                    continue
                tool_calls += 1
                print(f"[tool_use] {b.name}({json.dumps(b.input)})")
                try:
                    r = await asyncio.wait_for(mcp.call_tool(b.name, b.input), 120)
                    payload, is_err = json.dumps(r.structured_content), False
                except Exception as e:  # noqa: BLE001 - surfaced to the LLM
                    payload, is_err = f"{type(e).__name__}: {e}", True
                print(f"[tool_result{' ERROR' if is_err else ''}] {payload[:400]}")
                log({"turn": turn, "tool": b.name, "input": b.input, "result": payload, "is_error": is_err})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": payload, "is_error": is_err})
            messages.append({"role": "user", "content": results})
    print("ERROR: exceeded MAX_TURNS without a final answer", file=sys.stderr)
    return 1


def main() -> int:
    load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set (see .env.example)", file=sys.stderr)
        return 2
    model = os.environ.get("SPIKE_LLM_MODEL", "claude-sonnet-5-5")
    try:
        return asyncio.run(run_agent(model))
    except anthropic.APIError as e:
        print(f"ERROR: Anthropic API: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
