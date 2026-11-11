"""MCP <-> Gemini function-calling adapter.

Gemini function calling and MCP are separate interfaces. This adapter:
  1. discovers tools from the MCP server through an independent MCP client,
  2. converts their JSON Schemas into Gemini FunctionDeclarations (a conservative subset the Gemini
     Schema type accepts),
  3. executes Gemini's function-call requests through the MCP client,
  4. returns the real MCP results (compacted for the LLM; full results are kept by the caller).
It never imports the simulation engine.
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from fastmcp import Client
from fastmcp.client.transports import StdioTransport
from google.genai import types

from agent.payments.models import AgentError

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_PASS_KEYS = {"description", "enum", "required", "nullable", "minimum", "maximum", "format"}
_RENAME = {"minItems": "minItems", "maxItems": "maxItems", "minLength": "minLength", "maxLength": "maxLength"}


def _resolve(node: Any, defs: dict[str, Any]) -> Any:
    if isinstance(node, dict) and "$ref" in node:
        name = node["$ref"].split("/")[-1]
        return {**defs.get(name, {}), **{k: v for k, v in node.items() if k != "$ref"}}
    return node


def to_gemini_schema(node: Any, defs: Optional[dict[str, Any]] = None, hidden: frozenset[str] = frozenset(),
                     _depth: int = 0) -> dict[str, Any]:
    """Convert a JSON Schema fragment (as produced by FastMCP/pydantic) into the Gemini Schema subset.

    Dropped (unsupported by Gemini's Schema): $defs/$ref (inlined), additionalProperties, title, default,
    exclusiveMinimum/Maximum (approximated by minimum/maximum), anyOf unions (null removed -> nullable;
    const-unions -> enum; otherwise the first non-null branch). `hidden` removes top-level properties.
    """
    if _depth > 12:
        raise AgentError("schema too deeply nested to convert")
    defs = defs if defs is not None else dict(node.get("$defs", {})) if isinstance(node, dict) else {}
    node = _resolve(node, defs)
    out: dict[str, Any] = {}

    if "anyOf" in node or "oneOf" in node:
        branches = [_resolve(b, defs) for b in node.get("anyOf", node.get("oneOf", []))]
        non_null = [b for b in branches if b.get("type") != "null"]
        if len(non_null) < len(branches):
            out["nullable"] = True
        if non_null and all("const" in b for b in non_null):
            out["type"] = "string" if all(isinstance(b["const"], str) for b in non_null) else "number"
            out["enum"] = [str(b["const"]) for b in non_null]
        elif non_null:
            out.update(to_gemini_schema(non_null[0], defs, frozenset(), _depth + 1))
        for k in ("description",):
            if k in node:
                out[k] = node[k]
        return out

    t = node.get("type")
    if isinstance(t, list):  # ["string","null"]
        if "null" in t:
            out["nullable"] = True
        t = next((x for x in t if x != "null"), "string")
    if "const" in node:
        out["enum"], t = [str(node["const"])], t or "string"
    if t:
        out["type"] = t
    for k, v in node.items():
        if k in _PASS_KEYS and k != "required" and k != "enum":
            out[k] = v
        elif k == "enum":
            out["enum"] = [str(x) for x in v]
        elif k in _RENAME:
            out[_RENAME[k]] = v
    if "exclusiveMinimum" in node and "minimum" not in out:
        out["minimum"] = node["exclusiveMinimum"]
    if "exclusiveMaximum" in node and "maximum" not in out:
        out["maximum"] = node["exclusiveMaximum"]
    if t == "array" and "items" in node:
        out["items"] = to_gemini_schema(node["items"], defs, frozenset(), _depth + 1)
    if t == "object":
        props = {k: to_gemini_schema(v, defs, frozenset(), _depth + 1) for k, v in node.get("properties", {}).items() if k not in hidden}
        if props:
            out["properties"] = props
        req = [r for r in node.get("required", []) if r in props]
        if req:
            out["required"] = req
    return out


@dataclass
class McpToolInfo:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    ok: bool
    data: Any  # structured MCP result, or error text
    duration_ms: float


def make_stdio_client(server_module: str, db_path: Optional[str] = None, timeout_s: float = 300.0) -> Client:
    env = {**os.environ, **({"INOEMA_PAYMENT_DB": db_path} if db_path else {})}
    return Client(StdioTransport(command=sys.executable, args=["-m", server_module], cwd=ROOT, env=env), timeout=timeout_s)


class McpToolAdapter:
    def __init__(self, client_factory: Callable[[], Client], hidden_tools: frozenset[str] = frozenset(),
                 hidden_params: frozenset[str] = frozenset(), tool_timeout_s: float = 300.0):
        self._factory, self.hidden_tools, self.hidden_params = client_factory, hidden_tools, hidden_params
        self.timeout = tool_timeout_s
        self._client: Optional[Client] = None
        self.tools: dict[str, McpToolInfo] = {}

    async def __aenter__(self) -> "McpToolAdapter":
        self._client = self._factory()
        await self._client.__aenter__()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc)

    async def discover(self) -> list[McpToolInfo]:
        assert self._client is not None, "adapter not entered"
        raw = await asyncio.wait_for(self._client.list_tools(), self.timeout)
        self.tools = {t.name: McpToolInfo(t.name, t.description or "", t.input_schema) for t in raw}
        return list(self.tools.values())

    def visible_tools(self) -> list[McpToolInfo]:
        return [t for t in self.tools.values() if t.name not in self.hidden_tools]

    def declarations(self) -> list[types.FunctionDeclaration]:
        """Gemini function declarations for the visible MCP tools (validated through the SDK's Schema type)."""
        decls = []
        for t in self.visible_tools():
            schema = to_gemini_schema(t.input_schema, hidden=self.hidden_params)
            schema.setdefault("type", "object")
            decls.append(types.FunctionDeclaration(name=t.name, description=t.description[:1024],
                                                   parameters=types.Schema.model_validate(schema)))
        return decls

    async def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Execute a tool through MCP. Errors (validation, tool failure, timeout) become ok=False results."""
        assert self._client is not None, "adapter not entered"
        # hidden_tools only controls what the LLM is shown/allowed (see the agent's guard); the harness may still call them.
        if name not in self.tools:
            return ToolResult(False, f"unknown tool {name!r}", 0.0)
        t0 = time.monotonic()
        try:
            res = await asyncio.wait_for(self._client.call_tool(name, args), self.timeout)
            data = res.structured_content if res.structured_content is not None else res.data
            return ToolResult(True, data, (time.monotonic() - t0) * 1000)
        except asyncio.TimeoutError:
            return ToolResult(False, f"tool {name} timed out after {self.timeout}s", (time.monotonic() - t0) * 1000)
        except Exception as e:  # noqa: BLE001 -- surfaced to the LLM as a tool error
            return ToolResult(False, f"{type(e).__name__}: {str(e)[:600]}", (time.monotonic() - t0) * 1000)


# --- compaction of full MCP results for the LLM context ----------------------------------------------------

def _policy_row(name: str, summary: dict[str, Any]) -> dict[str, Any]:
    rr, pm = summary["recovery_rate"], summary["per_trial_means"]
    return {"recovery_rate": round(rr["value"], 4), "recovery_rate_ci95": [round(x, 4) for x in rr["ci95_wilson"]],
            "recovered_revenue_per_trial": round(pm["recovered_revenue"], 1),
            "avg_recovery_minutes": round(summary["avg_recovery_time_minutes"]["value"], 1),
            "total_retries": round(pm["total_retries"]), "unnecessary_retries": round(pm["unnecessary_retries"]),
            "total_cost": round(pm["total_processing_cost"], 1), "buyer_actions": round(pm["buyer_actions_requested"])}


def compact_result(tool: str, data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    if tool == "compare_recovery_policies":
        return {"best_policy": data["best_policy"], "best_policy_feasible": data["best_policy_feasible"],
                "n_trials": data["n_trials"], "n_transactions": data["n_transactions"], "master_seed": data["master_seed"],
                "ranking": [{"rank": r["rank"], "policy": r["policy"], "feasible": r["feasible"], "score": round(r["score_mean"], 4),
                             "violations": r["violations"][:4], **_policy_row(r["policy"], data["policies"][r["policy"]]["summary"])}
                            for r in data["ranking"]],
                "paired_score_difference_vs_best": {k: {"mean": round(v["mean_difference"], 4), "significant": v["ci_excludes_zero"]}
                                                    for k, v in data["paired_score_difference_vs_best"].items()},
                "note": "score/feasibility computed deterministically by the evaluator; SIMULATED, assumed distributions"}
    if tool == "run_payment_simulation":
        ev = data["evaluation"]
        return {"policy": data["policy_definition"]["name"], "feasible": ev["feasible"], "score": round(ev["score_mean"], 4),
                "violations": ev["violations"][:4], **_policy_row("", data["summary"]), "master_seed": data["master_seed"]}
    if tool == "run_sensitivity_analysis":
        return {"baseline_winner": data["baseline"]["winner"], "robustness": {k: data["robustness"][k] for k in
                ("label", "winner_stable_share", "flip_conditions")},
                "most_sensitive_parameters": data["most_sensitive_parameters"],
                "parameter_summary": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                                       if k in ("parameter", "n_winner_changes", "n_ranking_changes", "min_margin", "flip_conditions")}
                                      for s in data["parameter_summary"]],
                "note": "ranges are ASSUMED exploration ranges, not forecasts"}
    if tool == "generate_payment_scenarios":
        return {"scenarios": [{k: s[k] for k in ("scenario_id", "name", "n_transactions", "description")} for s in data["scenarios"]]}
    if tool == "get_payment_experiment_results":
        return {"runs": [{"run_id": r["run_id"], "kind": r["kind"], "scenario_id": r["scenario_id"], "n_trials": r["n_trials"]}
                         for r in data["runs"]]}
    return data
