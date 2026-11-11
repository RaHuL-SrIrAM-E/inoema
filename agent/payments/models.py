"""Shared types for the autonomous payment-recovery optimisation agent.

This package never imports engine/ (the simulation). It only talks to the simulation through MCP tool
calls and works on the JSON those tools return. All numbers it reports come from MCP results.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

# The ONLY place a default model name appears. Override with GEMINI_MODEL. Availability of any
# particular model for a given key is not verified by this code.
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
SERVER_MODULE = "spikes.mcp_sim.server"  # FastMCP server module (stdio); override with INOEMA_MCP_SERVER_MODULE

SIM_DISCLAIMER = ("All figures are SIMULATED under ASSUMED distributions. Confidence statements describe the "
                  "simulation only; they are not confidence in real-world (or PayPal) performance.")


class MissingCredentialsError(RuntimeError):
    """No Gemini API key is configured."""


class AgentError(RuntimeError):
    """Unrecoverable agent/harness error."""


@dataclass(frozen=True)
class AgentConfig:
    api_key: str = field(default="", repr=False)  # NEVER persisted or logged
    model: str = DEFAULT_GEMINI_MODEL
    max_iterations: int = 10  # LLM turns in the exploration loop
    max_tool_calls: int = 24  # tool calls the LLM may make (MCP + local)
    exploration_budget_units: int = 8_000_000  # transactions x trials x policies the LLM may spend
    exploration_trials: int = 10
    final_trials: int = 20
    final_seed: int = 2026
    robustness_n_transactions: int = 5000
    robustness_trials: int = 6
    llm_timeout_s: float = 90.0
    tool_timeout_s: float = 300.0
    llm_retries: int = 3
    max_rate_limit_wait_s: float = 90.0  # longest server-advised rate-limit delay we will wait out (longer = give up, e.g. daily quota)
    initial_policy: str = "immediate_retry"  # the merchant's "current" policy (a baseline name)
    include_llm_explanation: bool = True
    server_module: str = SERVER_MODULE
    db_path: Optional[str] = None  # simulation DB (INOEMA_PAYMENT_DB)
    agent_db_path: Optional[str] = None  # agent-run DB

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None, **overrides) -> "AgentConfig":
        env = os.environ if env is None else env
        key = env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY") or ""
        if not key and not overrides.get("api_key"):
            raise MissingCredentialsError("GEMINI_API_KEY is not set (see .env.example). The agent will not run "
                                          "without credentials and never falls back to a fake model.")
        return cls(api_key=overrides.pop("api_key", key), model=overrides.pop("model", None) or env.get("GEMINI_MODEL") or DEFAULT_GEMINI_MODEL,
                   server_module=overrides.pop("server_module", None) or env.get("INOEMA_MCP_SERVER_MODULE") or SERVER_MODULE,
                   **overrides)

    def redacted(self) -> dict[str, Any]:
        """Config safe to persist: everything except the API key."""
        return {k: v for k, v in self.__dict__.items() if k != "api_key"}


@dataclass(frozen=True)
class ObjectiveSpec:
    """The merchant's objective as enforced by the deterministic evaluator."""
    text: str
    n_transactions: int = 10_000
    scenario_preset: str = "baseline"
    amount_median: Optional[float] = None  # None = the preset's default order-value distribution
    amount_sigma: Optional[float] = None
    max_retries: Optional[int] = 5
    max_recovery_window_minutes: Optional[float] = 4320.0
    forbid_terminal_retries: bool = True
    require_idempotency: bool = True
    require_buyer_approval: bool = True
    weight_revenue: float = 1.0
    weight_cost: float = 0.2
    weight_latency: float = 0.1
    weight_unnecessary: float = 0.2
    provenance: Mapping[str, str] = field(default_factory=dict)  # field -> 'parsed' | 'default' | 'override'
    warnings: tuple[str, ...] = ()

    def constraints_payload(self) -> dict[str, Any]:
        return {"max_retries": self.max_retries, "max_recovery_window_minutes": self.max_recovery_window_minutes,
                "forbid_terminal_retries": self.forbid_terminal_retries, "require_idempotency": self.require_idempotency,
                "require_buyer_approval": self.require_buyer_approval}

    def objective_payload(self) -> dict[str, Any]:
        return {"revenue_weight": self.weight_revenue, "cost_weight": self.weight_cost,
                "latency_weight": self.weight_latency, "unnecessary_attempts_weight": self.weight_unnecessary}

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "n_transactions": self.n_transactions, "scenario_preset": self.scenario_preset,
                "amount_median": self.amount_median, "amount_sigma": self.amount_sigma, "constraints": self.constraints_payload(),
                "weights": self.objective_payload(), "provenance": dict(self.provenance), "warnings": list(self.warnings)}


@dataclass
class ToolCallRecord:
    seq: int
    iteration: int  # 0 = harness setup, >=1 LLM turn, -1 = final evaluation phase
    source: str  # 'gemini' | 'harness'
    tool: str
    args: dict[str, Any]
    ok: bool
    result: Any  # full MCP result (or error text)
    duration_ms: float
    work_units: int = 0

    def to_dict(self, include_result: bool = True) -> dict[str, Any]:
        d = {"seq": self.seq, "iteration": self.iteration, "source": self.source, "tool": self.tool, "args": self.args,
             "ok": self.ok, "duration_ms": round(self.duration_ms, 1), "work_units": self.work_units}
        if include_result:
            d["result"] = self.result
        return d


@dataclass
class Observation:
    seq: int  # tool call that produced it
    scenario_id: str
    n_trials: int
    master_seed: int
    score_mean: float
    feasible: bool
    violations: list[str]
    recovery_rate: float
    recovery_rate_ci95: list[float]
    recovered_revenue: float
    avg_recovery_time: float
    unnecessary_retries: float
    cost: float


@dataclass
class CandidateRecord:
    name: str
    definition: dict[str, Any]
    origin: str  # 'baseline' | 'gemini' | 'variation'
    observations: list[Observation] = field(default_factory=list)
    revisions: int = 0  # times this NAME was re-used with a different definition

    def best_feasible_score(self) -> Optional[float]:
        s = [o.score_mean for o in self.observations if o.feasible]
        return max(s) if s else None

    def ever_feasible(self) -> bool:
        return any(o.feasible for o in self.observations)
