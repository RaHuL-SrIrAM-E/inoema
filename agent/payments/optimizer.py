"""Candidate bookkeeping and deterministic candidate-variation generation.

The registry records every candidate policy the agent has seen (from tool arguments/results) together
with the numeric observations the evaluator returned. propose_variations() is a deterministic helper the LLM
may use for local search; the LLM remains free to ignore it and propose anything.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

from agent.payments.models import CandidateRecord, Observation

POLICY_FIELDS = ("retry_limit", "retry_intervals", "exponential_backoff", "backoff_base_minutes", "backoff_multiplier",
                 "backoff_max_interval_minutes", "maximum_recovery_window", "buyer_action_timeout", "request_buyer_action",
                 "alternate_funding_action", "terminal_failure_handling", "classification_aware", "idempotent_retries")
VARIATION_DIMENSIONS = ("retry_limit", "backoff_base_minutes", "backoff_multiplier", "buyer_action_timeout",
                        "maximum_recovery_window", "alternate_funding_action", "terminal_failure_handling")


class CandidateRegistry:
    def __init__(self) -> None:
        self.candidates: dict[str, CandidateRecord] = {}

    def register(self, name: str, definition: dict[str, Any], origin: str) -> CandidateRecord:
        cur = self.candidates.get(name)
        if cur is None:
            cur = self.candidates[name] = CandidateRecord(name, copy.deepcopy(definition), origin)
        elif cur.definition != definition:
            # Same name, different policy: the old observations describe a different policy, so discard them
            # rather than attribute them to the new definition.
            cur.definition, cur.observations, cur.revisions = copy.deepcopy(definition), [], cur.revisions + 1
        return cur

    def observe(self, name: str, obs: Observation) -> None:
        self.candidates[name].observations.append(obs)

    def top_feasible(self, k: int) -> list[str]:
        """Names of candidates ever observed feasible, best observed score first (ties broken by name)."""
        scored = [(c.best_feasible_score(), c.name) for c in self.candidates.values() if c.ever_feasible()]
        return [n for _, n in sorted(scored, key=lambda t: (-t[0], t[1]))[:k]]

    def definition(self, name: str) -> dict[str, Any]:
        return copy.deepcopy(self.candidates[name].definition)

    def to_dict(self) -> list[dict[str, Any]]:
        return [{"name": c.name, "origin": c.origin, "definition": c.definition, "revisions": c.revisions,
                 "observations": [o.__dict__ for o in c.observations]} for c in self.candidates.values()]


def _obs(seq: int, scenario_id: str, n_trials: int, seed: int, row: dict[str, Any], summary: dict[str, Any]) -> Observation:
    pm = summary["per_trial_means"]
    return Observation(seq=seq, scenario_id=scenario_id, n_trials=n_trials, master_seed=seed, score_mean=row["score_mean"],
                       feasible=row["feasible"], violations=list(row["violations"]),
                       recovery_rate=summary["recovery_rate"]["value"], recovery_rate_ci95=list(summary["recovery_rate"]["ci95_wilson"]),
                       recovered_revenue=pm["recovered_revenue"], avg_recovery_time=summary["avg_recovery_time_minutes"]["value"],
                       unnecessary_retries=pm["unnecessary_retries"], cost=pm["total_processing_cost"])


def ingest_tool_result(registry: CandidateRegistry, tool: str, args: dict[str, Any], data: Any, seq: int,
                       origin: str = "gemini") -> list[str]:
    """Register candidates and record observations from a successful run/compare result. Returns names touched."""
    if not isinstance(data, dict):
        return []
    touched: list[str] = []
    if tool == "compare_recovery_policies":
        rows = {r["policy"]: r for r in data["ranking"]}
        for name, pol in data["policies"].items():
            registry.register(name, pol["definition"], origin if name not in _BASELINE_NAMES else "baseline")
            registry.observe(name, _obs(seq, data["scenario_id"], data["n_trials"], data["master_seed"], rows[name], pol["summary"]))
            touched.append(name)
    elif tool == "run_payment_simulation":
        d = data["policy_definition"]
        registry.register(d["name"], d, origin if d["name"] not in _BASELINE_NAMES else "baseline")
        registry.observe(d["name"], _obs(seq, data["scenario_id"], data["summary"]["n_trials"], data["master_seed"],
                                         {"score_mean": data["evaluation"]["score_mean"], "feasible": data["evaluation"]["feasible"],
                                          "violations": data["evaluation"]["violations"]}, data["summary"]))
        touched.append(d["name"])
    return touched


_BASELINE_NAMES = {"immediate_retry", "exponential_backoff", "intelligent_recovery", "no_recovery"}


def propose_variations(definition: dict[str, Any], dimensions: Optional[list[str]] = None, max_variants: int = 8) -> list[dict[str, Any]]:
    """Deterministic neighbours of a full policy definition (changes one dimension at a time).

    Returns PolicySpec-shaped dicts, each with a unique `name` and a `change` description. Whether a variant
    is valid/feasible is decided by the MCP server and evaluator, not here.
    """
    dims = [d for d in (dimensions or VARIATION_DIMENSIONS) if d in VARIATION_DIMENSIONS]
    base_name = definition["name"]
    out: list[dict[str, Any]] = []

    def add(suffix: str, change: str, **mods: Any) -> None:
        v = {k: copy.deepcopy(definition[k]) for k in POLICY_FIELDS if k in definition}
        v.update(mods)
        out.append({**v, "name": f"{base_name}~{suffix}", "change": change})

    for d in dims:
        if d == "retry_limit":
            r = definition["retry_limit"]
            if r > 0:
                add(f"retries{r - 1}", f"retry_limit {r} -> {r - 1}", retry_limit=r - 1)
            if r < 20:
                add(f"retries{r + 1}", f"retry_limit {r} -> {r + 1}", retry_limit=r + 1)
        elif d == "backoff_base_minutes" and definition.get("exponential_backoff"):
            b = definition["backoff_base_minutes"]
            add(f"base{b / 2:g}", f"backoff_base_minutes {b:g} -> {b / 2:g}", backoff_base_minutes=b / 2,
                backoff_max_interval_minutes=max(definition["backoff_max_interval_minutes"], b / 2))
            add(f"base{b * 2:g}", f"backoff_base_minutes {b:g} -> {b * 2:g}", backoff_base_minutes=b * 2,
                backoff_max_interval_minutes=max(definition["backoff_max_interval_minutes"], b * 2))
        elif d == "backoff_multiplier" and definition.get("exponential_backoff"):
            m = definition["backoff_multiplier"]
            add(f"mult{m * 1.5:g}", f"backoff_multiplier {m:g} -> {m * 1.5:g}", backoff_multiplier=m * 1.5)
            if m > 1.0:
                add(f"mult{max(1.0, m / 1.5):g}", f"backoff_multiplier {m:g} -> {max(1.0, m / 1.5):g}", backoff_multiplier=max(1.0, m / 1.5))
        elif d == "buyer_action_timeout" and definition.get("request_buyer_action"):
            t = definition["buyer_action_timeout"]
            add(f"buyer{t / 2:g}", f"buyer_action_timeout {t:g} -> {t / 2:g}", buyer_action_timeout=t / 2)
            add(f"buyer{t * 2:g}", f"buyer_action_timeout {t:g} -> {t * 2:g}", buyer_action_timeout=t * 2)
        elif d == "maximum_recovery_window":
            w = definition["maximum_recovery_window"]
            add(f"window{w / 2:g}", f"maximum_recovery_window {w:g} -> {w / 2:g}", maximum_recovery_window=w / 2)
            add(f"window{w * 2:g}", f"maximum_recovery_window {w:g} -> {w * 2:g}", maximum_recovery_window=w * 2)
        elif d == "alternate_funding_action" and definition.get("request_buyer_action"):
            new = "NONE" if definition["alternate_funding_action"] == "REQUEST_ALTERNATE" else "REQUEST_ALTERNATE"
            add(f"altfund_{new.lower()}", f"alternate_funding_action -> {new}", alternate_funding_action=new)
        elif d == "terminal_failure_handling":
            new = "ABANDON" if definition["terminal_failure_handling"] == "FLAG_FOR_REVIEW" else "FLAG_FOR_REVIEW"
            add(f"terminal_{new.lower()}", f"terminal_failure_handling -> {new}", terminal_failure_handling=new)
    return out[:max_variants]
