"""Objective parsing (deterministic) and prompt construction.

parse_objective() extracts hard constraints from the merchant's text with explicit, auditable rules. Anything
constraint-like that it does NOT recognise is reported as a warning and is NOT enforced -- the LLM never
silently becomes the source of constraints. Callers may pass explicit overrides.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Optional

from agent.payments.models import AgentConfig, ObjectiveSpec

_NUM_WORDS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten".split())}
_TRIGGER = re.compile(r"\b(never|must not|do not|don't|no more than|at most|not exceed|without|only if)\b", re.I)


def _to_int(tok: str) -> Optional[int]:
    tok = tok.lower().replace(",", "")
    return int(tok) if tok.isdigit() else _NUM_WORDS.get(tok)


def parse_objective(text: str, **overrides: Any) -> ObjectiveSpec:
    spec = ObjectiveSpec(text=text)
    prov = {k: "default" for k in ("n_transactions", "max_retries", "max_recovery_window_minutes", "forbid_terminal_retries",
                                   "require_idempotency", "require_buyer_approval", "weights")}
    upd: dict[str, Any] = {}
    recognised: list[str] = []

    m = re.search(r"([\d,]+)\s+(?:payment\s+|hypothetical\s+)?transactions", text, re.I)
    if m and _to_int(m.group(1)):
        upd["n_transactions"], prov["n_transactions"] = _to_int(m.group(1)), "parsed"
        recognised.append(m.group(0))

    m = re.search(r"(?:not\s+exceed|no\s+more\s+than|at\s+most|maximum\s+of|max(?:imum)?)\s+(\w+)\s+(?:automatic\s+)?(?:retry|retries)", text, re.I)
    if m and _to_int(m.group(1)) is not None:
        upd["max_retries"], prov["max_retries"] = _to_int(m.group(1)), "parsed"
        recognised.append(m.group(0))

    m = re.search(r"(?:never|not|no)\s+(?:automatically\s+)?(?:retry|retries|retrying)\s+(?:for\s+|of\s+|on\s+)?terminal|no\s+automatic\s+retries\s+(?:for|of|on)\s+terminal", text, re.I)
    if m:
        upd["forbid_terminal_retries"], prov["forbid_terminal_retries"] = True, "parsed"
        recognised.append(m.group(0))

    m = re.search(r"(?:within|no\s+longer\s+than|recovery\s+window\s+of)\s+(\d+(?:\.\d+)?)\s*(minutes?|hours?|days?)", text, re.I)
    if m:
        mult = {"m": 1, "h": 60, "d": 1440}[m.group(2)[0].lower()]
        upd["max_recovery_window_minutes"], prov["max_recovery_window_minutes"] = float(m.group(1)) * mult, "parsed"
        recognised.append(m.group(0))

    if re.search(r"idempoten", text, re.I):
        prov["require_idempotency"] = "parsed"
        recognised.append("idempotency")

    warnings = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if _TRIGGER.search(sent) and not any(r.lower() in sent.lower() for r in recognised):
            warnings.append(f"Constraint-like phrase not recognised and NOT enforced: {sent.strip()!r}")

    for k, v in overrides.items():
        if not hasattr(spec, k) or k in ("text", "provenance", "warnings"):
            raise ValueError(f"unknown ObjectiveSpec field {k!r}")
        upd[k] = v
        prov["weights" if k.startswith("weight_") else k] = "override"
    return replace(spec, provenance=prov, warnings=tuple(warnings), **upd)


SYSTEM_PROMPT = """You are an autonomous payment-recovery optimisation agent for a merchant. You work ONLY through the \
function tools provided; never invent numbers. All results are from a SIMULATION with ASSUMED distributions \
(not PayPal data); say so when you interpret them.

Goal: find the recovery policy that best satisfies the merchant's objective subject to hard constraints.
How you work:
- Policies are specified as JSON (PolicySpec). Start from a baseline (`base`: immediate_retry, exponential_backoff, \
intelligent_recovery, no_recovery) and override fields, or specify a full custom policy. Give every candidate a unique `name`.
- Key fields: retry_limit, retry_intervals / exponential_backoff (+backoff_base_minutes, backoff_multiplier, \
backoff_max_interval_minutes), maximum_recovery_window (minutes), buyer_action_timeout (minutes), request_buyer_action, \
alternate_funding_action (NONE|REQUEST_ALTERNATE), terminal_failure_handling (ABANDON|FLAG_FOR_REVIEW), \
classification_aware (True: use failure classes; False: blind retry of everything), idempotent_retries.
- Failure classes: transient errors are auto-retryable; declined/expired/payer-action/not-approved need the buyer; \
validation/terminal failures cannot be recovered automatically.
- Use compare_recovery_policies to evaluate several candidates on identical simulated workloads. The tool result states \
each candidate's feasibility under the merchant's HARD CONSTRAINTS and a deterministic score. Do not compute scores yourself \
and do not treat an infeasible candidate as acceptable: read its violations and adapt.
- Adapt: let each result decide your next experiment (e.g. fix violated constraints, tune the most promising feasible candidate, \
try variations). propose_policy_variations suggests neighbouring candidates. Use run_sensitivity_analysis when you want to know \
whether a choice depends on the assumptions. Avoid repeating identical experiments.
- You have a limited number of turns, tool calls and simulation budget (reported in every tool response under `_harness`). 
- When done, call finish_optimization with a shortlist of candidate names and your rationale. The FINAL selection is made by a \
deterministic evaluator re-running your shortlist under fixed settings; your shortlist is advisory.
"""


def build_initial_message(spec: ObjectiveSpec, cfg: AgentConfig, experiment_id: str, scenario_id: str,
                          initial: dict[str, Any], tools: list[str]) -> str:
    c = spec.constraints_payload()
    lines = [
        "MERCHANT OBJECTIVE (verbatim):", spec.text, "",
        "HARD CONSTRAINTS ENFORCED BY THE EVALUATOR (authoritative; derived deterministically from the objective text and defaults):",
        *(f"- {k} = {v}  [{spec.provenance.get(k, 'default')}]" for k, v in c.items()),
        f"Objective weights (defaults unless stated): revenue {spec.weight_revenue}, cost {spec.weight_cost}, latency {spec.weight_latency}, "
        f"unnecessary attempts {spec.weight_unnecessary}.",
    ]
    lines += [f"WARNING: {w}" for w in spec.warnings]
    lines += ["", f"experiment_id = {experiment_id}", f"baseline scenario_id = {scenario_id} ({spec.n_transactions} transactions per trial)",
              f"Initial (current) policy `{cfg.initial_policy}` evaluated by the harness: {initial}", "",
              f"Budget: {cfg.max_iterations} turns, {cfg.max_tool_calls} tool calls, {cfg.exploration_budget_units} simulation work units "
              f"(transactions x trials x policies). Suggested trials per experiment: {cfg.exploration_trials}.",
              "Available tools: " + ", ".join(tools), "",
              "Begin: state briefly how you interpret the objective and plan your first experiment, then call tools."]
    return "\n".join(lines)
