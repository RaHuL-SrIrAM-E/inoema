"""Merchant form -> agent objective. The natural-language text is for Gemini's planning; the ENFORCED constraints come from
explicit structured overrides (and are echoed back through the agent's own deterministic parser for preview). No Streamlit imports."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent.payments.planner import parse_objective

NUMBER_WORDS = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


@dataclass
class MerchantForm:
    n_transactions: int = 10_000
    scenario_preset: str = "baseline"
    amount_median: Optional[float] = None
    amount_sigma: Optional[float] = None
    max_retries: int = 2
    recovery_window_hours: float = 72.0
    forbid_terminal_retries: bool = True
    require_idempotency: bool = True
    require_buyer_approval: bool = True
    latency_weight: float = 0.1
    cost_weight: float = 0.2
    unnecessary_weight: float = 0.2
    extra_notes: str = ""


def compose_objective_text(f: MerchantForm) -> str:
    n = NUMBER_WORDS.get(f.max_retries, str(f.max_retries))
    parts = [f"Optimize payment recovery for a merchant processing {f.n_transactions:,} transactions.",
             "Maximize recovered revenue while minimizing recovery cost and latency.",
             f"Do not exceed {n} automatic retries."]
    if f.forbid_terminal_retries:
        parts.append("Never automatically retry terminal failures.")
    if f.extra_notes.strip():
        parts.append(f.extra_notes.strip())
    return " ".join(parts)


def to_overrides(f: MerchantForm) -> dict[str, Any]:
    """Structured, authoritative values (these override anything the text parser infers)."""
    o: dict[str, Any] = {"n_transactions": int(f.n_transactions), "scenario_preset": f.scenario_preset, "max_retries": int(f.max_retries),
                         "max_recovery_window_minutes": float(f.recovery_window_hours) * 60.0, "forbid_terminal_retries": bool(f.forbid_terminal_retries),
                         "require_idempotency": bool(f.require_idempotency), "require_buyer_approval": bool(f.require_buyer_approval),
                         "weight_latency": float(f.latency_weight), "weight_cost": float(f.cost_weight), "weight_unnecessary": float(f.unnecessary_weight)}
    if f.amount_median is not None:
        o["amount_median"] = float(f.amount_median)
    if f.amount_sigma is not None:
        o["amount_sigma"] = float(f.amount_sigma)
    return o


def preview(f: MerchantForm):
    """The ObjectiveSpec the agent will enforce, built by the agent's own parser (so the UI cannot drift from the backend)."""
    return parse_objective(compose_objective_text(f), **to_overrides(f))


def validate(f: MerchantForm) -> list[str]:
    p = []
    if not 100 <= f.n_transactions <= 200_000:
        p.append("Transactions per trial must be between 100 and 200,000.")
    if not 0 <= f.max_retries <= 20:
        p.append("Maximum retries must be between 0 and 20.")
    if f.recovery_window_hours <= 0:
        p.append("Recovery window must be positive.")
    if min(f.latency_weight, f.cost_weight, f.unnecessary_weight) < 0:
        p.append("Weights cannot be negative.")
    if f.amount_median is not None and not 0 < f.amount_median <= 100_000:
        p.append("Median order value must be in (0, 100,000].")
    if f.amount_sigma is not None and not 0 < f.amount_sigma <= 3:
        p.append("Order-value sigma must be in (0, 3].")
    return p
