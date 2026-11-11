"""Spike 2: minimal FastMCP server wrapping the EXISTING INOEMA engine.

Run standalone (stdio):  python -m spikes.mcp_sim.server
The tool calls engine.montecarlo.run_monte_carlo / engine.rules.evaluate
for real; nothing here is hard-coded. Results are SYNTHETIC.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastmcp import FastMCP
from pydantic import BaseModel, Field, model_validator

from engine.model import SimConfig
from engine.montecarlo import run_monte_carlo
from engine.rules import evaluate
from mcp_server.payment_tools import register_payment_tools

mcp = FastMCP("inoema-sim-spike")
register_payment_tools(mcp)  # Phase 2: payment-recovery simulation tools


class ReleaseStep(BaseModel):
    time_minutes: float = Field(ge=0, description="Release time in minutes on the scenario clock")
    count: int = Field(gt=0, description="Number of payments released at that time")


class SimulateInput(BaseModel):
    release_schedule: list[ReleaseStep] = Field(min_length=1)
    n_workers: int = Field(gt=0, le=1000)
    reconciliation_close: float = Field(gt=0, description="Deadline in minutes on the scenario clock")
    initial_queue_depth: int = Field(default=0, ge=0)
    posting_time_median: float = Field(default=0.25, gt=0)
    failure_prob: float = Field(default=0.005, ge=0, le=1)
    n_trials: int = Field(default=100, ge=10, le=500)
    master_seed: int = Field(default=0, ge=0)


class SimulateOutput(BaseModel):
    batch_size: int
    n_trials: int
    master_seed: int
    p_shortfall: float
    p_shortfall_ci95: tuple[float, float]
    mean_missed_fraction: float
    expected_missed_count: float
    risk_level: str
    engine_decision: str
    confidence: float
    reasons: list[str]
    synthetic: bool = True


@mcp.tool
def simulate_release_schedule(params: SimulateInput) -> SimulateOutput:
    """Run a seeded Monte Carlo of the INOEMA discrete-event engine for a payment
    batch release schedule and return risk of missing the reconciliation deadline.
    Deterministic for the same inputs and master_seed. Synthetic data only."""
    schedule = [(s.time_minutes, s.count) for s in params.release_schedule]
    cfg = SimConfig(
        batch_size=sum(c for _, c in schedule),
        release_schedule=schedule,
        n_workers=params.n_workers,
        reconciliation_close=params.reconciliation_close,
        initial_queue_depth=params.initial_queue_depth,
        posting_time_median=params.posting_time_median,
        failure_prob=params.failure_prob,
    )
    problems = cfg.validate()
    if problems:
        raise ValueError("; ".join(problems))
    mc = run_monte_carlo(cfg, params.n_trials, params.master_seed)
    v = evaluate(mc, cfg)
    return SimulateOutput(
        batch_size=cfg.batch_size,
        n_trials=mc.n_trials,
        master_seed=mc.master_seed,
        p_shortfall=mc.p_shortfall,
        p_shortfall_ci95=mc.p_shortfall_ci,
        mean_missed_fraction=mc.mean_missed_fraction,
        expected_missed_count=v.expected_impact_count,
        risk_level=v.risk_level.value,
        engine_decision=v.decision.value,
        confidence=v.confidence,
        reasons=list(v.reasons),
    )


if __name__ == "__main__":
    mcp.run(show_banner=False)  # stdio transport
