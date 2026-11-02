"""Deterministic mock agent that proposes payment-batch release schedules,
simulates them through the Phase 1 engine, and decides whether to approve,
adjust, or abstain.

All data is SYNTHETIC / MOCKED. This agent never executes anything: it only
ever produces a recommendation (AgentDecision.approved_for_execution). No
risk score or simulation result is hardcoded here -- every number in the
execution trace comes from a genuine call into engine.montecarlo/engine.rules.

Time convention: this module uses its own epoch, t=0 at 16:00, independent
of engine.model's internal time convention (the engine itself has no
opinion about epochs -- it only takes relative minutes). This keeps the
agent's wall-clock scenario self-contained without requiring any change to
engine/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.model import SimConfig, SYNTHETIC_LABEL
from engine.montecarlo import DEFAULT_MATERIAL_SHORTFALL_THRESHOLD, MCResult, run_monte_carlo
from engine.rules import Decision, Verdict, evaluate


def minutes_from_epoch(hour: int, minute: int) -> float:
    """Convert a wall-clock HH:MM to minutes since this agent's epoch (16:00)."""
    return (hour - 16) * 60 + minute


# --- Primary scenario, matching the brief's illustrative parameters -------
BATCH_SIZE = 2000
N_WORKERS = 8
INITIAL_QUEUE_DEPTH = 350
FAILURE_PROB = 0.005
RETRY_DELAY = 5.0
LEDGER_CUTOFF = minutes_from_epoch(17, 30)
RECONCILIATION_CLOSE = minutes_from_epoch(19, 30)
INITIAL_RELEASE_TIME = minutes_from_epoch(18, 30)
ADJUSTED_RELEASE_TIMES = (minutes_from_epoch(16, 0), minutes_from_epoch(17, 15))

# posting_time_median was never specified in the brief and was already
# flagged in Phase 1 as our own modelling assumption. 0.15 (the Phase 1
# default) makes an 18:30-release/19:30-close window comfortably safe
# under genuine simulation, which would never trigger the ADJUST path this
# phase is meant to exercise. 0.25 is the smallest round value, found by a
# direct sweep against the real engine, at which the single 18:30 release
# is genuinely risky while the 16:00/17:15 split is genuinely safe -- i.e.
# this tunes a legitimate input assumption, not the simulated outcome.
POSTING_TIME_MEDIAN = 0.25
POSTING_TIME_SIGMA = 0.4


@dataclass(frozen=True)
class AgentConfig:
    """Agent-level policy, independent of engine.rules' own fixed
    thresholds. These gate whether the agent will *recommend* execution,
    on top of whatever engine.rules.evaluate already decided."""

    risk_threshold: float = 0.05
    min_confidence: float = 0.8
    n_trials: int = 200
    master_seed: int = 0
    shortfall_threshold: float = DEFAULT_MATERIAL_SHORTFALL_THRESHOLD


@dataclass(frozen=True)
class AgentAction:
    """A proposal the agent is considering, before it has been simulated."""

    label: str
    release_schedule: tuple[tuple[float, int], ...]
    rationale: str


@dataclass(frozen=True)
class AgentDecision:
    """What the agent decided after seeing one simulation's verdict.

    `approved_for_execution` is the only thing that matters for whether
    execution should happen -- and this agent never acts on it itself; a
    separate executor (outside Phase 2's scope) would. False always means
    the run stops here: either the simulation abstained, the proposal (or
    its one permitted revision) is still too risky, or it failed this
    agent's own risk/confidence gate despite engine.rules recommending
    PROCEED -- in every such case, Phase 2 requests human intervention
    rather than guessing further.
    """

    decision: Decision
    approved_for_execution: bool
    rationale: str


@dataclass(frozen=True)
class SimulationStep:
    """One proposal -> simulate -> decide cycle, kept for the execution trace."""

    step_index: int
    action: AgentAction
    mc_result: MCResult
    verdict: Verdict
    agent_decision: AgentDecision


@dataclass(frozen=True)
class AgentRunResult:
    """Full outcome of one agent run: every step taken, plus the final call."""

    trace: tuple[SimulationStep, ...]
    final_decision: AgentDecision
    synthetic: bool = field(default=True)
    label: str = field(default=SYNTHETIC_LABEL)


def build_initial_proposal(batch_size: int = BATCH_SIZE) -> AgentAction:
    return AgentAction(
        label="initial_full_batch_release",
        release_schedule=((INITIAL_RELEASE_TIME, batch_size),),
        rationale="Release the full batch at 18:30 as originally planned.",
    )


def build_adjusted_proposal(batch_size: int = BATCH_SIZE) -> AgentAction:
    first_time, second_time = ADJUSTED_RELEASE_TIMES
    first_count = batch_size // 2
    second_count = batch_size - first_count
    return AgentAction(
        label="adjusted_split_release",
        release_schedule=((first_time, first_count), (second_time, second_count)),
        rationale=(
            "Split the batch into two earlier releases (16:00 and 17:15) "
            "to give it more time to clear before the reconciliation close."
        ),
    )


def _build_sim_config(action: AgentAction, batch_size: int) -> SimConfig:
    return SimConfig(
        batch_size=batch_size,
        release_schedule=list(action.release_schedule),
        n_workers=N_WORKERS,
        reconciliation_close=RECONCILIATION_CLOSE,
        ledger_cutoff=LEDGER_CUTOFF,
        posting_time_median=POSTING_TIME_MEDIAN,
        posting_time_sigma=POSTING_TIME_SIGMA,
        failure_prob=FAILURE_PROB,
        retry_delay=RETRY_DELAY,
        initial_queue_depth=INITIAL_QUEUE_DEPTH,
    )


def _simulate(action: AgentAction, batch_size: int, agent_config: AgentConfig, seed: int) -> tuple[SimConfig, MCResult, Verdict]:
    config = _build_sim_config(action, batch_size)
    mc_result = run_monte_carlo(
        config,
        n_trials=agent_config.n_trials,
        master_seed=seed,
        shortfall_threshold=agent_config.shortfall_threshold,
    )
    verdict = evaluate(mc_result, config)
    return config, mc_result, verdict


def _decide(verdict: Verdict, agent_config: AgentConfig) -> AgentDecision:
    if verdict.decision == Decision.ABSTAIN:
        return AgentDecision(
            decision=Decision.ABSTAIN,
            approved_for_execution=False,
            rationale="Simulation abstained: " + "; ".join(verdict.reasons),
        )

    clears_agent_gate = (
        verdict.risk_score < agent_config.risk_threshold
        and verdict.confidence >= agent_config.min_confidence
    )

    if verdict.decision == Decision.PROCEED and clears_agent_gate:
        return AgentDecision(
            decision=Decision.PROCEED,
            approved_for_execution=True,
            rationale=(
                f"risk_score={verdict.risk_score:.4f} is below the agent's "
                f"execution threshold ({agent_config.risk_threshold}) and "
                f"confidence={verdict.confidence:.4f} meets the minimum "
                f"({agent_config.min_confidence})."
            ),
        )

    if verdict.decision == Decision.PROCEED and not clears_agent_gate:
        return AgentDecision(
            decision=Decision.PROCEED,
            approved_for_execution=False,
            rationale=(
                "Simulation rules recommended PROCEED but the proposal did "
                "not clear this agent's own risk/confidence thresholds; "
                "requesting human intervention."
            ),
        )

    return AgentDecision(
        decision=Decision.ADJUST,
        approved_for_execution=False,
        rationale=f"risk_level={verdict.risk_level}: proposal needs adjustment.",
    )


def run_agent(
    agent_config: AgentConfig | None = None, batch_size: int = BATCH_SIZE
) -> AgentRunResult:
    """Run the full propose -> simulate -> (adjust) -> decide flow once.

    Reproducible: the entire run is a deterministic function of
    (agent_config.master_seed, batch_size) -- the second simulation (if any)
    uses master_seed + 1, itself a deterministic derivation, so the whole
    trace is identical across repeated calls with the same inputs.
    """
    agent_config = agent_config or AgentConfig()
    trace: list[SimulationStep] = []

    initial_action = build_initial_proposal(batch_size)
    _, mc_result_1, verdict_1 = _simulate(
        initial_action, batch_size, agent_config, agent_config.master_seed
    )
    decision_1 = _decide(verdict_1, agent_config)
    trace.append(SimulationStep(0, initial_action, mc_result_1, verdict_1, decision_1))

    if decision_1.decision != Decision.ADJUST:
        return AgentRunResult(trace=tuple(trace), final_decision=decision_1)

    adjusted_action = build_adjusted_proposal(batch_size)
    _, mc_result_2, verdict_2 = _simulate(
        adjusted_action, batch_size, agent_config, agent_config.master_seed + 1
    )
    decision_2 = _decide(verdict_2, agent_config)
    trace.append(SimulationStep(1, adjusted_action, mc_result_2, verdict_2, decision_2))

    return AgentRunResult(trace=tuple(trace), final_decision=decision_2)
