"""Discrete-event simulation of a payments-reconciliation batch.

All data produced here is SYNTHETIC / MOCKED: no real PayPal, bank, or
ledger system is contacted. This module only models the queueing and
timing dynamics of a payments batch moving through a fixed pool of
workers before a hard reconciliation cutoff.

Time convention: minutes on a single axis with t=0 pinned at 17:00.
e.g. ledger cutoff 17:30 -> 30.0, reconciliation close 19:30 -> 150.0.

Note on ledger_cutoff: it is recorded on SimConfig as context for callers
(and for later phases) but is NOT currently enforced or read by the
simulation -- only reconciliation_close gates payment timing. See the
SimConfig.ledger_cutoff docstring below.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Union

import numpy as np
import simpy

SYNTHETIC_LABEL = "SYNTHETIC_MOCKED_SIMULATION"


@dataclass(frozen=True)
class SimConfig:
    """Parameters for a single simulated scenario.

    release_schedule: list of (release_time_minutes, count) pairs whose
    counts must sum to batch_size. A single-release proposal is just
    [(release_time, batch_size)]; a split proposal is multiple tuples.
    """

    batch_size: int
    release_schedule: list[tuple[float, int]]
    n_workers: int
    reconciliation_close: float
    # Informational only in Phase 1: NOT enforced by validate() or read by
    # the simulation. Nothing currently checks release_schedule times
    # against it. Carried on the config for context/later phases.
    ledger_cutoff: float = 0.0
    posting_time_median: float = 0.15  # minutes; lognormal median
    posting_time_sigma: float = 0.4  # lognormal shape param (sigma of underlying normal)
    failure_prob: float = 0.005
    retry_delay: float = 5.0  # minutes
    initial_queue_depth: int = 0
    max_retries: int = 5

    def validate(self) -> list[str]:
        """Return a list of validation problems (empty if config is sound)."""
        problems: list[str] = []
        if self.batch_size <= 0:
            problems.append("batch_size must be > 0")
        if self.n_workers <= 0:
            problems.append("n_workers must be > 0")
        if self.posting_time_median <= 0:
            problems.append("posting_time_median must be > 0")
        if self.posting_time_sigma <= 0:
            problems.append("posting_time_sigma must be > 0")
        if not (0.0 <= self.failure_prob <= 1.0):
            problems.append("failure_prob must be in [0, 1]")
        if self.retry_delay < 0:
            problems.append("retry_delay must be >= 0")
        if self.initial_queue_depth < 0:
            problems.append("initial_queue_depth must be >= 0")
        if self.max_retries < 0:
            problems.append("max_retries must be >= 0")
        if self.reconciliation_close <= 0:
            problems.append("reconciliation_close must be > 0")
        scheduled = sum(count for _, count in self.release_schedule)
        if scheduled != self.batch_size:
            problems.append(
                f"release_schedule counts ({scheduled}) must sum to batch_size "
                f"({self.batch_size})"
            )
        if any(count <= 0 for _, count in self.release_schedule):
            problems.append("release_schedule counts must all be > 0")
        if any(t < 0 for t, _ in self.release_schedule):
            problems.append("release_schedule times must be >= 0")
        release_times = [t for t, _ in self.release_schedule]
        if release_times != sorted(release_times):
            problems.append("release_schedule entries must be in non-decreasing time order")
        return problems


@dataclass(frozen=True)
class TrialResult:
    """Outcome of one discrete-event trial. Synthetic data only."""

    batch_size: int
    completion_times: dict[int, Optional[float]]  # payment_id -> completion time, None if never completed
    n_completed_by_close: int
    missed_fraction: float
    seed: Union[int, np.random.SeedSequence]
    synthetic: bool = field(default=True)
    label: str = field(default=SYNTHETIC_LABEL)


def _lognormal_duration(rng: np.random.Generator, median: float, sigma: float) -> float:
    mu = np.log(median)
    return float(rng.lognormal(mean=mu, sigma=sigma))


def _payment_process(
    env: simpy.Environment,
    resource: simpy.Resource,
    payment_id: int,
    is_backlog: bool,
    config: SimConfig,
    rng: np.random.Generator,
    completion_times: dict[int, Optional[float]],
):
    attempts_made = 0
    while True:
        with resource.request() as req:
            yield req
            duration = _lognormal_duration(rng, config.posting_time_median, config.posting_time_sigma)
            yield env.timeout(duration)

        attempts_made += 1
        failed = rng.random() < config.failure_prob

        if not failed:
            break

        if attempts_made >= config.max_retries + 1:
            # Permanently failed: never completes, counts as missed.
            return

        yield env.timeout(config.retry_delay)

    if not is_backlog:
        completion_times[payment_id] = env.now


def _release_process(env: simpy.Environment, resource: simpy.Resource, config: SimConfig,
                      rng: np.random.Generator, completion_times: dict[int, Optional[float]]):
    payment_id = 0
    for release_time, count in config.release_schedule:
        wait = release_time - env.now
        if wait > 0:
            yield env.timeout(wait)
        for _ in range(count):
            completion_times[payment_id] = None
            env.process(
                _payment_process(env, resource, payment_id, False, config, rng, completion_times)
            )
            payment_id += 1


def count_completed_by_close(
    completion_times: dict[int, Optional[float]], reconciliation_close: float
) -> int:
    """Pure classification rule: a payment made the reconciliation window
    iff it completed (not None) at or before the close time (inclusive
    boundary). Exposed standalone so the cutoff rule itself is directly
    unit-testable without running a simulation."""
    return sum(
        1 for t in completion_times.values() if t is not None and t <= reconciliation_close
    )


def run_once(config: SimConfig, seed: Union[int, np.random.SeedSequence]) -> TrialResult:
    """Run one discrete-event trial with an independently-seeded RNG.

    `seed` may be a plain int or a numpy SeedSequence (e.g. one spawned by
    engine.montecarlo.run_monte_carlo) -- np.random.default_rng accepts
    both directly, so callers doing Monte Carlo trials should pass the
    spawned SeedSequence itself rather than collapsing it to an int, to
    preserve its collision-resistance guarantees across trials.
    """
    problems = config.validate()
    if problems:
        raise ValueError(f"Invalid SimConfig: {'; '.join(problems)}")

    rng = np.random.default_rng(seed)
    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=config.n_workers)
    completion_times: dict[int, Optional[float]] = {}

    for _ in range(config.initial_queue_depth):
        env.process(_payment_process(env, resource, -1, True, config, rng, completion_times))

    env.process(_release_process(env, resource, config, rng, completion_times))

    env.run()

    n_completed_by_close = count_completed_by_close(completion_times, config.reconciliation_close)
    missed_fraction = (config.batch_size - n_completed_by_close) / config.batch_size

    return TrialResult(
        batch_size=config.batch_size,
        completion_times=completion_times,
        n_completed_by_close=n_completed_by_close,
        missed_fraction=missed_fraction,
        seed=seed,
    )
