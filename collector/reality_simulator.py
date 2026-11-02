"""Independent reality simulator.

Represents "what actually happens" to a released payment batch, as a
stand-in for the real operational environment. Deliberately does NOT
import engine.model or engine.montecarlo, does not take a predicted
MCResult/Verdict as input, and does not reuse the predictor's RNG stream
or distribution parameters -- it has its own parameters (RealityConfig,
most of which the prediction engine never sees) and its own seed
(`reality_seed`, unrelated to the predictor's master_seed).

Hidden factors modelled here, none of which the predictor knows about:
  - a random day-to-day slowdown multiplier ("today is a slow day")
  - independent per-worker outages (a worker goes down for a span of time)

All output is SYNTHETIC / MOCKED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import simpy

MOCKED_LABEL = "SYNTHETIC_MOCKED_REALITY"


@dataclass(frozen=True)
class RealityConfig:
    """Hidden-to-the-predictor operational parameters."""

    base_posting_time_median: float = 0.22
    base_posting_time_sigma: float = 0.5
    base_failure_prob: float = 0.01
    retry_delay: float = 5.0
    max_retries: int = 5
    slowdown_prob: float = 0.3
    slowdown_multiplier_range: tuple[float, float] = (1.2, 2.0)
    outage_prob_per_worker: float = 0.15
    outage_duration_range: tuple[float, float] = (10.0, 30.0)
    # Fraction-of-window range an outage's start time is drawn from, e.g.
    # (0.0, 1.0) (the default) lets an outage start anywhere across the
    # whole simulated window.
    outage_start_fraction_range: tuple[float, float] = (0.0, 1.0)


@dataclass(frozen=True)
class RealityOutcome:
    """What actually happened, as independently simulated. Synthetic data only."""

    batch_size: int
    completion_times: dict[int, Optional[float]]
    n_completed_by_close: int
    missed_fraction: float
    reality_seed: int
    day_slowdown_multiplier: float
    worker_outage_minutes: tuple[float, ...]  # per-worker outage duration, 0.0 if none
    synthetic: bool = field(default=True)
    label: str = field(default=MOCKED_LABEL)


def _lognormal_duration(rng: np.random.Generator, median: float, sigma: float) -> float:
    return float(rng.lognormal(mean=np.log(median), sigma=sigma))


def _payment_process(env, resource, payment_id, config, effective_median, rng, completion_times):
    attempts_made = 0
    while True:
        with resource.request() as req:
            yield req
            duration = _lognormal_duration(rng, effective_median, config.base_posting_time_sigma)
            yield env.timeout(duration)

        attempts_made += 1
        if rng.random() >= config.base_failure_prob:
            break
        if attempts_made >= config.max_retries + 1:
            return
        yield env.timeout(config.retry_delay)

    completion_times[payment_id] = env.now


def _backlog_process(env, resource, config, effective_median, rng):
    attempts_made = 0
    while True:
        with resource.request() as req:
            yield req
            duration = _lognormal_duration(rng, effective_median, config.base_posting_time_sigma)
            yield env.timeout(duration)

        attempts_made += 1
        if rng.random() >= config.base_failure_prob:
            return
        if attempts_made >= config.max_retries + 1:
            return
        yield env.timeout(config.retry_delay)


def _outage_process(env, resource, start_time, duration):
    wait = start_time - env.now
    if wait > 0:
        yield env.timeout(wait)
    with resource.request() as req:
        yield req
        yield env.timeout(duration)


def _release_process(env, resource, release_schedule, config, effective_median, rng, completion_times):
    payment_id = 0
    for release_time, count in release_schedule:
        wait = release_time - env.now
        if wait > 0:
            yield env.timeout(wait)
        for _ in range(count):
            completion_times[payment_id] = None
            env.process(
                _payment_process(env, resource, payment_id, config, effective_median, rng, completion_times)
            )
            payment_id += 1


def simulate_reality(
    batch_size: int,
    release_schedule: list[tuple[float, int]] | tuple[tuple[float, int], ...],
    n_workers: int,
    reconciliation_close: float,
    initial_queue_depth: int,
    reality_seed: int,
    reality_config: Optional[RealityConfig] = None,
) -> RealityOutcome:
    """Run one independent discrete-event trial of "what actually
    happened" for an already-approved release schedule. Only the public
    operational facts (batch_size, release_schedule, n_workers,
    reconciliation_close, initial_queue_depth) are shared with the
    predictor; everything else here is hidden from it."""
    reality_config = reality_config or RealityConfig()
    rng = np.random.default_rng(reality_seed)

    env = simpy.Environment()
    resource = simpy.Resource(env, capacity=n_workers)
    completion_times: dict[int, Optional[float]] = {}

    day_slowdown_multiplier = 1.0
    if rng.random() < reality_config.slowdown_prob:
        day_slowdown_multiplier = float(rng.uniform(*reality_config.slowdown_multiplier_range))
    effective_median = reality_config.base_posting_time_median * day_slowdown_multiplier

    outage_minutes: list[float] = []
    for _ in range(n_workers):
        if rng.random() < reality_config.outage_prob_per_worker:
            start_fraction = float(rng.uniform(*reality_config.outage_start_fraction_range))
            start_time = start_fraction * reconciliation_close
            duration = float(rng.uniform(*reality_config.outage_duration_range))
            outage_minutes.append(duration)
            env.process(_outage_process(env, resource, start_time, duration))
        else:
            outage_minutes.append(0.0)

    for _ in range(initial_queue_depth):
        env.process(_backlog_process(env, resource, reality_config, effective_median, rng))

    env.process(
        _release_process(env, resource, release_schedule, reality_config, effective_median, rng, completion_times)
    )

    env.run()

    n_completed_by_close = sum(
        1 for t in completion_times.values() if t is not None and t <= reconciliation_close
    )
    missed_fraction = (batch_size - n_completed_by_close) / batch_size

    return RealityOutcome(
        batch_size=batch_size,
        completion_times=completion_times,
        n_completed_by_close=n_completed_by_close,
        missed_fraction=missed_fraction,
        reality_seed=reality_seed,
        day_slowdown_multiplier=day_slowdown_multiplier,
        worker_outage_minutes=tuple(outage_minutes),
    )
