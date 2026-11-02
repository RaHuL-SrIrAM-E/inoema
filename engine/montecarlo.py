"""Monte Carlo aggregation over many discrete-event trials.

All results are SYNTHETIC / MOCKED. The Monte Carlo estimate of
shortfall probability (risk_score) is a plain frequency over simulated
trials -- no model or LLM judgement is involved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from engine.model import SimConfig, TrialResult, run_once, SYNTHETIC_LABEL

DEFAULT_MATERIAL_SHORTFALL_THRESHOLD = 0.05
_Z_95 = 1.959963984540054  # two-sided 95% normal quantile


def wilson_interval(successes: int, n: int, z: float = _Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Returns (low, high)."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    halfwidth = (z * np.sqrt((p_hat * (1 - p_hat) / n) + (z**2 / (4 * n**2)))) / denom
    return (max(0.0, center - halfwidth), min(1.0, center + halfwidth))


@dataclass(frozen=True)
class MCResult:
    """Aggregate Monte Carlo output for one scenario. Synthetic data only."""

    n_trials: int
    master_seed: int
    shortfall_threshold: float
    p_shortfall: float  # fraction of trials with missed_fraction > shortfall_threshold
    p_shortfall_ci: tuple[float, float]  # Wilson 95% CI on p_shortfall
    mean_missed_fraction: float
    std_missed_fraction: float
    missed_fractions: tuple[float, ...]  # per-trial, for diagnostics/tests
    synthetic: bool = field(default=True)
    label: str = field(default=SYNTHETIC_LABEL)


def run_monte_carlo(
    config: SimConfig,
    n_trials: int,
    master_seed: int,
    shortfall_threshold: float = DEFAULT_MATERIAL_SHORTFALL_THRESHOLD,
) -> MCResult:
    """Run n_trials independent discrete-event trials of `config`.

    Trial seeds are deterministically derived from master_seed via
    numpy.random.SeedSequence.spawn, so (config, n_trials, master_seed)
    fully determines the result -- this is what makes runs reproducible.
    """
    if n_trials <= 0:
        raise ValueError("n_trials must be > 0")

    # Validated once here rather than once per trial inside run_once's own
    # (still-present, defensive) check -- fails fast before spawning seeds
    # or running any trials.
    config_problems = config.validate()
    if config_problems:
        raise ValueError(f"Invalid SimConfig: {'; '.join(config_problems)}")

    seed_sequence = np.random.SeedSequence(master_seed)
    trial_seeds = seed_sequence.spawn(n_trials)

    missed_fractions: list[float] = []
    for trial_seed in trial_seeds:
        # Pass the spawned SeedSequence straight through to run_once (which
        # hands it to np.random.default_rng) instead of collapsing it to a
        # 32-bit int -- preserves spawn()'s collision-resistance guarantees
        # across trials.
        result: TrialResult = run_once(config, trial_seed)
        missed_fractions.append(result.missed_fraction)

    missed_arr = np.asarray(missed_fractions, dtype=float)
    n_shortfall = int(np.sum(missed_arr > shortfall_threshold))
    p_shortfall = n_shortfall / n_trials
    ci = wilson_interval(n_shortfall, n_trials)

    return MCResult(
        n_trials=n_trials,
        master_seed=master_seed,
        shortfall_threshold=shortfall_threshold,
        p_shortfall=p_shortfall,
        p_shortfall_ci=ci,
        mean_missed_fraction=float(np.mean(missed_arr)),
        std_missed_fraction=float(np.std(missed_arr)),
        missed_fractions=tuple(missed_fractions),
    )
