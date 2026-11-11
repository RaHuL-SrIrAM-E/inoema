"""Sampling helpers. All randomness comes from caller-supplied numpy
Generators / pre-drawn uniforms so results are seed-reproducible."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class AmountKind(str, Enum):
    LOGNORMAL = "LOGNORMAL"
    UNIFORM = "UNIFORM"
    FIXED = "FIXED"


@dataclass(frozen=True)
class AmountDistribution:
    """Transaction amount distribution (ASSUMED shape, not empirical)."""

    kind: AmountKind = AmountKind.LOGNORMAL
    median: float = 45.0  # LOGNORMAL: median of the underlying lognormal
    sigma: float = 0.9  # LOGNORMAL: sigma of the underlying normal
    low: float = 1.0  # clip floor (LOGNORMAL) / lower bound (UNIFORM)
    high: float = 2000.0  # clip cap (LOGNORMAL) / upper bound (UNIFORM)
    fixed_value: float = 50.0  # FIXED

    def validate(self) -> list[str]:
        p: list[str] = []
        if self.low <= 0:
            p.append("amount low must be > 0")
        if self.high < self.low:
            p.append("amount high must be >= low")
        if self.kind == AmountKind.LOGNORMAL and (self.median <= 0 or self.sigma <= 0):
            p.append("lognormal median and sigma must be > 0")
        if self.kind == AmountKind.FIXED and self.fixed_value <= 0:
            p.append("fixed_value must be > 0")
        return p

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.kind == AmountKind.LOGNORMAL:
            x = np.clip(rng.lognormal(np.log(self.median), self.sigma, n), self.low, self.high)
        elif self.kind == AmountKind.UNIFORM:
            x = rng.uniform(self.low, self.high, n)
        else:
            x = np.full(n, self.fixed_value)
        return np.maximum(np.round(x, 2), 0.01)


def categorical_from_uniform(u: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Map uniforms in [0,1) to category indices given probabilities (must sum to ~1)."""
    cum = np.cumsum(probs)
    cum[-1] = 1.0
    return np.searchsorted(cum, u, side="right").clip(max=len(probs) - 1)


def exponential_from_uniform(u: np.ndarray, mean: float | np.ndarray) -> np.ndarray:
    """Inverse-CDF exponential sample with the given mean; mean 0 -> 0."""
    return -np.asarray(mean, dtype=float) * np.log1p(-u)
