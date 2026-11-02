"""Deterministic decision rules applied to Monte Carlo output.

No LLM and no learned model is involved: this is fixed threshold logic
over the Monte Carlo result. All outputs are derived from SYNTHETIC /
MOCKED simulation data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from engine.model import SimConfig, SYNTHETIC_LABEL
from engine.montecarlo import MCResult

LOW_RISK_MAX = 0.05
HIGH_RISK_MIN = 0.25

MIN_TRIALS_FOR_CONFIDENCE = 100
CI_HALFWIDTH_TOLERANCE = 0.10


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"


class Decision(str, Enum):
    PROCEED = "PROCEED"
    ADJUST = "ADJUST"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Verdict:
    """Risk, confidence and expected impact are kept as independent fields
    on purpose -- none of them should be collapsed into a single score."""

    risk_score: float  # P(material shortfall), from Monte Carlo frequency
    confidence: float  # in [0, 1], based on MC statistical precision
    expected_impact_fraction: float  # E[fraction of batch missing window]
    expected_impact_count: float  # E[number of payments missing window]
    risk_level: RiskLevel
    decision: Decision
    reasons: tuple[str, ...] = field(default_factory=tuple)
    synthetic: bool = field(default=True)
    label: str = field(default=SYNTHETIC_LABEL)


def _risk_level(risk_score: float) -> RiskLevel:
    if risk_score < LOW_RISK_MAX:
        return RiskLevel.LOW
    if risk_score <= HIGH_RISK_MIN:
        return RiskLevel.MEDIUM
    return RiskLevel.HIGH


def evaluate(mc_result: MCResult, config: SimConfig) -> Verdict:
    """Turn a Monte Carlo result into a risk verdict via deterministic rules."""
    reasons: list[str] = []

    config_problems = config.validate()
    if config_problems:
        reasons.extend(config_problems)

    if mc_result.n_trials < MIN_TRIALS_FOR_CONFIDENCE:
        reasons.append(
            f"n_trials ({mc_result.n_trials}) below minimum "
            f"({MIN_TRIALS_FOR_CONFIDENCE}) for a confident estimate"
        )

    ci_low, ci_high = mc_result.p_shortfall_ci
    ci_halfwidth = (ci_high - ci_low) / 2.0
    if ci_halfwidth > CI_HALFWIDTH_TOLERANCE:
        reasons.append(
            f"Monte Carlo estimate too noisy: CI half-width "
            f"({ci_halfwidth:.3f}) exceeds tolerance ({CI_HALFWIDTH_TOLERANCE})"
        )

    confidence = max(0.0, min(1.0, 1.0 - (ci_halfwidth / CI_HALFWIDTH_TOLERANCE)))

    expected_impact_fraction = mc_result.mean_missed_fraction
    expected_impact_count = expected_impact_fraction * config.batch_size

    if reasons:
        return Verdict(
            risk_score=mc_result.p_shortfall,
            confidence=confidence,
            expected_impact_fraction=expected_impact_fraction,
            expected_impact_count=expected_impact_count,
            risk_level=RiskLevel.UNKNOWN,
            decision=Decision.ABSTAIN,
            reasons=tuple(reasons),
        )

    risk_level = _risk_level(mc_result.p_shortfall)
    decision = Decision.PROCEED if risk_level == RiskLevel.LOW else Decision.ADJUST

    return Verdict(
        risk_score=mc_result.p_shortfall,
        confidence=confidence,
        expected_impact_fraction=expected_impact_fraction,
        expected_impact_count=expected_impact_count,
        risk_level=risk_level,
        decision=decision,
        reasons=tuple(reasons),
    )
