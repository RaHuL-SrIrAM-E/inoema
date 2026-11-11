"""Deterministic objective evaluation for payment-recovery policies.

Two strictly separate stages, and NO LLM involvement in either:

1. HARD CONSTRAINTS (feasibility): a policy that violates any is INFEASIBLE
   regardless of score. Checked statically on the policy configuration and
   again against observed simulation counts.
2. WEIGHTED SCORE (ranking among feasible policies), per trial:

     score = w_rev*R - w_cost*C - w_lat*L - w_unn*U
       R = recovered_revenue / initially_failed_revenue                    in [0,1]
       C = min(1, recovery_cost per initially-failed txn / cost_reference)  in [0,1]
       L = min(1, avg_recovery_time / latency_reference_minutes)            in [0,1]
       U = unnecessary_retries / automatic_retries (0 if no retries)        in [0,1]

   Scores are computed per trial; policies are compared on trial-mean scores,
   and differences are reported as PAIRED differences (all policies see the
   same workload and latent draws in each trial).
The reference values and weights are merchant preferences, not facts.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from engine.payments.failure_model import FailureRules, RecoveryClass
from engine.payments.metrics import Z95, TrialMetrics, mean_ci
from engine.payments.models import FailureCategory
from engine.payments.recovery_policy import RecoveryPolicy, is_retry_eligible
from engine.payments.simulator import PaymentMCResult


@dataclass(frozen=True)
class ObjectiveWeights:
    revenue: float = 1.0
    cost: float = 0.2
    latency: float = 0.1
    unnecessary_attempts: float = 0.2
    cost_reference: float = 1.0  # recovery cost per initially-failed txn at which C saturates at 1
    latency_reference_minutes: float = 1440.0  # avg recovery time at which L saturates at 1

    def validate(self) -> list[str]:
        p = [f"{n} must be >= 0" for n in ("revenue", "cost", "latency", "unnecessary_attempts")
             if getattr(self, n) < 0]
        if self.revenue <= 0:
            p.append("revenue weight must be > 0")
        if self.cost_reference <= 0 or self.latency_reference_minutes <= 0:
            p.append("reference values must be > 0")
        return p


@dataclass(frozen=True)
class HardConstraints:
    max_retries: int | None = 5
    max_recovery_window_minutes: float | None = 4320.0
    forbid_terminal_retries: bool = True  # no automatic retries on TERMINAL-class failures
    require_idempotency: bool = True  # retries must reuse an idempotency key
    require_buyer_approval: bool = True  # never auto-retry failures that need the buyer to act


@dataclass(frozen=True)
class ScoreBreakdown:
    revenue_term: float
    cost_term: float
    latency_term: float
    unnecessary_term: float
    score: float


def score_trial(m: TrialMetrics, w: ObjectiveWeights) -> ScoreBreakdown:
    r = m.recovered_revenue / m.initially_failed_revenue if m.initially_failed_revenue else 0.0
    c = min(1.0, (m.recovery_cost / m.initially_failed) / w.cost_reference) if m.initially_failed else 0.0
    l = min(1.0, m.avg_recovery_time / w.latency_reference_minutes)
    u = m.unnecessary_retry_fraction
    return ScoreBreakdown(r, c, l, u, w.revenue * r - w.cost * c - w.latency * l - w.unnecessary_attempts * u)


def static_constraint_violations(policy: RecoveryPolicy, rules: FailureRules, hc: HardConstraints) -> list[str]:
    v: list[str] = []
    if hc.max_retries is not None and policy.retry_limit > hc.max_retries:
        v.append(f"retry_limit {policy.retry_limit} exceeds max_retries {hc.max_retries}")
    if hc.max_recovery_window_minutes is not None and policy.maximum_recovery_window > hc.max_recovery_window_minutes:
        v.append(f"maximum_recovery_window {policy.maximum_recovery_window} exceeds {hc.max_recovery_window_minutes}")
    if hc.require_idempotency and policy.retry_limit > 0 and not policy.idempotent_retries:
        v.append("retries are not idempotent")
    for cat in FailureCategory:
        if not is_retry_eligible(policy, rules, cat):
            continue
        cls = rules.class_of(cat)
        if hc.forbid_terminal_retries and cls == RecoveryClass.TERMINAL:
            v.append(f"policy automatically retries terminal failure {cat.value}")
        if hc.require_buyer_approval and cls == RecoveryClass.BUYER_ACTION:
            v.append(f"policy automatically retries {cat.value} without required buyer action")
    return v


def observed_constraint_violations(trials: Sequence[TrialMetrics], hc: HardConstraints) -> list[str]:
    v: list[str] = []
    if hc.max_retries is not None and max(t.max_retry_count for t in trials) > hc.max_retries:
        v.append("observed a transaction with more retries than max_retries")
    if hc.max_recovery_window_minutes is not None and max(t.max_recovery_time for t in trials) > hc.max_recovery_window_minutes:
        v.append("observed a recovery longer than max_recovery_window_minutes")
    if hc.forbid_terminal_retries and sum(t.terminal_retries for t in trials) > 0:
        v.append(f"observed {sum(t.terminal_retries for t in trials)} automatic retries of terminal failures")
    if hc.require_buyer_approval and sum(t.buyer_class_retries for t in trials) > 0:
        v.append(f"observed {sum(t.buyer_class_retries for t in trials)} automatic retries of buyer-action failures")
    return v


@dataclass(frozen=True)
class PolicyEvaluation:
    policy_name: str
    feasible: bool
    violations: tuple[str, ...]
    score_mean: float
    score_ci95: tuple[float, float] | None
    components_mean: dict
    trial_scores: tuple[float, ...] = field(repr=False, default=())

    def to_dict(self) -> dict:
        return {"policy": self.policy_name, "feasible": self.feasible, "violations": list(self.violations),
                "score_mean": self.score_mean, "score_ci95": list(self.score_ci95) if self.score_ci95 else None,
                "components_mean": self.components_mean}


def evaluate_policy(result: PaymentMCResult, rules: FailureRules, w: ObjectiveWeights,
                    hc: HardConstraints) -> PolicyEvaluation:
    problems = w.validate()
    if problems:
        raise ValueError("Invalid ObjectiveWeights: " + "; ".join(problems))
    parts = [score_trial(t, w) for t in result.trial_metrics]
    scores = [p.score for p in parts]
    ci = mean_ci(scores)["ci95"]
    violations = static_constraint_violations(result.policy, rules, hc) + \
        observed_constraint_violations(result.trial_metrics, hc)
    comps = {k: float(np.mean([getattr(p, k) for p in parts]))
             for k in ("revenue_term", "cost_term", "latency_term", "unnecessary_term")}
    return PolicyEvaluation(result.policy.name, not violations, tuple(violations), float(np.mean(scores)),
                            tuple(ci) if ci else None, comps, tuple(scores))


def rank_policies(evals: Sequence[PolicyEvaluation]) -> list[PolicyEvaluation]:
    """Feasible policies first (by score desc), then infeasible ones (by score desc). Deterministic tie-break on name."""
    return sorted(evals, key=lambda e: (not e.feasible, -e.score_mean, e.policy_name))


def paired_difference(a: Sequence[float], b: Sequence[float]) -> dict:
    """Mean of (a - b) over trials with a normal-approx 95% CI (valid because trials are paired by seed)."""
    if len(a) != len(b):
        raise ValueError("paired samples must have equal length")
    d = np.asarray(a) - np.asarray(b)
    r = mean_ci(d)
    significant = bool(r["ci95"] and (r["ci95"][0] > 0 or r["ci95"][1] < 0))
    return {"mean_difference": r["mean"], "ci95": r["ci95"], "ci_excludes_zero": significant}
