"""Per-trial metrics, cross-trial aggregation and confidence intervals.

DENOMINATORS (explicit, to avoid ambiguity):
  recovery_rate    = recovered / initially_failed
  completion_rate  = (successful_initial + recovered) / total_transactions
  avg_recovery_time = mean over RECOVERED transactions of (resolution - first failure), minutes
  avg_retries_per_transaction = automatic retries / total_transactions
  avg_retries_per_failed      = automatic retries / initially_failed
  cost_per_successful_transaction = total_processing_cost / (successful_initial + recovered)
  unnecessary_retry_fraction = unnecessary_retries / automatic retries
Rates are 0.0 when their denominator is 0 (flagged by the denominator in pooled output).

CONFIDENCE INTERVALS:
  * recovery_rate and completion_rate (pooled across trials): Wilson 95% interval on pooled
    counts. Assumes transactions are independent Bernoulli trials given the assumed model.
  * every other metric: mean over trials with a normal-approximation 95% interval
    (mean +/- 1.96 * sd / sqrt(n_trials)); None when n_trials < 2. Reasonable for >= ~10 trials.
  These intervals quantify SIMULATION sampling error only -- not uncertainty about the assumed
  model parameters, which is usually far larger.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Optional, Sequence

import numpy as np

from engine.montecarlo import wilson_interval
from engine.payments.models import FinalStatus, PaymentStatus, RecoveryStatus, Transaction

Z95 = 1.959963984540054


@dataclass(frozen=True)
class TrialMetrics:
    total_transactions: int
    successful_initial: int
    initially_failed: int
    recovered: int
    final_failed: int
    total_revenue: float
    initially_failed_revenue: float
    recovered_revenue: float
    recovery_rate: float
    completion_rate: float
    recovery_time_sum: float  # minutes, over recovered txns
    avg_recovery_time: float
    total_retries: int
    avg_retries_per_transaction: float
    avg_retries_per_failed: float
    unnecessary_retries: int
    terminal_retries: int
    buyer_class_retries: int
    unnecessary_retry_fraction: float
    total_attempts: int
    total_processing_cost: float
    recovery_cost: float
    cost_per_successful_transaction: float
    buyer_actions_requested: int
    buyer_actions_completed: int
    escalated_for_review: int
    duplicate_capture_risk_count: int
    max_retry_count: int
    max_recovery_time: float

    def to_dict(self) -> dict:
        return asdict(self)


def _div(a: float, b: float) -> float:
    return a / b if b else 0.0


def compute_trial_metrics(txns: Sequence[Transaction]) -> TrialMetrics:
    n = len(txns)
    failed = [t for t in txns if t.initial_status == PaymentStatus.FAILED]
    recovered = [t for t in failed if t.recovery_status == RecoveryStatus.RECOVERED]
    ok_initial = n - len(failed)
    completed = ok_initial + len(recovered)
    total_retries = sum(t.retry_count for t in txns)
    unnecessary = sum(t.unnecessary_retries for t in txns)
    cost = float(sum(t.processing_cost for t in txns))
    rec_time = float(sum(t.total_recovery_time or 0.0 for t in recovered))
    assert all(t.final_status == FinalStatus.COMPLETED for t in recovered)
    return TrialMetrics(
        total_transactions=n, successful_initial=ok_initial, initially_failed=len(failed),
        recovered=len(recovered), final_failed=len(failed) - len(recovered),
        total_revenue=float(sum(t.amount for t in txns)),
        initially_failed_revenue=float(sum(t.amount for t in failed)),
        recovered_revenue=float(sum(t.amount for t in recovered)),
        recovery_rate=_div(len(recovered), len(failed)), completion_rate=_div(completed, n),
        recovery_time_sum=rec_time, avg_recovery_time=_div(rec_time, len(recovered)),
        total_retries=total_retries, avg_retries_per_transaction=_div(total_retries, n),
        avg_retries_per_failed=_div(total_retries, len(failed)),
        unnecessary_retries=unnecessary, terminal_retries=sum(t.terminal_retries for t in txns),
        buyer_class_retries=sum(t.buyer_class_retries for t in txns),
        unnecessary_retry_fraction=_div(unnecessary, total_retries),
        total_attempts=sum(t.attempt_count for t in txns), total_processing_cost=cost,
        recovery_cost=float(sum(t.recovery_cost for t in txns)),
        cost_per_successful_transaction=_div(cost, completed),
        buyer_actions_requested=sum(t.buyer_actions_requested for t in txns),
        buyer_actions_completed=sum(t.buyer_actions_completed for t in txns),
        escalated_for_review=sum(1 for t in txns if t.recovery_status == RecoveryStatus.ESCALATED_FOR_REVIEW),
        duplicate_capture_risk_count=sum(1 for t in txns if t.duplicate_capture_risk),
        max_retry_count=max((t.retry_count for t in txns), default=0),
        max_recovery_time=max((t.total_recovery_time or 0.0 for t in txns), default=0.0),
    )


def mean_ci(values: Sequence[float]) -> dict:
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(a.mean())
    if n < 2:
        return {"mean": mean, "std": 0.0, "ci95": None, "n": n}
    sd = float(a.std(ddof=1))
    half = Z95 * sd / math.sqrt(n)
    return {"mean": mean, "std": sd, "ci95": [mean - half, mean + half], "n": n}


@dataclass(frozen=True)
class AggregateMetrics:
    n_trials: int
    means: dict  # metric name -> {mean, std, ci95, n}
    pooled: dict  # pooled-count rates with Wilson CIs and explicit denominators

    def to_dict(self) -> dict:
        return {"n_trials": self.n_trials, "means": self.means, "pooled": self.pooled}


def _pooled_rate(num: int, den: int, definition: str) -> dict:
    lo, hi = wilson_interval(num, den)
    return {"value": _div(num, den), "ci95_wilson": [lo, hi], "numerator": num, "denominator": den,
            "definition": definition}


def aggregate_trials(trials: Sequence[TrialMetrics]) -> AggregateMetrics:
    if not trials:
        raise ValueError("no trials to aggregate")
    means = {f.name: mean_ci([getattr(t, f.name) for t in trials])
             for f in fields(TrialMetrics) if f.name not in ("max_retry_count",)}
    means["max_retry_count"] = {"mean": float(np.mean([t.max_retry_count for t in trials])),
                                "max": max(t.max_retry_count for t in trials)}
    s = lambda k: sum(getattr(t, k) for t in trials)  # noqa: E731
    completed = s("successful_initial") + s("recovered")
    pooled = {
        "recovery_rate": _pooled_rate(s("recovered"), s("initially_failed"), "recovered / initially_failed"),
        "completion_rate": _pooled_rate(completed, s("total_transactions"),
                                        "(successful_initial + recovered) / total_transactions"),
        "initial_success_rate": _pooled_rate(s("successful_initial"), s("total_transactions"),
                                             "successful_initial / total_transactions"),
        "avg_recovery_time_minutes": {"value": _div(s("recovery_time_sum"), s("recovered")),
                                      "denominator": s("recovered"),
                                      "definition": "sum recovery time of recovered txns / recovered"},
        "cost_per_successful_transaction": {"value": _div(s("total_processing_cost"), completed),
                                            "denominator": completed,
                                            "definition": "total_processing_cost / (successful_initial + recovered)"},
        "unnecessary_retry_fraction": {"value": _div(s("unnecessary_retries"), s("total_retries")),
                                       "denominator": s("total_retries"),
                                       "definition": "unnecessary_retries / automatic retries"},
    }
    return AggregateMetrics(len(trials), means, pooled)
