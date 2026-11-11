"""Seeded workload generation. A trial seed (SeedSequence) is split into two
independent child streams: one for the workload (amounts, methods, arrival
times) and one for the per-transaction latent "world" draws."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np

from engine.payments.distributions import AmountDistribution, categorical_from_uniform
from engine.payments.failure_model import CATEGORIES, METHODS, FailureModel, LatentOutcomes, draw_latent
from engine.payments.models import PaymentMethodCategory as PM, PaymentStatus, Transaction

MAX_TRANSACTIONS = 200_000


@dataclass(frozen=True)
class WorkloadConfig:
    n_transactions: int = 10_000
    merchant_id: str = "merchant-sim-001"
    currency: str = "USD"
    amount: AmountDistribution = field(default_factory=AmountDistribution)
    method_mix: dict = field(default_factory=lambda: {
        PM.CARD: 0.55, PM.WALLET_BALANCE: 0.25, PM.BANK_ACCOUNT: 0.12, PM.PAY_LATER: 0.08})
    horizon_minutes: float = 1440.0  # arrivals uniform over this window (= Poisson process conditioned on N)
    start_time: datetime = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def validate(self) -> list[str]:
        p = list(self.amount.validate())
        if not 1 <= self.n_transactions <= MAX_TRANSACTIONS:
            p.append(f"n_transactions must be in [1, {MAX_TRANSACTIONS}]")
        if self.horizon_minutes <= 0:
            p.append("horizon_minutes must be > 0")
        if len(self.currency) != 3:
            p.append("currency must be a 3-letter code")
        if abs(sum(self.method_mix.get(m, 0.0) for m in METHODS) - 1.0) > 1e-6 or any(v < 0 for v in self.method_mix.values()):
            p.append("method_mix must be non-negative and sum to 1")
        return p


@dataclass(frozen=True)
class Workload:
    config: WorkloadConfig
    amounts: np.ndarray
    method_idx: np.ndarray
    offsets: np.ndarray  # minutes since scenario start, sorted ascending


def generate_workload(config: WorkloadConfig, seed: np.random.SeedSequence) -> Workload:
    problems = config.validate()
    if problems:
        raise ValueError("Invalid WorkloadConfig: " + "; ".join(problems))
    rng = np.random.default_rng(seed)
    n = config.n_transactions
    offsets = np.sort(rng.uniform(0.0, config.horizon_minutes, n))
    probs = np.array([config.method_mix.get(m, 0.0) for m in METHODS])
    method_idx = categorical_from_uniform(rng.random(n), probs)
    amounts = config.amount.sample(rng, n)
    return Workload(config, amounts, method_idx, offsets)


def prepare_trial(config: WorkloadConfig, model: FailureModel, trial_seed: np.random.SeedSequence
                  ) -> tuple[Workload, LatentOutcomes]:
    """Deterministically build the workload and latent world for one trial."""
    problems = model.validate()
    if problems:
        raise ValueError("Invalid FailureModel: " + "; ".join(problems))
    workload_ss, world_ss = trial_seed.spawn(2)
    workload = generate_workload(config, workload_ss)
    latent = draw_latent(np.random.default_rng(world_ss), workload.method_idx, model)
    return workload, latent


def instantiate_transactions(workload: Workload, latent: LatentOutcomes) -> list[Transaction]:
    """Fresh mutable Transaction objects (initial status/category filled in from the latent world)."""
    cfg = workload.config
    out: list[Transaction] = []
    for i in range(cfg.n_transactions):
        failed = bool(latent.initial_failed[i])
        off = float(workload.offsets[i])
        out.append(Transaction(
            transaction_id=f"txn-{i:06d}", merchant_id=cfg.merchant_id, amount=float(workload.amounts[i]),
            currency=cfg.currency, timestamp=cfg.start_time + timedelta(minutes=off),
            payment_method_category=METHODS[int(workload.method_idx[i])], offset_minutes=off,
            failure_category=CATEGORIES[int(latent.category_idx[i])] if failed else None,
            initial_status=PaymentStatus.FAILED if failed else PaymentStatus.APPROVED,
        ))
    return out
