"""Named, deterministic scenario presets (workload + assumed world + rules + costs)."""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from engine.payments.distributions import AmountDistribution, AmountKind
from engine.payments.failure_model import FailureModel, FailureRules
from engine.payments.models import CostModel, FailureCategory as FC, PaymentMethodCategory as PM
from engine.payments.transaction_generator import WorkloadConfig


@dataclass(frozen=True)
class PaymentScenario:
    name: str
    workload: WorkloadConfig
    failure_model: FailureModel = field(default_factory=FailureModel)
    rules: FailureRules = field(default_factory=FailureRules)
    costs: CostModel = field(default_factory=CostModel)
    description: str = ""

    def validate(self) -> list[str]:
        return (self.workload.validate() + self.failure_model.validate()
                + self.rules.validate() + self.costs.validate())


PRESETS = {
    "baseline": "Mixed methods, default assumed failure mix.",
    "network_instability": "Transient network/processing errors 3x more common (e.g. an unstable integration or provider incident).",
    "buyer_friction": "Declined, expired and unapproved-order failures dominate (consumer-card-heavy checkout).",
    "high_value_orders": "Larger average order value (lognormal median 220); same failure mix.",
    "high_failure_rate": "Initial failure rate doubled across all methods.",
}


def build_scenario(preset: str = "baseline", n_transactions: int = 10_000, name: str | None = None,
                   costs: CostModel | None = None) -> PaymentScenario:
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose from {sorted(PRESETS)}")
    amount = AmountDistribution()
    model = FailureModel()
    if preset == "network_instability":
        model = model.scaled(category_weight_multipliers={FC.TRANSIENT_NETWORK_ERROR: 3.0,
                                                          FC.TEMPORARY_PROCESSING_ERROR: 3.0})
    elif preset == "buyer_friction":
        model = model.scaled(category_weight_multipliers={FC.INSTRUMENT_DECLINED: 1.6, FC.EXPIRED_PAYMENT_METHOD: 2.0,
                                                          FC.ORDER_NOT_APPROVED: 1.6})
    elif preset == "high_value_orders":
        amount = AmountDistribution(kind=AmountKind.LOGNORMAL, median=220.0, sigma=0.8, low=5.0, high=10000.0)
    elif preset == "high_failure_rate":
        model = model.scaled(initial_failure_multiplier=2.0)
    workload = replace(WorkloadConfig(), n_transactions=n_transactions, amount=amount)
    scenario = PaymentScenario(name=name or preset, workload=workload, failure_model=model,
                               costs=costs or CostModel(), description=PRESETS[preset])
    problems = scenario.validate()
    if problems:
        raise ValueError("Invalid scenario: " + "; ".join(problems))
    return scenario
