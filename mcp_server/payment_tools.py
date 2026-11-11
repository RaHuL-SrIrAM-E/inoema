"""MCP tools for the payment-recovery simulation. Every tool calls the real
engine.payments / engine.objectives code; nothing is canned. All results are
SIMULATED under ASSUMED distributions and carry explicit provenance labels.

Register on a FastMCP instance with register_payment_tools(mcp).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

from engine.objectives import (HardConstraints, ObjectiveWeights, evaluate_policy, paired_difference,
                               rank_policies)
from engine.payments.models import (ASSUMPTION_LABEL, MODEL_VERSION, SIM_LABEL, CostModel)
from engine.payments.recovery_policy import (BASELINES, AlternateFundingAction, TerminalFailureHandling,
                                             resolve_policy)
from engine.payments.scenarios import PRESETS, build_scenario
from engine.payments.sensitivity import PARAMETERS as SENS_PARAMS, estimate_work_units, run_sensitivity, sweep_plan
from engine.payments.simulator import (PaymentMCResult, compare_policies_monte_carlo, trace_transaction,
                                       trial_seeds)
from engine.payments.transaction_generator import MAX_TRANSACTIONS, prepare_trial
from storage import payment_experiments as store

MAX_TRIALS = 50
MAX_SENSITIVITY_WORK_UNITS = 15_000_000
MAX_WORK_UNITS = 6_000_000  # transactions x trials x policies per call (keeps calls to ~tens of seconds)
PROVENANCE = {"provenance": "SIMULATED", "label": SIM_LABEL, "assumptions": ASSUMPTION_LABEL,
              "model_version": MODEL_VERSION,
              "warning": "Simulated merchant-side recovery under assumed distributions. Not PayPal data."}

BaselineName = Literal["immediate_retry", "exponential_backoff", "intelligent_recovery", "no_recovery"]
PresetName = Literal["baseline", "network_instability", "buyer_friction", "high_value_orders", "high_failure_rate"]


class ObjectiveInput(BaseModel):
    revenue_weight: float = Field(default=1.0, gt=0)
    cost_weight: float = Field(default=0.2, ge=0)
    latency_weight: float = Field(default=0.1, ge=0)
    unnecessary_attempts_weight: float = Field(default=0.2, ge=0)
    cost_reference: float = Field(default=1.0, gt=0)
    latency_reference_minutes: float = Field(default=1440.0, gt=0)


class ConstraintInput(BaseModel):
    max_retries: Optional[int] = Field(default=5, ge=0, le=20)
    max_recovery_window_minutes: Optional[float] = Field(default=4320.0, gt=0)
    forbid_terminal_retries: bool = True
    require_idempotency: bool = True
    require_buyer_approval: bool = True


class CostInput(BaseModel):
    attempt_cost: float = Field(default=0.03, ge=0)
    buyer_notification_cost: float = Field(default=0.02, ge=0)
    manual_review_cost: float = Field(default=0.25, ge=0)


class ScenarioRequest(BaseModel):
    preset: PresetName = "baseline"
    n_transactions: int = Field(default=10_000, ge=100, le=MAX_TRANSACTIONS)
    name: Optional[str] = None


SweepName = Literal[tuple(SENS_PARAMS)]  # type: ignore[valid-type]


class SweepSpec(BaseModel):
    parameter: SweepName
    values: Optional[list[float]] = Field(default=None, min_length=1, max_length=8,
                                          description="Values to test; omit to use the default exploration grid.")


class PolicySpec(BaseModel):
    """A baseline (optionally with overrides) or a fully custom policy (base omitted)."""
    name: Optional[str] = None
    base: Optional[BaselineName] = None
    retry_limit: Optional[int] = Field(default=None, ge=0, le=20)
    retry_intervals: Optional[list[float]] = None
    exponential_backoff: Optional[bool] = None
    backoff_base_minutes: Optional[float] = Field(default=None, gt=0)
    backoff_multiplier: Optional[float] = Field(default=None, ge=1)
    backoff_max_interval_minutes: Optional[float] = Field(default=None, gt=0)
    maximum_recovery_window: Optional[float] = Field(default=None, gt=0)
    buyer_action_timeout: Optional[float] = Field(default=None, gt=0)
    request_buyer_action: Optional[bool] = None
    alternate_funding_action: Optional[AlternateFundingAction] = None
    terminal_failure_handling: Optional[TerminalFailureHandling] = None
    classification_aware: Optional[bool] = None
    idempotent_retries: Optional[bool] = None

    @model_validator(mode="after")
    def _need_name_or_base(self):
        if self.name is None and self.base is None:
            raise ValueError("policy needs a name or a base baseline")
        return self


def _policy(spec: PolicySpec):
    overrides = spec.model_dump(exclude={"base", "name"}, exclude_none=True)
    if "retry_intervals" in overrides:
        overrides["retry_intervals"] = tuple(overrides["retry_intervals"])
    p = resolve_policy(spec.base, **overrides)
    from dataclasses import replace
    return replace(p, name=spec.name or p.name)


def _policy_dict(p) -> dict:
    from dataclasses import asdict
    d = asdict(p)
    d["retry_intervals"] = list(d["retry_intervals"])
    d["alternate_funding_action"] = p.alternate_funding_action.value
    d["terminal_failure_handling"] = p.terminal_failure_handling.value
    return d


def _load(conn, experiment_id: str, scenario_id: str):
    exp = store.get_experiment(conn, experiment_id)
    if exp is None:
        raise ValueError(f"unknown experiment_id {experiment_id!r}")
    sc = store.get_scenario(conn, experiment_id, scenario_id)
    if sc is None:
        raise ValueError(f"unknown scenario_id {scenario_id!r} for this experiment")
    cfg = exp["config"]
    scenario = build_scenario(sc["spec"]["preset"], sc["spec"]["n_transactions"], sc["name"],
                              costs=CostModel(**cfg["costs"]))
    return exp, scenario, ObjectiveWeights(**cfg["objective"]), HardConstraints(**cfg["constraints"])


def _check_budget(n_tx: int, n_trials: int, n_policies: int, cap: Optional[int] = None) -> int:
    """Enforce the per-call work cap (and an optional caller-supplied tighter cap); returns the work units."""
    units = n_tx * n_trials * n_policies
    if units > MAX_WORK_UNITS:
        raise ValueError(f"workload too large ({n_tx} x {n_trials} trials x {n_policies} policies > {MAX_WORK_UNITS}); "
                         "reduce n_trials or n_transactions")
    if cap is not None and units > cap:
        raise ValueError(f"simulation budget exceeded: this call needs {units} work units but only {cap} remain")
    return units


def _summary(r: PaymentMCResult) -> dict:
    a, m, p = r.aggregate, r.aggregate.means, r.aggregate.pooled
    return {
        "policy": r.policy.name, "n_trials": r.n_trials,
        "per_trial_means": {k: m[k]["mean"] for k in (
            "total_transactions", "successful_initial", "initially_failed", "recovered", "final_failed",
            "recovered_revenue", "total_retries", "unnecessary_retries", "total_processing_cost",
            "buyer_actions_requested", "escalated_for_review", "duplicate_capture_risk_count")},
        "recovery_rate": p["recovery_rate"], "completion_rate": p["completion_rate"],
        "initial_success_rate": p["initial_success_rate"],
        "avg_recovery_time_minutes": p["avg_recovery_time_minutes"],
        "avg_retries_per_transaction": m["avg_retries_per_transaction"],
        "cost_per_successful_transaction": p["cost_per_successful_transaction"],
        "unnecessary_retry_fraction": p["unnecessary_retry_fraction"],
        "recovered_revenue_ci95": m["recovered_revenue"]["ci95"],
        "total_processing_cost_ci95": m["total_processing_cost"]["ci95"],
    }


def register_payment_tools(mcp) -> None:
    @mcp.tool
    def create_payment_experiment(name: str = Field(min_length=1, max_length=120),
                                  objective: ObjectiveInput = ObjectiveInput(),
                                  constraints: ConstraintInput = ConstraintInput(),
                                  costs: CostInput = CostInput()) -> dict[str, Any]:
        """Create a persisted payment-recovery experiment holding the merchant's objective weights,
        hard constraints and (assumed) operational costs. Returns experiment_id."""
        cfg = {"objective": {"revenue": objective.revenue_weight, "cost": objective.cost_weight,
                             "latency": objective.latency_weight,
                             "unnecessary_attempts": objective.unnecessary_attempts_weight,
                             "cost_reference": objective.cost_reference,
                             "latency_reference_minutes": objective.latency_reference_minutes},
               "constraints": constraints.model_dump(), "costs": costs.model_dump()}
        conn = store.connect()
        try:
            eid = store.create_experiment(conn, name, cfg)
        finally:
            conn.close()
        return {"experiment_id": eid, "name": name, "config": cfg, **PROVENANCE}

    @mcp.tool
    def generate_payment_scenarios(experiment_id: str, scenarios: list[ScenarioRequest] = Field(min_length=1, max_length=10)
                                   ) -> dict[str, Any]:
        """Generate workload scenarios (from presets) for an experiment. Presets: """ + "; ".join(
        f"{k}: {v}" for k, v in PRESETS.items()) + """. Returns scenario_ids to simulate against."""
        conn = store.connect()
        try:
            if store.get_experiment(conn, experiment_id) is None:
                raise ValueError(f"unknown experiment_id {experiment_id!r}")
            out = []
            for req in scenarios:
                sc = build_scenario(req.preset, req.n_transactions, req.name)
                sid = store.add_scenario(conn, experiment_id, sc.name,
                                         {"preset": req.preset, "n_transactions": req.n_transactions})
                out.append({"scenario_id": sid, "name": sc.name, "preset": req.preset,
                            "n_transactions": req.n_transactions, "description": sc.description,
                            "assumed_initial_failure_prob": {k.value: v for k, v in sc.failure_model.initial_failure_prob.items()}})
        finally:
            conn.close()
        return {"experiment_id": experiment_id, "scenarios": out, **PROVENANCE}

    @mcp.tool
    def run_payment_simulation(experiment_id: str, scenario_id: str, policy: PolicySpec,
                               n_trials: int = Field(default=10, ge=1, le=MAX_TRIALS),
                               master_seed: int = Field(default=0, ge=0),
                               max_work_units: Optional[int] = Field(default=None, ge=1)) -> dict[str, Any]:
        """Run a seeded Monte Carlo of ONE recovery policy on a scenario and return metrics with 95% CIs,
        plus its objective score and hard-constraint feasibility. Reproducible for the same inputs."""
        conn = store.connect()
        try:
            _, scenario, w, hc = _load(conn, experiment_id, scenario_id)
            pol = _policy(policy)
            units = _check_budget(scenario.workload.n_transactions, n_trials, 1, max_work_units)
            res = compare_policies_monte_carlo(scenario, [pol], n_trials, master_seed)[pol.name]
            ev = evaluate_policy(res, scenario.rules, w, hc)
            out = {"experiment_id": experiment_id, "scenario_id": scenario_id, "policy_definition": _policy_dict(pol),
                   "summary": _summary(res), "evaluation": ev.to_dict(), "master_seed": master_seed,
                   "work_units": units, **PROVENANCE}
            out["run_id"] = store.add_run(conn, experiment_id, scenario_id, "run", master_seed, n_trials,
                                          {"policy": _policy_dict(pol)}, out)
        finally:
            conn.close()
        return out

    @mcp.tool
    def compare_recovery_policies(experiment_id: str, scenario_id: str,
                                  policies: list[PolicySpec] = Field(min_length=2, max_length=8),
                                  n_trials: int = Field(default=10, ge=2, le=MAX_TRIALS),
                                  master_seed: int = Field(default=0, ge=0),
                                  max_work_units: Optional[int] = Field(default=None, ge=1)) -> dict[str, Any]:
        """Compare several recovery policies on IDENTICAL simulated workloads. Scoring and ranking are
        computed deterministically in code (hard constraints first, then weighted score); includes paired
        score differences vs the top-ranked policy. Baselines: immediate_retry, exponential_backoff,
        intelligent_recovery, no_recovery."""
        conn = store.connect()
        try:
            _, scenario, w, hc = _load(conn, experiment_id, scenario_id)
            pols = [_policy(p) for p in policies]
            units = _check_budget(scenario.workload.n_transactions, n_trials, len(pols), max_work_units)
            results = compare_policies_monte_carlo(scenario, pols, n_trials, master_seed)
            evals = {n: evaluate_policy(r, scenario.rules, w, hc) for n, r in results.items()}
            ranked = rank_policies(list(evals.values()))
            best = ranked[0]
            out = {
                "experiment_id": experiment_id, "scenario_id": scenario_id, "master_seed": master_seed,
                "n_trials": n_trials, "n_transactions": scenario.workload.n_transactions,
                "ranking": [{"rank": i + 1, **e.to_dict()} for i, e in enumerate(ranked)],
                "best_policy": best.policy_name, "best_policy_feasible": best.feasible,
                "paired_score_difference_vs_best": {
                    e.policy_name: paired_difference(best.trial_scores, e.trial_scores)
                    for e in ranked[1:]},
                "policies": {n: {"definition": _policy_dict(r.policy), "summary": _summary(r)} for n, r in results.items()},
                "work_units": units,
                "scoring_note": "score = w_rev*R - w_cost*C - w_lat*L - w_unn*U per trial; computed in code, not by an LLM.",
                **PROVENANCE}
            out["run_id"] = store.add_run(conn, experiment_id, scenario_id, "comparison", master_seed, n_trials,
                                          {"policies": [_policy_dict(p) for p in pols]}, out)
        finally:
            conn.close()
        return out

    @mcp.tool
    def run_sensitivity_analysis(experiment_id: str, scenario_id: str,
                                 policies: list[PolicySpec] = Field(min_length=2, max_length=6),
                                 parameters: Optional[list[SweepSpec]] = None,
                                 n_trials: int = Field(default=6, ge=2, le=30),
                                 master_seed: int = Field(default=0, ge=0),
                                 sweep_n_transactions: int = Field(default=5000, ge=500, le=50_000),
                                 max_work_units: Optional[int] = Field(default=None, ge=1)) -> dict[str, Any]:
        """Controlled one-at-a-time sensitivity analysis of the policy ranking. Varies (ranges are ASSUMED,
        not empirical): network_instability, buyer_response_probability, failure_persistence, workload_size,
        latency_weight, terminal_share, buyer_action_share. Every point uses the same seed; all policies see
        identical workloads. Returns, per point, recovery rates with CIs, objective scores, feasibility and
        rank; where the winner/ranking changes; which parameters the recommendation is most sensitive to;
        and a robustness label. Omit `parameters` to sweep all with default grids."""
        conn = store.connect()
        try:
            _, scenario, w, hc = _load(conn, experiment_id, scenario_id)
            pols = [_policy(p) for p in policies]
            names = [p.name for p in pols]
            if len(set(names)) != len(names):
                raise ValueError("policy names must be unique")
            sweep_sc = replace(scenario, workload=replace(scenario.workload, n_transactions=sweep_n_transactions))
            chosen = {sp.parameter: sp.values for sp in parameters} if parameters else None
            plan = sweep_plan(chosen, sweep_n_transactions, w.latency)
            units = estimate_work_units(plan, sweep_n_transactions, n_trials, len(pols))
            if units > MAX_SENSITIVITY_WORK_UNITS:
                raise ValueError(f"sensitivity analysis too large ({units} > {MAX_SENSITIVITY_WORK_UNITS}); "
                                 "use fewer parameters/values/policies/trials")
            if max_work_units is not None and units > max_work_units:
                raise ValueError(f"simulation budget exceeded: this call needs {units} work units but only {max_work_units} remain")
            report = run_sensitivity(sweep_sc, pols, w, hc, n_trials, master_seed, chosen)
            # report goes AFTER PROVENANCE: its detailed `assumptions` dict must not be replaced by the label string
            out = {"experiment_id": experiment_id, "scenario_id": scenario_id, **PROVENANCE, **report,
                   "policy_definitions": {p.name: _policy_dict(p) for p in pols}}
            out["run_id"] = store.add_run(conn, experiment_id, scenario_id, "sensitivity", master_seed, n_trials,
                                          {"policies": [_policy_dict(p) for p in pols], "parameters": chosen,
                                           "sweep_n_transactions": sweep_n_transactions}, out)
        finally:
            conn.close()
        return out

    @mcp.tool
    def get_payment_experiment_results(experiment_id: str) -> dict[str, Any]:
        """Return the stored experiment config, scenarios and all stored simulation runs/comparisons."""
        conn = store.connect()
        try:
            exp = store.get_experiment(conn, experiment_id)
            if exp is None:
                raise ValueError(f"unknown experiment_id {experiment_id!r}")
            return {"experiment": exp, "scenarios": store.list_scenarios(conn, experiment_id),
                    "runs": store.list_runs(conn, experiment_id), **PROVENANCE}
        finally:
            conn.close()

    @mcp.tool
    def get_payment_transaction_trace(experiment_id: str, scenario_id: str, policy: PolicySpec,
                                      master_seed: int = Field(default=0, ge=0),
                                      trial_index: int = Field(default=0, ge=0, le=MAX_TRIALS - 1),
                                      transaction_index: Optional[int] = Field(default=None, ge=0),
                                      outcome_filter: Optional[Literal["RECOVERED", "UNRECOVERED", "ESCALATED_FOR_REVIEW"]] = None
                                      ) -> dict[str, Any]:
        """Return the full simulated event trace of one transaction under a policy (re-simulated
        deterministically, identical to its outcome in the full run). Give transaction_index, or an
        outcome_filter to pick the first transaction in the trial with that recovery outcome."""
        conn = store.connect()
        try:
            _, scenario, _, _ = _load(conn, experiment_id, scenario_id)
        finally:
            conn.close()
        pol = _policy(policy)
        if transaction_index is None:
            from engine.payments.simulator import simulate_trial
            seed = trial_seeds(master_seed, trial_index + 1)[trial_index]
            wl, lat = prepare_trial(scenario.workload, scenario.failure_model, seed)
            run = simulate_trial(scenario, pol, wl, lat)
            want = outcome_filter or "RECOVERED"
            matches = [i for i, t in enumerate(run.transactions) if t.recovery_status.value == want]
            if not matches:
                raise ValueError(f"no transaction with recovery_status={want} in trial {trial_index}")
            transaction_index = matches[0]
        txn, events = trace_transaction(scenario, pol, master_seed, trial_index, transaction_index)
        return {"transaction": txn.to_dict(), "transaction_index": transaction_index, "trial_index": trial_index,
                "master_seed": master_seed, "policy": pol.name, "events": [e.to_dict() for e in events], **PROVENANCE}
