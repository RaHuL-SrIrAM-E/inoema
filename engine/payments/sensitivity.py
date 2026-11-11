"""Sensitivity analysis: how do policy rankings and the recommendation change when the
ASSUMED world (or the merchant's weighting) changes?

Design (controlled experiments):
  * One-at-a-time sweeps around a baseline. Everything except the swept parameter is held fixed.
  * Every sweep point uses the SAME master_seed. Within a point all policies face identical workloads
    and latent draws (common random numbers); across points the random streams are aligned too, so
    differences are caused by the parameter, not by fresh sampling noise.
  * latency_weight is an objective-side parameter: it re-scores the baseline simulation (no re-simulation).
  * All grids below are ASSUMED exploration ranges chosen for illustration -- they are not empirical
    distributions and not forecasts. Each report embeds the baseline assumptions and the grids used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Mapping, Optional, Sequence

from engine.objectives import (HardConstraints, ObjectiveWeights, PolicyEvaluation, evaluate_policy,
                               paired_difference, rank_policies)
from engine.payments.failure_model import DEFAULT_CLASSIFICATION, FailureRules, RecoveryClass
from engine.payments.models import (ASSUMPTION_LABEL, MODEL_VERSION, SIM_LABEL, FailureCategory as FC)
from engine.payments.recovery_policy import RecoveryPolicy
from engine.payments.scenarios import PaymentScenario
from engine.payments.simulator import PaymentMCResult, compare_policies_monte_carlo

TRANSIENT = (FC.TRANSIENT_NETWORK_ERROR, FC.TEMPORARY_PROCESSING_ERROR)
MAX_SWEEP_POINTS = 60


@dataclass(frozen=True)
class SweepParameter:
    name: str
    description: str
    baseline: float
    default_values: tuple[float, ...]  # non-baseline exploration grid (ASSUMED ranges)
    resimulate: bool = True


PARAMETERS: dict[str, SweepParameter] = {p.name: p for p in (
    SweepParameter("network_instability", "Multiplier on the share of transient network/processing failures "
                   "(other categories renormalised). 1.0 = baseline mix.", 1.0, (0.5, 2.0, 3.0, 5.0)),
    SweepParameter("buyer_response_probability", "Multiplier on the probability that a buyer responds to a "
                   "buyer-action request (clipped to 1).", 1.0, (0.25, 0.5, 1.25, 1.5)),
    SweepParameter("failure_persistence", "Multiplier on mean time (minutes) for transient failures to clear.",
                   1.0, (0.25, 0.5, 2.0, 4.0)),
    SweepParameter("workload_size", "Transactions per trial (absolute). Baseline = sweep_n_transactions.",
                   0.0, (1000.0, 2500.0, 10000.0, 20000.0)),
    SweepParameter("latency_weight", "Merchant objective weight on recovery latency (absolute; objective-side, "
                   "re-scores the baseline simulation).", 0.1, (0.0, 0.5, 2.0, 5.0, 10.0, 20.0), resimulate=False),
    SweepParameter("terminal_share", "Multiplier on the share of TERMINAL-class failures (others renormalised).",
                   1.0, (0.5, 2.0, 4.0)),
    SweepParameter("buyer_action_share", "Multiplier on the share of BUYER_ACTION-class failures (others renormalised).",
                   1.0, (0.5, 1.5, 2.0)),
)}


def _scale_dict(d: Mapping, f) -> dict:
    return {k: f(v) for k, v in d.items()}


def _class_multipliers(cls: RecoveryClass, m: float) -> dict:
    return {c: m for c, k in DEFAULT_CLASSIFICATION.items() if k == cls}


def validate_value(name: str, value: float) -> None:
    if name not in PARAMETERS:
        raise ValueError(f"unknown sensitivity parameter {name!r}; choose from {sorted(PARAMETERS)}")
    if name == "latency_weight":
        if value < 0:
            raise ValueError("latency_weight must be >= 0")
    elif name == "workload_size":
        if not 100 <= value <= 200_000 or int(value) != value:
            raise ValueError("workload_size must be an integer in [100, 200000]")
    elif value <= 0 or value > 100:
        raise ValueError(f"{name} multiplier must be in (0, 100]")


def apply_parameter(scenario: PaymentScenario, weights: ObjectiveWeights, name: str, value: float
                    ) -> tuple[PaymentScenario, ObjectiveWeights]:
    """Return a modified (scenario, weights) copy with the swept parameter applied."""
    validate_value(name, value)
    fm = scenario.failure_model
    if name == "network_instability":
        fm = fm.scaled(category_weight_multipliers={c: value for c in TRANSIENT})
    elif name == "terminal_share":
        fm = fm.scaled(category_weight_multipliers=_class_multipliers(RecoveryClass.TERMINAL, value))
    elif name == "buyer_action_share":
        fm = fm.scaled(category_weight_multipliers=_class_multipliers(RecoveryClass.BUYER_ACTION, value))
    elif name == "buyer_response_probability":
        fm = replace(fm, buyer_response_prob=_scale_dict(fm.buyer_response_prob, lambda p: min(1.0, p * value)))
    elif name == "failure_persistence":
        fm = replace(fm, mean_clear_minutes=_scale_dict(fm.mean_clear_minutes, lambda m: m * value))
    elif name == "workload_size":
        scenario = replace(scenario, workload=replace(scenario.workload, n_transactions=int(value)))
    elif name == "latency_weight":
        weights = replace(weights, latency=float(value))
    return replace(scenario, failure_model=fm), weights


def sweep_plan(parameters: Optional[Mapping[str, Optional[Sequence[float]]]], base_n: int,
               base_latency_weight: float) -> list[tuple[str, float]]:
    """Ordered (parameter, value) points; `None` parameters -> every parameter with its default grid."""
    plan: list[tuple[str, float]] = []
    chosen = parameters if parameters else {n: None for n in PARAMETERS}
    for name, values in chosen.items():
        if name not in PARAMETERS:
            raise ValueError(f"unknown sensitivity parameter {name!r}; choose from {sorted(PARAMETERS)}")
        vals = list(values) if values else list(PARAMETERS[name].default_values)
        baseline = {"workload_size": float(base_n), "latency_weight": base_latency_weight}.get(name, PARAMETERS[name].baseline)
        for v in vals:
            validate_value(name, v)
            if v != baseline:  # baseline is evaluated once, separately
                plan.append((name, float(v)))
    if len(plan) > MAX_SWEEP_POINTS:
        raise ValueError(f"too many sweep points ({len(plan)} > {MAX_SWEEP_POINTS})")
    return plan


def estimate_work_units(plan: Sequence[tuple[str, float]], base_n: int, n_trials: int, n_policies: int) -> int:
    """Transactions x trials x policies simulated (latency_weight points are re-scored, not re-simulated)."""
    units = base_n * n_trials * n_policies  # baseline
    for name, v in plan:
        if PARAMETERS[name].resimulate:
            units += (int(v) if name == "workload_size" else base_n) * n_trials * n_policies
    return units


def _point(evals: Sequence[PolicyEvaluation], results: Mapping[str, PaymentMCResult]) -> dict:
    ranked = rank_policies(evals)
    feasible = [e for e in ranked if e.feasible]
    margin = None
    if len(feasible) >= 2:
        d = paired_difference(feasible[0].trial_scores, feasible[1].trial_scores)
        margin = {"between": [feasible[0].policy_name, feasible[1].policy_name], "mean": d["mean_difference"],
                  "ci95": d["ci95"], "significant": d["ci_excludes_zero"]}
    rows = []
    for i, e in enumerate(ranked):
        rr = results[e.policy_name].aggregate.pooled["recovery_rate"]
        rows.append({"rank": i + 1, "policy": e.policy_name, "feasible": e.feasible, "score_mean": e.score_mean,
                     "score_ci95": list(e.score_ci95) if e.score_ci95 else None,
                     "recovery_rate": rr["value"], "recovery_rate_ci95": rr["ci95_wilson"],
                     "n_violations": len(e.violations)})
    return {"winner": feasible[0].policy_name if feasible else None, "ranking": rows, "margin": margin}


def _assumptions(scenario: PaymentScenario, weights: ObjectiveWeights, hc: HardConstraints) -> dict:
    fm = scenario.failure_model
    jsonable = lambda m: {getattr(k, "value", k): (v if not isinstance(v, Mapping) else {getattr(kk, "value", kk): vv for kk, vv in v.items()}) for k, v in m.items()}  # noqa: E731
    return {
        "label": ASSUMPTION_LABEL, "model_version": MODEL_VERSION,
        "statement": "All values are ASSUMPTIONS, not empirical statistics. Sweeps vary one parameter at a time over "
                     "illustrative ranges; they describe sensitivity of the SIMULATED recommendation only.",
        "scenario": {"name": scenario.name, "description": scenario.description},
        "failure_model": {"initial_failure_prob": jsonable(fm.initial_failure_prob), "category_mix": jsonable(fm.category_mix),
                          "mean_clear_minutes": jsonable(fm.mean_clear_minutes),
                          "retry_success_after_clear": jsonable(fm.retry_success_after_clear),
                          "buyer_response_prob": jsonable(fm.buyer_response_prob),
                          "buyer_response_mean_minutes": jsonable(fm.buyer_response_mean_minutes),
                          "post_buyer_success_prob": jsonable(fm.post_buyer_success_prob)},
        "costs": asdict(scenario.costs), "objective_weights": asdict(weights), "hard_constraints": asdict(hc),
        "parameters": {n: {"description": p.description, "baseline": p.baseline, "default_grid": list(p.default_values)}
                       for n, p in PARAMETERS.items()},
    }


def run_sensitivity(scenario: PaymentScenario, policies: Sequence[RecoveryPolicy], weights: ObjectiveWeights,
                    hc: HardConstraints, n_trials: int, master_seed: int,
                    parameters: Optional[Mapping[str, Optional[Sequence[float]]]] = None) -> dict:
    """Run the controlled sweeps and return a structured, JSON-serialisable report."""
    if len(policies) < 2:
        raise ValueError("sensitivity analysis needs at least two policies")
    base_n = scenario.workload.n_transactions
    plan = sweep_plan(parameters, base_n, weights.latency)

    base_results = compare_policies_monte_carlo(scenario, list(policies), n_trials, master_seed)
    base_evals = [evaluate_policy(r, scenario.rules, weights, hc) for r in base_results.values()]
    baseline = _point(base_evals, base_results)
    base_rank = [r["policy"] for r in baseline["ranking"]]
    base_winner = baseline["winner"]

    points = []
    for name, value in plan:
        if PARAMETERS[name].resimulate:
            sc2, w2 = apply_parameter(scenario, weights, name, value)
            results = compare_policies_monte_carlo(sc2, list(policies), n_trials, master_seed)
        else:
            _, w2 = apply_parameter(scenario, weights, name, value)
            results = base_results
        evals = [evaluate_policy(r, scenario.rules, w2, hc) for r in results.values()]
        pt = _point(evals, results)
        pt.update({"parameter": name, "value": value,
                   "winner_changed": pt["winner"] != base_winner,
                   "ranking_changed": [r["policy"] for r in pt["ranking"]] != base_rank})
        points.append(pt)

    summary = []
    for name in dict.fromkeys(n for n, _ in plan):
        pts = [p for p in points if p["parameter"] == name]
        margins = [(p["margin"]["mean"], p["value"]) for p in pts if p["margin"]]
        if baseline["margin"]:
            margins.append((baseline["margin"]["mean"], "baseline"))
        flips = [{"value": p["value"], "new_winner": p["winner"]} for p in pts if p["winner_changed"]]
        base_scores = [next((r["score_mean"] for r in p["ranking"] if r["policy"] == base_winner), None) for p in pts]
        base_scores = [s for s in base_scores if s is not None]
        summary.append({
            "parameter": name, "values_tested": [p["value"] for p in pts],
            "n_winner_changes": len(flips), "flip_conditions": flips,
            "n_ranking_changes": sum(p["ranking_changed"] for p in pts),
            "min_margin": min((m for m, _ in margins), default=None),
            "min_margin_at": min(margins, key=lambda m: m[0], default=(None, None))[1],
            "baseline_winner_score_range": [min(base_scores), max(base_scores)] if base_scores else None,
        })
    summary.sort(key=lambda s: (-s["n_winner_changes"], -s["n_ranking_changes"],
                                s["min_margin"] if s["min_margin"] is not None else float("inf"), s["parameter"]))

    stable = sum(not p["winner_changed"] for p in points)
    share = stable / len(points) if points else 1.0
    label = "ROBUST" if share >= 0.9 else "MODERATE" if share >= 0.6 else "FRAGILE"
    return {
        "provenance": "SIMULATED", "label": SIM_LABEL, "assumptions_label": ASSUMPTION_LABEL,
        "n_trials": n_trials, "master_seed": master_seed, "n_transactions_baseline": base_n,
        "policies": [p.name for p in policies],
        "design": "one-at-a-time sweeps; identical master_seed at every point; identical workloads across policies within a point",
        "assumptions": _assumptions(scenario, weights, hc),
        "baseline": {"winner": base_winner, **baseline},
        "points": points, "parameter_summary": summary,
        "most_sensitive_parameters": [s["parameter"] for s in summary if s["n_winner_changes"] or s["n_ranking_changes"]][:3],
        "robustness": {"n_points": len(points), "winner_stable_points": stable, "winner_stable_share": share,
                       "label": label, "baseline_winner": base_winner,
                       "flip_conditions": [{"parameter": s["parameter"], **f} for s in summary for f in s["flip_conditions"]],
                       "note": "Heuristic: ROBUST >= 90% of sweep points keep the baseline winner, MODERATE >= 60%, else FRAGILE. "
                               "Describes sensitivity within the simulation under the swept assumptions only."},
        "work_units": estimate_work_units(plan, base_n, n_trials, len(policies)),
    }
