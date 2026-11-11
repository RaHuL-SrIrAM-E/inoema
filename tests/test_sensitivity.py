"""Sensitivity analysis: controlled sweeps, flip detection, reproducibility."""
import pytest

from engine.objectives import HardConstraints, ObjectiveWeights, evaluate_policy, rank_policies
from engine.payments.failure_model import FailureModel
from engine.payments.models import FailureCategory as FC
from engine.payments.recovery_policy import exponential_backoff, immediate_retry, intelligent_recovery
from engine.payments.scenarios import build_scenario
from engine.payments.sensitivity import (PARAMETERS, apply_parameter, estimate_work_units, run_sensitivity,
                                         sweep_plan, validate_value)
from engine.payments.simulator import compare_policies_monte_carlo

FREE = HardConstraints(None, None, False, False, False)
POLS = lambda: [immediate_retry(), exponential_backoff(), intelligent_recovery()]  # noqa: E731


def test_all_requested_dimensions_exist():
    assert {"network_instability", "buyer_response_probability", "failure_persistence", "workload_size",
            "latency_weight", "terminal_share", "buyer_action_share"} == set(PARAMETERS)


def test_sweep_plan_excludes_baseline_and_validates():
    plan = sweep_plan({"network_instability": [1.0, 2.0, 3.0]}, 5000, 0.1)
    assert plan == [("network_instability", 2.0), ("network_instability", 3.0)]
    assert sweep_plan({"latency_weight": [0.1, 5.0]}, 5000, 0.1) == [("latency_weight", 5.0)]  # baseline = experiment weight
    assert ("workload_size", 5000.0) not in sweep_plan({"workload_size": [5000, 1000]}, 5000, 0.1)
    assert len(sweep_plan(None, 5000, 0.1)) == sum(len(p.default_values) for p in PARAMETERS.values())  # no default grid contains its baseline
    with pytest.raises(ValueError):
        sweep_plan({"nope": [1.0]}, 5000, 0.1)
    for bad in [("network_instability", 0.0), ("network_instability", -1.0), ("workload_size", 50.0),
                ("workload_size", 1000.5), ("latency_weight", -0.1), ("failure_persistence", 1000.0)]:
        with pytest.raises(ValueError):
            validate_value(*bad)
    with pytest.raises(ValueError):
        sweep_plan({"network_instability": [1.5] * 61}, 5000, 0.1)


def test_apply_parameter_changes_only_the_target():
    sc = build_scenario("baseline", 2000)
    w = ObjectiveWeights()
    base_mix = sc.failure_model.category_mix[list(sc.failure_model.category_mix)[0]]

    s2, _ = apply_parameter(sc, w, "network_instability", 3.0)
    row0 = list(s2.failure_model.category_mix)[0]
    new = s2.failure_model.category_mix[row0]
    assert new[FC.TRANSIENT_NETWORK_ERROR] > base_mix[FC.TRANSIENT_NETWORK_ERROR] * 2
    assert sum(new.values()) == pytest.approx(1.0) and s2.failure_model.validate() == []
    assert s2.failure_model.buyer_response_prob == sc.failure_model.buyer_response_prob  # untouched

    s3, _ = apply_parameter(sc, w, "buyer_response_probability", 1.5)
    assert s3.failure_model.buyer_response_prob[FC.PAYER_ACTION_REQUIRED] == pytest.approx(0.9)
    assert s3.failure_model.buyer_response_prob[FC.INSTRUMENT_DECLINED] == pytest.approx(0.825)
    s3b, _ = apply_parameter(sc, w, "buyer_response_probability", 100.0)
    assert max(s3b.failure_model.buyer_response_prob.values()) == 1.0  # clipped

    s4, _ = apply_parameter(sc, w, "failure_persistence", 2.0)
    assert s4.failure_model.mean_clear_minutes[FC.TEMPORARY_PROCESSING_ERROR] == 40.0
    assert s4.failure_model.mean_clear_minutes[FC.INSTRUMENT_DECLINED] == 0.0

    s5, _ = apply_parameter(sc, w, "workload_size", 777)
    assert s5.workload.n_transactions == 777 and sc.workload.n_transactions == 2000  # original not mutated

    _, w2 = apply_parameter(sc, w, "latency_weight", 4.0)
    assert w2.latency == 4.0 and w.latency == 0.1

    s6, _ = apply_parameter(sc, w, "terminal_share", 4.0)
    r = s6.failure_model.category_mix[row0]
    assert r[FC.TERMINAL_PAYMENT_FAILURE] + r[FC.VALIDATION_ERROR] > 0.3
    s7, _ = apply_parameter(sc, w, "buyer_action_share", 2.0)
    assert sum(s7.failure_model.category_mix[row0][c] for c in (FC.INSTRUMENT_DECLINED, FC.EXPIRED_PAYMENT_METHOD,
                                                                 FC.PAYER_ACTION_REQUIRED, FC.ORDER_NOT_APPROVED)) > 0.6


def test_work_unit_estimate_counts_resimulated_points_only():
    plan = [("network_instability", 2.0), ("latency_weight", 5.0), ("workload_size", 10000.0)]
    assert estimate_work_units(plan, 5000, 6, 3) == 5000 * 6 * 3 + 5000 * 6 * 3 + 0 + 10000 * 6 * 3


@pytest.fixture(scope="module")
def report():
    sc = build_scenario("baseline", 3000)
    return run_sensitivity(sc, POLS(), ObjectiveWeights(), FREE, 5, 11,
                           {"latency_weight": [0.0, 5.0, 20.0], "network_instability": [0.5, 5.0],
                            "failure_persistence": [0.25, 4.0], "buyer_response_probability": [0.25, 1.5]})


def test_report_structure_and_recorded_assumptions(report):
    assert report["provenance"] == "SIMULATED" and "NOT_EMPIRICAL" in report["assumptions_label"]
    assert report["policies"] == ["immediate_retry", "exponential_backoff", "intelligent_recovery"]
    a = report["assumptions"]
    assert "ASSUMPTIONS" in a["statement"] and a["failure_model"]["initial_failure_prob"]["CARD"] == 0.08
    assert a["costs"]["attempt_cost"] == 0.03 and a["objective_weights"]["latency"] == 0.1
    assert set(a["parameters"]) == set(PARAMETERS)
    assert report["n_trials"] == 5 and report["master_seed"] == 11 and "identical master_seed" in report["design"]
    assert len(report["points"]) == 3 + 2 + 2 + 2  # latency 3 values + 2 values for each of 3 other parameters (none equals its baseline)
    for p in report["points"]:
        for r in p["ranking"]:
            lo, hi = r["recovery_rate_ci95"]
            assert lo <= r["recovery_rate"] <= hi and isinstance(r["feasible"], bool) and "score_mean" in r


def test_baseline_point_equals_direct_simulation(report):
    sc = build_scenario("baseline", 3000)
    res = compare_policies_monte_carlo(sc, POLS(), 5, 11)
    ranked = rank_policies([evaluate_policy(r, sc.rules, ObjectiveWeights(), FREE) for r in res.values()])
    base = report["baseline"]
    assert [r["policy"] for r in base["ranking"]] == [e.policy_name for e in ranked]
    assert base["ranking"][0]["score_mean"] == pytest.approx(ranked[0].score_mean)


def test_latency_weighting_flips_winner_and_is_flagged(report):
    assert report["baseline"]["winner"] == "intelligent_recovery"
    by = {p["value"]: p for p in report["points"] if p["parameter"] == "latency_weight"}
    assert by[0.0]["winner"] == "intelligent_recovery" and not by[0.0]["winner_changed"]
    assert by[20.0]["winner"] == "immediate_retry" and by[20.0]["winner_changed"] is True
    assert {"parameter": "latency_weight", "value": 20.0, "new_winner": "immediate_retry"} in report["robustness"]["flip_conditions"]
    assert report["most_sensitive_parameters"][0] == "latency_weight"
    # latency weighting is objective-side: identical simulation => identical recovery rates at every point
    rrs = {tuple(round(r["recovery_rate"], 12) for r in sorted(p["ranking"], key=lambda r: r["policy"]))
           for p in by.values()}
    assert len(rrs) == 1
    assert report["robustness"]["winner_stable_share"] < 1.0 and report["robustness"]["label"] in ("ROBUST", "MODERATE", "FRAGILE")


def test_directional_effects_of_assumptions(report):
    rr = lambda param, val, pol: next(r["recovery_rate"] for p in report["points"] if p["parameter"] == param and p["value"] == val  # noqa: E731
                                      for r in p["ranking"] if r["policy"] == pol)
    base = {r["policy"]: r["recovery_rate"] for r in report["baseline"]["ranking"]}
    assert rr("failure_persistence", 4.0, "exponential_backoff") < base["exponential_backoff"] < rr("failure_persistence", 0.25, "exponential_backoff")
    assert rr("buyer_response_probability", 1.5, "intelligent_recovery") > base["intelligent_recovery"] > rr("buyer_response_probability", 0.25, "intelligent_recovery")
    # buyer response is irrelevant to a blind policy (it never contacts buyers)
    assert rr("buyer_response_probability", 1.5, "immediate_retry") == pytest.approx(base["immediate_retry"])


def test_parameter_summary_sorted_by_sensitivity(report):
    keys = [(-s["n_winner_changes"], -s["n_ranking_changes"], s["min_margin"]) for s in report["parameter_summary"]]
    assert keys == sorted(keys)
    assert {s["parameter"] for s in report["parameter_summary"]} == {"latency_weight", "network_instability",
                                                                    "failure_persistence", "buyer_response_probability"}


def test_sensitivity_is_reproducible_and_seed_dependent():
    sc = build_scenario("baseline", 1500)
    kw = dict(parameters={"network_instability": [3.0]})
    a = run_sensitivity(sc, POLS(), ObjectiveWeights(), FREE, 3, 5, **kw)
    b = run_sensitivity(sc, POLS(), ObjectiveWeights(), FREE, 3, 5, **kw)
    c = run_sensitivity(sc, POLS(), ObjectiveWeights(), FREE, 3, 6, **kw)
    assert a == b and a != c


def test_feasibility_reported_per_point_with_default_constraints():
    sc = build_scenario("baseline", 1500)
    r = run_sensitivity(sc, POLS(), ObjectiveWeights(), HardConstraints(), 3, 5, {"network_instability": [2.0]})
    for p in [r["baseline"]] + r["points"]:
        feas = {row["policy"]: row["feasible"] for row in p["ranking"]}
        assert feas == {"intelligent_recovery": True, "exponential_backoff": False, "immediate_retry": False}
        assert p["margin"] is None and p["winner"] == "intelligent_recovery"  # only one feasible policy: no margin


def test_needs_two_policies():
    with pytest.raises(ValueError):
        run_sensitivity(build_scenario("baseline", 500), [intelligent_recovery()], ObjectiveWeights(), FREE, 2, 0)
