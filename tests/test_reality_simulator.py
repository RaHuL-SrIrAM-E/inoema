import inspect

from collector.reality_simulator import RealityConfig, simulate_reality

COMMON = dict(
    batch_size=300,
    release_schedule=[(0.0, 300)],
    n_workers=4,
    reconciliation_close=20.0,
    initial_queue_depth=30,
)


def test_reality_reproducibility():
    r1 = simulate_reality(**COMMON, reality_seed=7)
    r2 = simulate_reality(**COMMON, reality_seed=7)
    assert r1.completion_times == r2.completion_times
    assert r1.missed_fraction == r2.missed_fraction
    assert r1.day_slowdown_multiplier == r2.day_slowdown_multiplier
    assert r1.worker_outage_minutes == r2.worker_outage_minutes


def test_reality_has_no_predictor_dependency():
    # The function only ever takes public scenario facts -- no predicted
    # risk score, verdict, or Monte Carlo result can be passed in.
    params = inspect.signature(simulate_reality).parameters
    for forbidden in ("mc_result", "verdict", "risk_score", "predicted"):
        assert forbidden not in params


def test_reality_varies_across_seeds():
    outcomes = {simulate_reality(**COMMON, reality_seed=seed).missed_fraction for seed in range(10)}
    assert len(outcomes) > 1


def test_hidden_slowdown_increases_missed_fraction():
    no_slowdown = RealityConfig(slowdown_prob=0.0, outage_prob_per_worker=0.0)
    forced_slowdown = RealityConfig(
        slowdown_prob=1.0, slowdown_multiplier_range=(2.5, 2.5), outage_prob_per_worker=0.0
    )

    baseline = simulate_reality(**COMMON, reality_seed=1, reality_config=no_slowdown)
    slowed = simulate_reality(**COMMON, reality_seed=1, reality_config=forced_slowdown)

    assert slowed.day_slowdown_multiplier > baseline.day_slowdown_multiplier
    assert slowed.missed_fraction > baseline.missed_fraction


def test_hidden_worker_outage_increases_missed_fraction():
    no_outage = RealityConfig(slowdown_prob=0.0, outage_prob_per_worker=0.0)
    forced_early_outage = RealityConfig(
        slowdown_prob=0.0,
        outage_prob_per_worker=1.0,
        outage_duration_range=(30.0, 30.0),
        outage_start_fraction_range=(0.0, 0.0),
    )

    baseline = simulate_reality(**COMMON, reality_seed=1, reality_config=no_outage)
    outaged = simulate_reality(**COMMON, reality_seed=1, reality_config=forced_early_outage)

    assert all(minutes == 0.0 for minutes in baseline.worker_outage_minutes)
    assert all(minutes == 30.0 for minutes in outaged.worker_outage_minutes)
    assert outaged.missed_fraction > baseline.missed_fraction
