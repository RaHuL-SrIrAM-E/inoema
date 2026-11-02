from dataclasses import replace

from engine.montecarlo import run_monte_carlo
from tests.conftest import make_config


def test_mc_reproducibility_same_master_seed():
    config = make_config()
    r1 = run_monte_carlo(config, n_trials=40, master_seed=123)
    r2 = run_monte_carlo(config, n_trials=40, master_seed=123)
    assert r1.missed_fractions == r2.missed_fractions
    assert r1.p_shortfall == r2.p_shortfall
    assert r1.mean_missed_fraction == r2.mean_missed_fraction


def test_safe_early_release_is_low_risk():
    # Release immediately (t=0), giving the full window before close.
    config = make_config(release_schedule=[(0.0, 2000)])
    result = run_monte_carlo(config, n_trials=60, master_seed=1)
    assert result.p_shortfall < 0.05
    assert result.mean_missed_fraction < 0.01


def test_risky_late_release_is_high_risk():
    # Only 10 minutes between release and reconciliation close for 2000
    # payments plus backlog -- nowhere near enough throughput.
    config = make_config(release_schedule=[(140.0, 2000)])
    result = run_monte_carlo(config, n_trials=60, master_seed=1)
    assert result.p_shortfall > 0.5
    assert result.mean_missed_fraction > 0.05


def test_risk_monotonicity_with_release_time():
    release_times = [20.0, 70.0, 110.0, 140.0]
    mean_missed = []
    for release_time in release_times:
        config = make_config(
            batch_size=600,
            release_schedule=[(release_time, 600)],
            initial_queue_depth=80,
            reconciliation_close=150.0,
        )
        result = run_monte_carlo(config, n_trials=150, master_seed=99)
        mean_missed.append(result.mean_missed_fraction)

    # Allow a small tolerance for Monte Carlo noise, but the trend across
    # releases spaced this far apart must be non-decreasing.
    tolerance = 0.02
    for earlier, later in zip(mean_missed, mean_missed[1:]):
        assert later >= earlier - tolerance
    assert mean_missed[-1] > mean_missed[0]


def test_risk_monotonicity_with_worker_count():
    worker_counts = [1, 2, 8, 20]
    risk_scores = []
    for n_workers in worker_counts:
        config = make_config(
            batch_size=500,
            release_schedule=[(100.0, 500)],
            initial_queue_depth=50,
            n_workers=n_workers,
            reconciliation_close=150.0,
        )
        result = run_monte_carlo(config, n_trials=100, master_seed=55)
        risk_scores.append(result.mean_missed_fraction)

    for more_workers_score, fewer_workers_score in zip(risk_scores[1:], risk_scores):
        assert more_workers_score <= fewer_workers_score


def test_two_tranche_split_release_reduces_risk_vs_single_late_release():
    # Same total batch and backlog, same close time: one scenario releases
    # everything late in a single burst, the other splits it into two
    # earlier tranches. This is the exact mitigation move the agent is
    # expected to make, so it needs direct coverage rather than just
    # inference from the single-release sweep.
    common = dict(
        batch_size=600,
        initial_queue_depth=60,
        n_workers=8,
        reconciliation_close=150.0,
    )
    single_late = make_config(**common, release_schedule=[(140.0, 600)])
    split_early = make_config(
        **common, release_schedule=[(90.0, 300), (110.0, 300)]
    )

    result_single = run_monte_carlo(single_late, n_trials=150, master_seed=2024)
    result_split = run_monte_carlo(split_early, n_trials=150, master_seed=2024)

    assert result_split.mean_missed_fraction < result_single.mean_missed_fraction
    assert result_split.p_shortfall < result_single.p_shortfall


def test_retry_behaviour_increases_missed_fraction():
    tight_timing = dict(
        batch_size=400,
        release_schedule=[(110.0, 400)],
        initial_queue_depth=40,
        n_workers=8,
        reconciliation_close=150.0,
        retry_delay=5.0,
        max_retries=5,
    )
    no_failures = make_config(**tight_timing, failure_prob=0.0)
    high_failures = make_config(**tight_timing, failure_prob=0.3)

    result_no_failures = run_monte_carlo(no_failures, n_trials=100, master_seed=321)
    result_high_failures = run_monte_carlo(high_failures, n_trials=100, master_seed=321)

    assert result_high_failures.mean_missed_fraction >= result_no_failures.mean_missed_fraction
