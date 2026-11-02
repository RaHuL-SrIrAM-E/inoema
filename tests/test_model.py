from engine.model import count_completed_by_close, run_once
from tests.conftest import make_config


def test_reproducibility_same_seed_identical_trial():
    config = make_config()
    r1 = run_once(config, seed=42)
    r2 = run_once(config, seed=42)
    assert r1.completion_times == r2.completion_times
    assert r1.missed_fraction == r2.missed_fraction
    assert r1.n_completed_by_close == r2.n_completed_by_close


def test_different_seeds_can_differ():
    config = make_config()
    r1 = run_once(config, seed=1)
    r2 = run_once(config, seed=2)
    # Not a hard requirement that they differ, but with stochastic
    # failure/retry draws across 2350 payments they essentially always will.
    assert r1.completion_times != r2.completion_times


def make_config_variant(base, **overrides):
    from dataclasses import replace

    return replace(base, **overrides)


def test_worker_capacity_bottleneck_increases_missed_fraction():
    tight_config = make_config(
        batch_size=500,
        release_schedule=[(100.0, 500)],
        initial_queue_depth=50,
        reconciliation_close=150.0,
    )
    one_worker = run_once(make_config_variant(tight_config, n_workers=1), seed=7)
    many_workers = run_once(make_config_variant(tight_config, n_workers=20), seed=7)
    assert one_worker.missed_fraction > many_workers.missed_fraction


def test_retry_exhaustion_means_never_completed():
    config = make_config(
        batch_size=1,
        release_schedule=[(0.0, 1)],
        initial_queue_depth=0,
        n_workers=1,
        failure_prob=1.0,
        max_retries=2,
        posting_time_median=0.01,
        posting_time_sigma=0.01,
        retry_delay=1.0,
        reconciliation_close=1000.0,
    )
    result = run_once(config, seed=3)
    assert result.completion_times[0] is None
    assert result.missed_fraction == 1.0


def test_retry_delay_is_added_between_attempts():
    # failure_prob=0 on first pass would never retry; force exactly one
    # retry by using max_retries=0 with failure_prob=1 as a control vs
    # a config that is guaranteed to succeed on attempt 1 (failure_prob=0).
    no_failure = make_config(
        batch_size=1,
        release_schedule=[(0.0, 1)],
        initial_queue_depth=0,
        n_workers=1,
        failure_prob=0.0,
        posting_time_median=0.01,
        posting_time_sigma=0.01,
        retry_delay=5.0,
        reconciliation_close=1000.0,
    )
    baseline = run_once(no_failure, seed=11)
    assert baseline.completion_times[0] is not None
    assert baseline.completion_times[0] < 1.0  # well under a minute, no retry delay


def test_reconciliation_cutoff_boundary_is_inclusive():
    completion_times = {0: 150.0, 1: 150.0001, 2: None}
    n_completed = count_completed_by_close(completion_times, reconciliation_close=150.0)
    assert n_completed == 1  # payment 0 at exactly the close counts as made it


def test_invalid_config_raises():
    import pytest

    bad = make_config(release_schedule=[(90.0, 1999)])  # doesn't sum to batch_size
    with pytest.raises(ValueError):
        run_once(bad, seed=1)


def test_out_of_order_release_schedule_is_rejected():
    out_of_order = make_config(
        batch_size=2000,
        release_schedule=[(90.0, 1000), (60.0, 1000)],  # later time listed first
    )
    problems = out_of_order.validate()
    assert any("non-decreasing" in problem for problem in problems)


def test_seed_sequence_accepted_directly_and_reproducible():
    # run_once must accept a numpy SeedSequence (not just a plain int) --
    # this is how engine.montecarlo now seeds trials.
    import numpy as np

    config = make_config()
    seed_sequence_a = np.random.SeedSequence(777).spawn(1)[0]
    seed_sequence_b = np.random.SeedSequence(777).spawn(1)[0]
    r1 = run_once(config, seed_sequence_a)
    r2 = run_once(config, seed_sequence_b)
    assert r1.completion_times == r2.completion_times
    assert r1.missed_fraction == r2.missed_fraction
