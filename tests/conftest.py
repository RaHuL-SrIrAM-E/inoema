from engine.model import SimConfig


def make_config(**overrides) -> SimConfig:
    """Base scenario matching the brief's illustrative parameters, in
    minutes with t=0 == 17:00. Any field can be overridden per test."""
    defaults = dict(
        batch_size=2000,
        release_schedule=[(90.0, 2000)],  # 18:30
        n_workers=8,
        reconciliation_close=150.0,  # 19:30
        ledger_cutoff=30.0,  # 17:30
        posting_time_median=0.15,
        posting_time_sigma=0.4,
        failure_prob=0.005,
        retry_delay=5.0,
        initial_queue_depth=350,
        max_retries=5,
    )
    defaults.update(overrides)
    return SimConfig(**defaults)
