from agent.demo_agent import AgentAction
from execution.mock_paypal import (
    get_batch,
    get_batch_status,
    get_payment_item_statuses,
    submit_payout_batch,
)


def _action(label="test_action"):
    return AgentAction(label=label, release_schedule=((0.0, 10),), rationale="test")


def test_submit_and_retrieve_status():
    batch = submit_payout_batch(
        _action(), batch_size=10, n_workers=4, reconciliation_close=100.0, initial_queue_depth=0
    )
    assert get_batch_status(batch.batch_id) == "SUBMITTED"
    statuses = get_payment_item_statuses(batch.batch_id)
    assert len(statuses) == 10
    assert all(item.status == "SUBMITTED" for item in statuses)
    assert get_batch(batch.batch_id).batch_id == batch.batch_id
    assert get_batch(batch.batch_id).synthetic is True


def test_unknown_batch_id_returns_none():
    assert get_batch_status("not-a-real-batch-id") is None
    assert get_payment_item_statuses("not-a-real-batch-id") is None
    assert get_batch("not-a-real-batch-id") is None


def test_unique_batch_ids():
    action = _action()
    batch1 = submit_payout_batch(
        action, batch_size=5, n_workers=2, reconciliation_close=50.0, initial_queue_depth=0
    )
    batch2 = submit_payout_batch(
        action, batch_size=5, n_workers=2, reconciliation_close=50.0, initial_queue_depth=0
    )
    assert batch1.batch_id != batch2.batch_id
