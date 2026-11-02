"""Mocked PayPal-style payout execution service.

No real PayPal API is ever called. This module only ever records that a
batch was "submitted" and tracks mocked statuses for it -- it has no
opinion about whether payments actually post in time; that is determined
independently by collector.reality_simulator. Every record produced here
is explicitly labelled MOCKED.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

MOCKED_LABEL = "SYNTHETIC_MOCKED_EXECUTION"
STATUS_SUBMITTED = "SUBMITTED"


@dataclass(frozen=True)
class MockPaymentItem:
    """A single payment item's mocked gateway status (not its operational
    outcome -- the gateway doesn't know if it actually posts in time)."""

    payment_id: int
    status: str = STATUS_SUBMITTED


@dataclass(frozen=True)
class MockPayoutBatch:
    """A mocked payout batch submission record."""

    batch_id: str
    batch_size: int
    release_schedule: tuple[tuple[float, int], ...]
    n_workers: int
    reconciliation_close: float
    initial_queue_depth: int
    submitted_at: str
    status: str
    items: tuple[MockPaymentItem, ...]
    synthetic: bool = field(default=True)
    label: str = field(default=MOCKED_LABEL)


# In-memory mock gateway store, standing in for "the external PayPal
# system's own records" -- a real integration would be an HTTP API call;
# this is the mocked equivalent, kept process-local on purpose.
_BATCH_STORE: dict[str, MockPayoutBatch] = {}


def submit_payout_batch(
    action,
    batch_size: int,
    n_workers: int,
    reconciliation_close: float,
    initial_queue_depth: int,
) -> MockPayoutBatch:
    """Accept an approved proposal (an agent.demo_agent.AgentAction, or
    anything exposing a `.release_schedule`) and record a mocked payout
    submission. Generates a unique batch_id and per-item statuses."""
    batch_id = str(uuid.uuid4())
    items = tuple(MockPaymentItem(payment_id=i) for i in range(batch_size))
    batch = MockPayoutBatch(
        batch_id=batch_id,
        batch_size=batch_size,
        release_schedule=tuple(action.release_schedule),
        n_workers=n_workers,
        reconciliation_close=reconciliation_close,
        initial_queue_depth=initial_queue_depth,
        submitted_at=datetime.now(timezone.utc).isoformat(),
        status=STATUS_SUBMITTED,
        items=items,
    )
    _BATCH_STORE[batch_id] = batch
    return batch


def get_batch(batch_id: str) -> Optional[MockPayoutBatch]:
    """Retrieve a previously submitted mocked batch record, as a real
    execution service's "get batch" endpoint would."""
    return _BATCH_STORE.get(batch_id)


def get_batch_status(batch_id: str) -> Optional[str]:
    batch = _BATCH_STORE.get(batch_id)
    return batch.status if batch is not None else None


def get_payment_item_statuses(batch_id: str) -> Optional[tuple[MockPaymentItem, ...]]:
    batch = _BATCH_STORE.get(batch_id)
    return batch.items if batch is not None else None
