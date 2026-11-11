"""Thin SimPy wrapper: a shared clock (minutes) on which transaction
lifecycle processes run, plus an optional per-transaction event log."""
from __future__ import annotations

from typing import Callable, Optional

import simpy

from engine.payments.models import EventType, TxnEvent


class EventLog:
    """Records TxnEvents per transaction index; a no-op unless enabled (keeps big runs cheap)."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.events: dict[int, list[TxnEvent]] = {}

    def record(self, idx: int, time_minutes: float, type_: EventType, **detail) -> None:
        if self.enabled:
            self.events.setdefault(idx, []).append(TxnEvent(time_minutes, type_, detail))


class EventScheduler:
    def __init__(self) -> None:
        self.env = simpy.Environment()

    @property
    def now(self) -> float:
        return self.env.now

    def spawn(self, process: Callable, *args) -> simpy.Process:
        return self.env.process(process(self.env, *args))

    def run(self, until: Optional[float] = None) -> None:
        self.env.run(until=until)
