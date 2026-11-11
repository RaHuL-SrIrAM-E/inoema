"""Database locations (read at call time so tests / deployments can redirect them with the same env vars the backends use)."""
from __future__ import annotations

import os

from paypal import persistence as _sandbox
from storage import payment_agent_runs as _agent
from storage import payment_experiments as _sim


def sim_db_path() -> str:
    return os.environ.get("INOEMA_PAYMENT_DB", _sim.DEFAULT_DB)


def agent_db_path() -> str:
    return os.environ.get("INOEMA_AGENT_DB", _agent.DEFAULT_DB)


def sandbox_db_path() -> str:
    return os.environ.get("INOEMA_PAYPAL_DB", _sandbox.DEFAULT_DB)
