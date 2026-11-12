"""Optional persistence of plans and decision traces in their OWN database (data/execution_bridge.db); never touches existing evidence DBs."""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Iterable

from execution_bridge.executor import TraceStep
from execution_bridge.plan import ExecutionPlan

DEFAULT_DB = os.path.join("data", "execution_bridge.db")
_SCHEMA = """
CREATE TABLE IF NOT EXISTS bridge_plans (plan_id TEXT PRIMARY KEY, plan_version TEXT NOT NULL, source_run_id TEXT NOT NULL, policy_name TEXT NOT NULL,
  policy_hash TEXT NOT NULL, status TEXT NOT NULL, plan_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS bridge_decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, plan_id TEXT NOT NULL REFERENCES bridge_plans(plan_id), scenario TEXT NOT NULL,
  step INTEGER NOT NULL, state_source TEXT NOT NULL, step_json TEXT NOT NULL, created_at TEXT NOT NULL);
"""


class BridgeStore:
    def __init__(self, path: str | None = None):
        self.path = path or os.environ.get("INOEMA_BRIDGE_DB", DEFAULT_DB)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)

    def save(self, plan: ExecutionPlan, scenario: str, steps: Iterable[TraceStep]) -> int:
        now = datetime.now(timezone.utc).isoformat()
        self.conn.execute("INSERT OR IGNORE INTO bridge_plans VALUES (?,?,?,?,?,?,?,?)", (plan.plan_id, plan.plan_version, plan.source_run_id, plan.policy_name,
                                                                                         plan.policy_hash, plan.status, json.dumps(plan.to_dict(), default=str), now))
        n = 0
        for s in steps:
            self.conn.execute("INSERT INTO bridge_decisions (plan_id,scenario,step,state_source,step_json,created_at) VALUES (?,?,?,?,?,?)",
                              (plan.plan_id, scenario, s.index, s.state_source, json.dumps(s.__dict__, default=str), now))
            n += 1
        self.conn.commit()
        return n

    def decisions(self, plan_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM bridge_decisions WHERE plan_id=? ORDER BY id", (plan_id,))]
