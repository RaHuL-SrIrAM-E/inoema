"""SQLite persistence of autonomous-agent executions (separate DB from simulation results and from any
PayPal Sandbox logs). Everything is passed through scrub() first: API keys/credentials are never stored."""
from __future__ import annotations

import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "payment_agent_runs.db")
_SECRET_KEY = re.compile(r"(api[_-]?key|authorization|secret|password|credential|bearer)", re.I)
REDACTED = "***REDACTED***"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS payment_agent_runs (
    execution_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT, status TEXT NOT NULL,
    llm_provider TEXT NOT NULL, model TEXT NOT NULL, config_json TEXT NOT NULL, objective_text TEXT NOT NULL,
    objective_json TEXT NOT NULL, experiment_id TEXT, termination_reason TEXT, final_report_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS payment_agent_tool_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id TEXT NOT NULL REFERENCES payment_agent_runs(execution_id),
    seq INTEGER NOT NULL, iteration INTEGER NOT NULL, source TEXT NOT NULL, tool TEXT NOT NULL, args_json TEXT NOT NULL,
    ok INTEGER NOT NULL, result_json TEXT, duration_ms REAL, work_units INTEGER, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS payment_agent_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id TEXT NOT NULL REFERENCES payment_agent_runs(execution_id),
    seq INTEGER NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS payment_agent_candidates (
    execution_id TEXT NOT NULL REFERENCES payment_agent_runs(execution_id), name TEXT NOT NULL, origin TEXT NOT NULL,
    definition_json TEXT NOT NULL, observations_json TEXT NOT NULL, PRIMARY KEY (execution_id, name));
"""


def scrub(value: Any, secrets: Iterable[str] = ()) -> Any:
    """Recursively redact secret-named keys and any literal occurrence of the given secret strings."""
    secrets = [s for s in secrets if s]
    if isinstance(value, dict):
        return {k: (REDACTED if isinstance(k, str) and _SECRET_KEY.search(k) else scrub(v, secrets)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v, secrets) for v in value]
    if isinstance(value, str):
        for s in secrets:
            value = value.replace(s, REDACTED)
        return value
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AgentRunStore:
    def __init__(self, db_path: Optional[str] = None, secrets: Iterable[str] = ()):
        self.path = db_path or os.environ.get("INOEMA_AGENT_DB", DEFAULT_DB)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_SCHEMA)
        self.secrets = [s for s in secrets if s]
        self._event_seq = 0

    def _j(self, obj: Any) -> str:
        return json.dumps(scrub(obj, self.secrets), default=str)

    def start_run(self, llm_provider: str, model: str, config: dict, objective_text: str, objective: dict) -> str:
        eid = str(uuid.uuid4())
        self.conn.execute("INSERT INTO payment_agent_runs (execution_id,started_at,status,llm_provider,model,config_json,objective_text,"
                          "objective_json) VALUES (?,?,?,?,?,?,?,?)",
                          (eid, _now(), "RUNNING", llm_provider, model, self._j(config), self._j(objective_text), self._j(objective)))
        self.conn.commit()
        return eid

    def add_tool_call(self, execution_id: str, rec: dict) -> None:
        self.conn.execute("INSERT INTO payment_agent_tool_calls (execution_id,seq,iteration,source,tool,args_json,ok,result_json,duration_ms,"
                          "work_units,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                          (execution_id, rec["seq"], rec["iteration"], rec["source"], rec["tool"], self._j(rec["args"]), int(rec["ok"]),
                           self._j(rec.get("result")), rec["duration_ms"], rec.get("work_units", 0), _now()))
        self.conn.commit()

    def add_event(self, execution_id: str, kind: str, payload: Any) -> None:
        self._event_seq += 1
        self.conn.execute("INSERT INTO payment_agent_events (execution_id,seq,kind,payload_json,created_at) VALUES (?,?,?,?,?)",
                          (execution_id, self._event_seq, kind, self._j(payload), _now()))
        self.conn.commit()

    def save_candidates(self, execution_id: str, candidates: list[dict]) -> None:
        for c in candidates:
            self.conn.execute("INSERT OR REPLACE INTO payment_agent_candidates VALUES (?,?,?,?,?)",
                              (execution_id, c["name"], c["origin"], self._j(c["definition"]), self._j(c["observations"])))
        self.conn.commit()

    def set_experiment(self, execution_id: str, experiment_id: str) -> None:
        self.conn.execute("UPDATE payment_agent_runs SET experiment_id=? WHERE execution_id=?", (experiment_id, execution_id))
        self.conn.commit()

    def finish_run(self, execution_id: str, status: str, termination_reason: Optional[str], report: Optional[dict], error: Optional[str] = None) -> None:
        self.conn.execute("UPDATE payment_agent_runs SET finished_at=?, status=?, termination_reason=?, final_report_json=?, error=? WHERE execution_id=?",
                          (_now(), status, termination_reason, self._j(report) if report is not None else None,
                           self._j(error) if error else None, execution_id))
        self.conn.commit()

    # --- reads (tests / demo) ---
    def get_run(self, execution_id: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM payment_agent_runs WHERE execution_id=?", (execution_id,)).fetchone()
        return None if r is None else dict(r)

    def tool_calls(self, execution_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM payment_agent_tool_calls WHERE execution_id=? ORDER BY seq", (execution_id,))]

    def events(self, execution_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM payment_agent_events WHERE execution_id=? ORDER BY seq", (execution_id,))]

    def candidates(self, execution_id: str) -> list[dict]:
        return [dict(r) for r in self.conn.execute("SELECT * FROM payment_agent_candidates WHERE execution_id=? ORDER BY name", (execution_id,))]

    def close(self) -> None:
        self.conn.close()
