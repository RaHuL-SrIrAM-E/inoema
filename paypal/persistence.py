"""SQLite persistence for PayPal SANDBOX evidence, executions, transitions and validation records.

Separate database file (data/paypal_sandbox.db) and table family (`sandbox_*`); every row is pinned to
environment='PAYPAL_SANDBOX' by a CHECK constraint. Simulation tables (`payment_sim_*`, provenance 'SIMULATED')
live in other files, and nothing here aggregates across the two: validation_summary() returns raw OUTCOME COUNTS only,
never a rate that could be blended with simulated rates.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Optional

from paypal.client import redact
from paypal.models import ENVIRONMENT_LABEL

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "paypal_sandbox.db")
_ENV = "environment TEXT NOT NULL DEFAULT 'PAYPAL_SANDBOX' CHECK (environment = 'PAYPAL_SANDBOX')"

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS sandbox_executions (
    execution_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL, policy_id TEXT NOT NULL, policy_name TEXT NOT NULL,
    policy_json TEXT NOT NULL, reference_id TEXT NOT NULL, status TEXT NOT NULL, order_id TEXT, capture_id TEXT,
    approval_url TEXT, amount TEXT NOT NULL, currency TEXT NOT NULL, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    last_error_json TEXT, {_ENV}, UNIQUE (experiment_id, reference_id));
CREATE TABLE IF NOT EXISTS sandbox_api_calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id TEXT, ts TEXT NOT NULL, operation TEXT NOT NULL, method TEXT NOT NULL,
    url TEXT NOT NULL, request_id TEXT, request_headers TEXT, request_body TEXT, status INTEGER, response_headers TEXT,
    response_body TEXT, latency_ms REAL, paypal_debug_id TEXT, error TEXT, {_ENV});
CREATE TABLE IF NOT EXISTS sandbox_transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT, execution_id TEXT NOT NULL REFERENCES sandbox_executions(execution_id),
    seq INTEGER NOT NULL, ts TEXT NOT NULL, from_state TEXT, to_state TEXT NOT NULL, trigger TEXT NOT NULL, detail_json TEXT, {_ENV});
CREATE TABLE IF NOT EXISTS sandbox_validation_records (
    record_id TEXT PRIMARY KEY, execution_id TEXT NOT NULL REFERENCES sandbox_executions(execution_id), experiment_id TEXT NOT NULL,
    selected_policy_id TEXT NOT NULL, order_id TEXT, capture_id TEXT, api_operation TEXT NOT NULL, actual_api_status TEXT,
    observed_error_category TEXT, executed_at TEXT NOT NULL, predicted_action TEXT, actual_action TEXT NOT NULL,
    predicted_vs_observed_json TEXT NOT NULL, validation_outcome TEXT NOT NULL, {_ENV});
CREATE TABLE IF NOT EXISTS sandbox_negative_tests (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, mock_code TEXT NOT NULL, target_operation TEXT NOT NULL,
    observed_status INTEGER, observed_name TEXT, observed_issues TEXT, paypal_debug_id TEXT, reproduced INTEGER NOT NULL,
    detail_json TEXT, {_ENV});
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SandboxStore:
    def __init__(self, db_path: Optional[str] = None, secrets: Iterable[str] = ()):
        self.path = db_path or os.environ.get("INOEMA_PAYPAL_DB", DEFAULT_DB)
        if self.path != ":memory:":
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(_SCHEMA)
        self.secrets = tuple(s for s in secrets if s)

    def add_secrets(self, *secrets: str) -> None:
        self.secrets = tuple({*self.secrets, *(s for s in secrets if s)})

    def _j(self, obj: Any) -> str:
        return json.dumps(redact(obj, self.secrets), default=str)

    # --- EvidenceSink -------------------------------------------------------------------------------
    def record_api_call(self, rec: dict[str, Any]) -> None:
        meta = rec.get("meta") or {}
        self.conn.execute(
            "INSERT INTO sandbox_api_calls (execution_id,ts,operation,method,url,request_id,request_headers,request_body,status,"
            "response_headers,response_body,latency_ms,paypal_debug_id,error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (meta.get("execution_id"), _now(), rec["operation"], rec["method"], rec["url"], rec.get("request_id"),
             self._j(rec.get("request_headers")), self._j(rec.get("request_body")), rec.get("status"),
             self._j(rec.get("response_headers")), self._j(rec.get("response_body")), rec.get("latency_ms"),
             rec.get("paypal_debug_id"), rec.get("error")))
        self.conn.commit()

    # --- executions ---------------------------------------------------------------------------------
    def get_or_create_execution(self, experiment_id: str, reference_id: str, policy_id: str, policy_name: str, policy: dict,
                                amount: str, currency: str) -> tuple[dict, bool]:
        """Duplicate-execution prevention: one execution per (experiment_id, reference_id)."""
        row = self.get_execution_by_reference(experiment_id, reference_id)
        if row is not None:
            return row, False
        eid = str(uuid.uuid4())
        try:
            self.conn.execute("INSERT INTO sandbox_executions (execution_id,experiment_id,policy_id,policy_name,policy_json,reference_id,"
                              "status,amount,currency,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                              (eid, experiment_id, policy_id, policy_name, self._j(policy), reference_id, "NEW", amount, currency, _now(), _now()))
            self.conn.commit()
        except sqlite3.IntegrityError:  # lost a race: someone else created it
            return self.get_execution_by_reference(experiment_id, reference_id), False
        return self.get_execution(eid), True

    def get_execution(self, execution_id: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM sandbox_executions WHERE execution_id=?", (execution_id,)).fetchone()
        return dict(r) if r else None

    def get_execution_by_reference(self, experiment_id: str, reference_id: str) -> Optional[dict]:
        r = self.conn.execute("SELECT * FROM sandbox_executions WHERE experiment_id=? AND reference_id=?", (experiment_id, reference_id)).fetchone()
        return dict(r) if r else None

    def update_execution(self, execution_id: str, **fields: Any) -> None:
        allowed = {"status", "order_id", "capture_id", "approval_url", "last_error_json"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update {bad}")
        if "last_error_json" in fields and not isinstance(fields["last_error_json"], str):
            fields["last_error_json"] = self._j(fields["last_error_json"])
        sets = ", ".join(f"{k}=?" for k in fields) + (", " if fields else "") + "updated_at=?"
        self.conn.execute(f"UPDATE sandbox_executions SET {sets} WHERE execution_id=?", (*fields.values(), _now(), execution_id))
        self.conn.commit()

    # --- transitions / validation -------------------------------------------------------------------
    def add_transition(self, execution_id: str, from_state: Optional[str], to_state: str, trigger: str, detail: Optional[dict] = None) -> None:
        seq = (self.conn.execute("SELECT COALESCE(MAX(seq),0)+1 FROM sandbox_transitions WHERE execution_id=?", (execution_id,)).fetchone()[0])
        self.conn.execute("INSERT INTO sandbox_transitions (execution_id,seq,ts,from_state,to_state,trigger,detail_json) VALUES (?,?,?,?,?,?,?)",
                          (execution_id, seq, _now(), from_state, to_state, trigger, self._j(detail or {})))
        self.conn.commit()

    def add_validation_record(self, rec: dict[str, Any]) -> str:
        rid = str(uuid.uuid4())
        self.conn.execute(
            "INSERT INTO sandbox_validation_records (record_id,execution_id,experiment_id,selected_policy_id,order_id,capture_id,api_operation,"
            "actual_api_status,observed_error_category,executed_at,predicted_action,actual_action,predicted_vs_observed_json,validation_outcome)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rid, rec["execution_id"], rec["experiment_id"], rec["selected_policy_id"], rec.get("order_id"), rec.get("capture_id"),
             rec["api_operation"], rec.get("actual_api_status"), rec.get("observed_error_category"), rec.get("executed_at") or _now(),
             rec.get("predicted_action"), rec["actual_action"], self._j(rec["predicted_vs_observed"]), rec["validation_outcome"]))
        self.conn.commit()
        return rid

    def add_negative_test(self, rec: dict[str, Any]) -> None:
        self.conn.execute("INSERT INTO sandbox_negative_tests (ts,mock_code,target_operation,observed_status,observed_name,observed_issues,"
                          "paypal_debug_id,reproduced,detail_json) VALUES (?,?,?,?,?,?,?,?,?)",
                          (_now(), rec["mock_code"], rec["target_operation"], rec.get("observed_status"), rec.get("observed_name"),
                           self._j(rec.get("observed_issues", [])), rec.get("paypal_debug_id"), int(rec["reproduced"]), self._j(rec.get("detail", {}))))
        self.conn.commit()

    # --- reads --------------------------------------------------------------------------------------
    def _all(self, sql: str, *args: Any) -> list[dict]:
        return [dict(r) for r in self.conn.execute(sql, args)]

    def transitions(self, execution_id: str) -> list[dict]:
        return self._all("SELECT * FROM sandbox_transitions WHERE execution_id=? ORDER BY seq", execution_id)

    def api_calls(self, execution_id: Optional[str] = None) -> list[dict]:
        if execution_id:
            return self._all("SELECT * FROM sandbox_api_calls WHERE execution_id=? ORDER BY id", execution_id)
        return self._all("SELECT * FROM sandbox_api_calls ORDER BY id")

    def validation_records(self, execution_id: Optional[str] = None) -> list[dict]:
        if execution_id:
            return self._all("SELECT * FROM sandbox_validation_records WHERE execution_id=? ORDER BY rowid", execution_id)
        return self._all("SELECT * FROM sandbox_validation_records ORDER BY rowid")

    def negative_tests(self) -> list[dict]:
        return self._all("SELECT * FROM sandbox_negative_tests ORDER BY id")

    def validation_summary(self) -> dict[str, Any]:
        """RAW COUNTS of observed validation outcomes. Deliberately no rates: a handful of sandbox calls must never be
        presented alongside, or blended with, simulated population rates."""
        rows = self._all("SELECT validation_outcome, COUNT(*) AS n FROM sandbox_validation_records GROUP BY validation_outcome")
        return {"environment": ENVIRONMENT_LABEL, "outcome_counts": {r["validation_outcome"]: r["n"] for r in rows},
                "note": "counts of individual Sandbox observations; not comparable to simulated population rates"}

    def close(self) -> None:
        self.conn.close()
