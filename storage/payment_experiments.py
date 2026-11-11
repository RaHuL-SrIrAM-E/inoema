"""SQLite persistence for SIMULATED payment-recovery experiments.

Deliberately a separate database file and table family (`payment_sim_*`,
provenance column fixed to 'SIMULATED') so simulation outputs can never be
mixed up with PayPal Sandbox observations, which will live elsewhere
(`sandbox_*` tables / spikes/paypal log). Primitive-typed like storage/database.py.
"""
from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "payment_simulations.db")
PROVENANCE = "SIMULATED"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS payment_sim_experiments (
    experiment_id TEXT PRIMARY KEY, name TEXT NOT NULL, config_json TEXT NOT NULL,
    created_at TEXT NOT NULL, provenance TEXT NOT NULL DEFAULT 'SIMULATED' CHECK (provenance = 'SIMULATED'));
CREATE TABLE IF NOT EXISTS payment_sim_scenarios (
    scenario_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES payment_sim_experiments(experiment_id),
    name TEXT NOT NULL, spec_json TEXT NOT NULL, created_at TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT 'SIMULATED' CHECK (provenance = 'SIMULATED'));
CREATE TABLE IF NOT EXISTS payment_sim_runs (
    run_id TEXT PRIMARY KEY, experiment_id TEXT NOT NULL REFERENCES payment_sim_experiments(experiment_id),
    scenario_id TEXT NOT NULL REFERENCES payment_sim_scenarios(scenario_id),
    kind TEXT NOT NULL CHECK (kind IN ('run','comparison')), master_seed INTEGER NOT NULL, n_trials INTEGER NOT NULL,
    request_json TEXT NOT NULL, result_json TEXT NOT NULL, created_at TEXT NOT NULL,
    provenance TEXT NOT NULL DEFAULT 'SIMULATED' CHECK (provenance = 'SIMULATED'));
"""


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    db_path = db_path or os.environ.get("INOEMA_PAYMENT_DB", DEFAULT_DB)
    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_experiment(conn, name: str, config: dict) -> str:
    eid = str(uuid.uuid4())
    conn.execute("INSERT INTO payment_sim_experiments (experiment_id,name,config_json,created_at) VALUES (?,?,?,?)",
                 (eid, name, json.dumps(config), _now()))
    conn.commit()
    return eid


def get_experiment(conn, experiment_id: str) -> Optional[dict]:
    r = conn.execute("SELECT * FROM payment_sim_experiments WHERE experiment_id=?", (experiment_id,)).fetchone()
    return None if r is None else {"experiment_id": r["experiment_id"], "name": r["name"],
                                   "config": json.loads(r["config_json"]), "created_at": r["created_at"]}


def add_scenario(conn, experiment_id: str, name: str, spec: dict) -> str:
    sid = str(uuid.uuid4())
    conn.execute("INSERT INTO payment_sim_scenarios (scenario_id,experiment_id,name,spec_json,created_at) VALUES (?,?,?,?,?)",
                 (sid, experiment_id, name, json.dumps(spec), _now()))
    conn.commit()
    return sid


def get_scenario(conn, experiment_id: str, scenario_id: str) -> Optional[dict]:
    r = conn.execute("SELECT * FROM payment_sim_scenarios WHERE scenario_id=? AND experiment_id=?",
                     (scenario_id, experiment_id)).fetchone()
    return None if r is None else {"scenario_id": r["scenario_id"], "name": r["name"], "spec": json.loads(r["spec_json"])}


def list_scenarios(conn, experiment_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM payment_sim_scenarios WHERE experiment_id=? ORDER BY created_at, rowid",
                        (experiment_id,)).fetchall()
    return [{"scenario_id": r["scenario_id"], "name": r["name"], "spec": json.loads(r["spec_json"])} for r in rows]


def add_run(conn, experiment_id: str, scenario_id: str, kind: str, master_seed: int, n_trials: int,
            request: dict, result: dict) -> str:
    rid = str(uuid.uuid4())
    conn.execute("INSERT INTO payment_sim_runs (run_id,experiment_id,scenario_id,kind,master_seed,n_trials,"
                 "request_json,result_json,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                 (rid, experiment_id, scenario_id, kind, master_seed, n_trials, json.dumps(request),
                  json.dumps(result), _now()))
    conn.commit()
    return rid


def list_runs(conn, experiment_id: str) -> list[dict]:
    rows = conn.execute("SELECT * FROM payment_sim_runs WHERE experiment_id=? ORDER BY created_at, rowid",
                        (experiment_id,)).fetchall()
    return [{"run_id": r["run_id"], "scenario_id": r["scenario_id"], "kind": r["kind"],
             "master_seed": r["master_seed"], "n_trials": r["n_trials"], "created_at": r["created_at"],
             "request": json.loads(r["request_json"]), "result": json.loads(r["result_json"])} for r in rows]
