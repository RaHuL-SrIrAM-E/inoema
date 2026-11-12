"""Load a persisted optimization result, gate it, and compile its selected policy into a versioned ExecutionPlan."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional

from engine.payments.failure_model import FailureRules
from engine.payments.models import FailureCategory
from engine.payments.recovery_policy import ActionKind, RecoveryPolicy, next_action
from paypal.execution import ExecutionPolicy, PolicyNotApproved
from storage import payment_agent_runs as _agent

PLAN_VERSION = "inoema-execution-plan/1"
ACCEPTABLE_CONFIDENCE = ("HIGH", "MEDIUM")  # LOW / INSUFFICIENT / missing are refused unless explicitly overridden
SUPPORTED_CONSTRAINTS = ("max_retries", "max_recovery_window_minutes", "forbid_terminal_retries", "require_idempotency", "require_buyer_approval")


class BridgeError(Exception):
    """Base class: the bridge refuses rather than guessing."""


class ResultNotFound(BridgeError):
    pass


class PlanRefused(BridgeError):
    def __init__(self, problems: list[str]):
        super().__init__("execution plan refused: " + "; ".join(problems))
        self.problems = problems


@dataclass(frozen=True)
class LoadedResult:
    execution_id: str
    provider: str
    status: str
    termination_reason: Optional[str]
    report: Optional[dict]
    db_path: str

    @property
    def is_complete(self) -> bool:  # same rule as the dashboard: genuine Gemini, finished BY the model
        return self.provider == "GEMINI_API" and self.status == "COMPLETED" and self.termination_reason == "finished_by_llm"


def load_persisted_result(execution_id: Optional[str] = None, db_path: Optional[str] = None) -> LoadedResult:
    """Read-only. With no id: the latest COMPLETE genuine Gemini run, else the latest genuine one (the gate will then refuse it)."""
    path = db_path or os.environ.get("INOEMA_AGENT_DB", _agent.DEFAULT_DB)
    if not os.path.exists(path):
        raise ResultNotFound(f"no optimization database at {path}")
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        if execution_id:
            rows = conn.execute("SELECT * FROM payment_agent_runs WHERE execution_id=?", (execution_id,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM payment_agent_runs WHERE llm_provider='GEMINI_API' ORDER BY started_at DESC").fetchall()
    finally:
        conn.close()
    if not rows:
        raise ResultNotFound(f"no optimization run {execution_id!r}" if execution_id else "no genuine Gemini optimization run is persisted")
    loaded = [LoadedResult(r["execution_id"], r["llm_provider"], r["status"], r["termination_reason"],
                           json.loads(r["final_report_json"]) if r["final_report_json"] else None, path) for r in rows]
    return next((x for x in loaded if x.is_complete), loaded[0])


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    waivable: bool = False  # only completeness / confidence may be waived, and only by an explicit override


@dataclass
class GateResult:
    checks: list[Check]
    override: bool = False

    @property
    def problems(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.ok]

    @property
    def blocking(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.ok and not (self.override and c.waivable)]

    @property
    def ok(self) -> bool:
        return not self.blocking

    @property
    def waived(self) -> list[str]:
        return [f"{c.name}: {c.detail}" for c in self.checks if not c.ok and self.override and c.waivable]


def validate_result(res: LoadedResult, *, allow_incomplete_source: bool = False) -> GateResult:
    checks: list[Check] = []
    rep = res.report or {}
    checks.append(Check("report_present", bool(res.report), "a final report is persisted" if res.report else "run has no persisted final report"))
    checks.append(Check("source_run_complete", res.is_complete,
                        "genuine Gemini run finished by the model" if res.is_complete
                        else f"run is not COMPLETE (provider {res.provider}, status {res.status}, termination {res.termination_reason})", waivable=True))
    sel = rep.get("selected_policy")
    checks.append(Check("selected_policy_present", bool(sel and sel.get("definition")), "selected policy found" if sel else "no selected policy in the report"))
    feasible = False
    if sel:
        row = next((r for r in (rep.get("final_evaluation") or {}).get("ranking", []) if r.get("policy") == sel.get("name")), None)
        feasible = bool(row and row.get("feasible"))
    checks.append(Check("selected_policy_feasible", feasible, "feasible in the final deterministic evaluation" if feasible else "selected policy is not feasible in the final evaluation"))
    label = (rep.get("confidence") or {}).get("label")
    checks.append(Check("confidence_acceptable", label in ACCEPTABLE_CONFIDENCE,
                        f"confidence {label}" if label in ACCEPTABLE_CONFIDENCE else f"confidence {label!r} is not in {list(ACCEPTABLE_CONFIDENCE)}", waivable=True))
    cons = (rep.get("merchant_objective") or {}).get("constraints")
    unknown = sorted(set(cons or {}) - set(SUPPORTED_CONSTRAINTS))
    missing = sorted(set(SUPPORTED_CONSTRAINTS) - set(cons or {}))
    checks.append(Check("constraints_supported", cons is not None and not unknown and not missing,
                        "all constraints are supported" if cons is not None and not unknown and not missing
                        else f"unsupported constraint(s) {unknown}" if unknown else f"constraint(s) missing from the report: {missing}" if cons is not None else "no constraints in the report"))
    if sel and sel.get("definition") and cons is not None and not unknown:
        d = sel["definition"]
        bad = []
        if d.get("retry_limit", 0) > cons.get("max_retries", 0):
            bad.append(f"retry_limit {d.get('retry_limit')} > max_retries {cons.get('max_retries')}")
        if d.get("maximum_recovery_window", 0) > cons.get("max_recovery_window_minutes", 0):
            bad.append("maximum_recovery_window exceeds the constraint")
        if cons.get("forbid_terminal_retries") and not d.get("classification_aware"):
            bad.append("policy is not classification-aware but terminal retries are forbidden")
        if cons.get("require_idempotency") and not d.get("idempotent_retries"):
            bad.append("policy retries are not idempotent but idempotency is required")
        checks.append(Check("policy_satisfies_constraints", not bad, "policy satisfies every constraint" if not bad else "; ".join(bad)))
    return GateResult(checks, override=allow_incomplete_source)


@dataclass(frozen=True)
class ExecutionPlan:
    plan_version: str
    plan_id: str
    source_run_id: str
    source_run_state: str
    source_confidence: Optional[str]
    policy_name: str
    policy_hash: str
    policy_definition: dict
    retry_limit: int
    retry_delays_minutes: tuple
    maximum_recovery_window_minutes: float
    buyer_action_required: bool
    buyer_action_timeout_minutes: float
    alternate_funding: str
    terminal_failure_handling: str
    constraints: dict
    failure_rules: dict
    status: str  # APPROVED | OVERRIDDEN_INCOMPLETE_SOURCE
    waived_checks: tuple = ()
    warnings: tuple = ()

    def engine_policy(self) -> RecoveryPolicy:
        return ExecutionPolicy.from_definition(self.policy_definition, feasible=True, source="plan").engine_policy

    def to_dict(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}


def compile_plan(res: LoadedResult, *, allow_incomplete_source: bool = False, rules: Optional[FailureRules] = None) -> ExecutionPlan:
    gate = validate_result(res, allow_incomplete_source=allow_incomplete_source)
    if not gate.ok:
        raise PlanRefused(gate.blocking)
    rep = res.report or {}
    try:
        xp = ExecutionPolicy.from_decision_report({**rep, "execution_id": res.execution_id})  # re-validates feasibility + policy fields
    except PolicyNotApproved as e:
        raise PlanRefused([str(e)]) from e
    pol, rules = xp.engine_policy, rules or FailureRules()
    cons = dict(rep["merchant_objective"]["constraints"])
    delays = tuple(round(pol.delay_for_retry(i), 4) for i in range(pol.retry_limit))
    frules = {}
    for cat in FailureCategory:
        a = next_action(pol, rules, cat, 0, 0.0, False)
        frules[cat.value] = {"recovery_class": rules.class_of(cat).value, "first_action": "STOP" if a.kind == ActionKind.TERMINATE else
                             "REQUIRE_BUYER_ACTION" if a.kind == ActionKind.REQUEST_BUYER_ACTION else "RETRY", "reason": a.reason, "first_delay_minutes": a.delay_minutes}
    overridden = bool(gate.waived)
    warnings = tuple(f"OVERRIDE (demonstration only): {w}" for w in gate.waived)
    core = {"v": PLAN_VERSION, "run": res.execution_id, "policy": xp.policy_id, "constraints": cons, "rules": frules, "delays": delays,
            "status": "OVERRIDDEN_INCOMPLETE_SOURCE" if overridden else "APPROVED"}
    plan_id = hashlib.sha256(json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return ExecutionPlan(PLAN_VERSION, plan_id, res.execution_id,
                         "COMPLETE" if res.is_complete else f"INTERRUPTED ({res.termination_reason})" if res.provider == "GEMINI_API" else f"NOT A GEMINI RUN ({res.provider})",
                         (rep.get("confidence") or {}).get("label"), pol.name, xp.policy_id, dict(xp.definition), pol.retry_limit, delays,
                         pol.maximum_recovery_window, pol.request_buyer_action, pol.buyer_action_timeout, pol.alternate_funding_action.value,
                         pol.terminal_failure_handling.value, cons, frules, core["status"], tuple(gate.waived), warnings)
