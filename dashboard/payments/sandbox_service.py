"""Thin service layer between the Sandbox page and paypal.execution. It adds NO payment logic: every decision (state checks,
idempotency, no capture before approval, retry rules) is made by paypal.execution.RecoveryExecutor."""
from __future__ import annotations

import re
import uuid
from typing import Any, Callable, Optional

from dashboard.payments import data, paths
from paypal.client import PayPalClient
from paypal.execution import DEMO_POLICY, ExecStatus, ExecutionPolicy, ExecutionResult, PaymentRequest, PolicyNotApproved, RecoveryExecutor
from paypal.models import PayPalConfig
from paypal.persistence import SandboxStore

# Tests replace this to inject a mock HTTP transport. Production always builds the real Sandbox client.
ClientFactory = Callable[[PayPalConfig, SandboxStore], PayPalClient]


def _default_factory(cfg: PayPalConfig, store: SandboxStore) -> PayPalClient:
    return PayPalClient(cfg, sink=store)


client_factory: ClientFactory = _default_factory


def make_client(cfg: PayPalConfig, store: SandboxStore) -> PayPalClient:
    return client_factory(cfg, store)


def credentials_configured() -> bool:
    try:
        PayPalConfig.from_env()
        return True
    except Exception:  # noqa: BLE001  (missing credentials or a Sandbox violation: either way, not usable)
        return False


def available_policies() -> list[dict[str, Any]]:
    """Policies the Sandbox page may execute. Each entry states where it came from. The agent-selected policy is taken from the latest
    COMPLETE genuine execution when one exists; otherwise from the latest genuine execution, flagged as not complete."""
    out: list[dict[str, Any]] = []
    last = data.latest_genuine_agent_run()
    if last is not None:
        detail = data.get_agent_run(last["execution_id"])
        try:
            pol = ExecutionPolicy.from_decision_report({**(detail["run"]["report"] or {}), "execution_id": last["execution_id"]})
            flag = "" if last["complete"] else f" — ⚠ from an {last['state']} run"
            out.append({"key": f"agent:{last['execution_id']}", "label": f"Agent-selected: {pol.name} (execution {last['execution_id'][:8]}, {last['model']}){flag}",
                        "origin": "AGENT_SELECTED", "policy": pol, "experiment_id": last["execution_id"], "complete": last["complete"], "state": last["state"]})
        except PolicyNotApproved:
            pass
    out.append({"key": "demo", "label": f"Built-in DEMO policy: {DEMO_POLICY['name']} (not agent-selected)", "origin": "DEMO",
                "policy": ExecutionPolicy.from_definition(DEMO_POLICY, feasible=True, source="dashboard-demo"), "experiment_id": "dashboard-demo"})
    return out


def new_reference_id() -> str:
    return f"dash-{uuid.uuid4().hex[:10]}"


_AMOUNT_OK = re.compile(r"^\d{1,5}(\.\d{1,2})?$")


def _executor(policy: ExecutionPolicy, experiment_id: str, sleep: Optional[Callable[[float], None]] = None) -> RecoveryExecutor:
    cfg = PayPalConfig.from_env()  # raises if credentials are missing / mode is not sandbox
    store = SandboxStore(paths.sandbox_db_path(), secrets=cfg.secrets())
    kw = {"sleep": sleep} if sleep else {}
    return RecoveryExecutor(make_client(cfg, store), store, policy, experiment_id, **kw)


def create_order(policy: ExecutionPolicy, experiment_id: str, amount: str, currency: str, reference_id: str) -> ExecutionResult:
    """Explicit user action: create a real Sandbox order and STOP at buyer approval (the executor never captures an unapproved order)."""
    if not _AMOUNT_OK.match(str(amount)):
        raise ValueError("amount must look like 25 or 25.00")
    return _executor(policy, experiment_id).execute(PaymentRequest(amount, currency, reference_id))


def check_and_capture(policy: ExecutionPolicy, experiment_id: str, reference_id: str) -> ExecutionResult:
    """Explicit user action: re-inspect the REAL order state; capture happens only if PayPal reports it APPROVED."""
    return _executor(policy, experiment_id).resume(reference_id)


def status_is_awaiting_buyer(status: str) -> bool:
    return status == ExecStatus.AWAITING_BUYER.value
