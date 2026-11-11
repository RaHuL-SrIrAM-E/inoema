"""Genuine PayPal SANDBOX validation. Requires PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET (env or .env). SANDBOX ONLY.

  python -m scripts.paypal_sandbox_validation check                         # OAuth only
  python -m scripts.paypal_sandbox_validation negative                      # attempt documented mock codes (needs Negative Testing ON)
  python -m scripts.paypal_sandbox_validation run --reference REF [--report agent_report.json | --demo-policy] [--wait 600]
  python -m scripts.paypal_sandbox_validation resume --reference REF [--wait 600]
  python -m scripts.paypal_sandbox_validation coverage | evidence --reference REF

`run` creates a real Sandbox order and STOPS at buyer approval: open the printed URL in a browser, log in with a Sandbox
PERSONAL (buyer) account and approve. With --wait the script polls (GET order) until it can capture. Nothing is faked:
if credentials are missing it exits with an error.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from paypal.client import PayPalClient
from paypal.execution import ACTION_SUPPORT, ExecStatus, ExecutionPolicy, PaymentRequest, RecoveryExecutor
from paypal.models import PayPalApiError, PayPalConfig, PayPalConfigError, SandboxViolation
from paypal.negative_testing import coverage_report, run_negative_tests
from paypal.orders import OrdersApi
from paypal.persistence import SandboxStore
from spikes.envutil import load_env

DEMO_POLICY = {"name": "demo_intelligent_2_retries", "retry_limit": 2, "retry_intervals": [5.0], "exponential_backoff": True, "backoff_base_minutes": 5.0,
               "backoff_multiplier": 2.0, "backoff_max_interval_minutes": 240.0, "maximum_recovery_window": 4320.0, "buyer_action_timeout": 1440.0,
               "request_buyer_action": True, "alternate_funding_action": "REQUEST_ALTERNATE", "terminal_failure_handling": "FLAG_FOR_REVIEW",
               "classification_aware": True, "idempotent_retries": True}


def build(args):
    cfg = PayPalConfig.from_env()
    store = SandboxStore(args.db, secrets=cfg.secrets())
    client = PayPalClient(cfg, sink=store)
    return cfg, store, client


def load_policy(args) -> tuple[ExecutionPolicy, str]:
    if args.report:
        rep = json.load(open(args.report))
        return ExecutionPolicy.from_decision_report(rep), args.experiment_id or rep.get("execution_id", "unlinked-experiment")
    print("NOTE: using the built-in DEMO policy (not an agent-selected policy)")
    return ExecutionPolicy.from_definition(DEMO_POLICY, feasible=True, source="demo"), args.experiment_id or "demo-experiment"


def show(res) -> None:
    print(json.dumps({"execution_id": res.execution_id, "status": res.status.value, "order_id": res.order_id, "capture_id": res.capture_id,
                      "approval_url": res.approval_url, "reason": res.reason, "new_api_calls": res.new_api_calls,
                      "verification": (res.verification.ok if res.verification else None)}, indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check", "negative", "run", "resume", "coverage", "evidence", "actions"])
    ap.add_argument("--reference"), ap.add_argument("--amount", default="25.00"), ap.add_argument("--currency", default="USD")
    ap.add_argument("--report"), ap.add_argument("--experiment-id"), ap.add_argument("--demo-policy", action="store_true")
    ap.add_argument("--wait", type=int, default=0, help="seconds to poll for buyer approval")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)
    load_env()
    try:
        if args.cmd == "actions":
            print(json.dumps(ACTION_SUPPORT, indent=2))
            return 0
        cfg, store, client = build(args)
        if args.cmd == "coverage":
            print(json.dumps(coverage_report(store), indent=2))
            return 0
        if args.cmd == "check":
            client.tokens.get_token()
            print(f"OAuth OK against {client.base} (token not shown). Evidence rows: {len(store.api_calls())}")
            return 0
        if args.cmd == "negative":
            for r in run_negative_tests(OrdersApi(client), store):
                print(f"{r['mock_code']:<60} -> HTTP {r['observed_status']} {r['observed_name']} {r['observed_issues']} reproduced={r['reproduced']} debug_id={r['paypal_debug_id']}")
            print(json.dumps(coverage_report(store), indent=2))
            return 0
        policy, exp = load_policy(args)
        ex = RecoveryExecutor(client, store, policy, exp)
        if args.cmd == "evidence":
            row = store.get_execution_by_reference(exp, args.reference)
            print(json.dumps({"execution": row, "transitions": store.transitions(row["execution_id"]),
                              "api_calls": [{k: c[k] for k in ("operation", "status", "paypal_debug_id", "request_id", "ts")} for c in store.api_calls(row["execution_id"])],
                              "validation_records": store.validation_records(row["execution_id"])}, indent=2, default=str))
            return 0
        if not args.reference:
            ap.error("--reference is required")
        res = ex.execute(PaymentRequest(args.amount, args.currency, args.reference)) if args.cmd == "run" else ex.resume(args.reference)
        show(res)
        deadline = time.time() + args.wait
        while res.status == ExecStatus.AWAITING_BUYER and time.time() < deadline:
            print(f">>> BUYER ACTION REQUIRED (manual): open {res.approval_url} and approve with a Sandbox PERSONAL account. Polling...")
            time.sleep(10)
            res = ex.resume(args.reference)
        show(res)
        return 0 if res.status in (ExecStatus.COMPLETED, ExecStatus.AWAITING_BUYER) else 1
    except (PayPalConfigError, SandboxViolation) as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2
    except PayPalApiError as e:
        print(f"PAYPAL API ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
