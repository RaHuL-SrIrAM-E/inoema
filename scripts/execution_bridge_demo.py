"""Compile the persisted optimization result into an execution plan and run the SIMULATED decision scenarios through the deterministic executor.

  python -m scripts.execution_bridge_demo                         # gate the latest genuine run (refuses an interrupted / low-confidence run)
  python -m scripts.execution_bridge_demo --allow-incomplete-source   # DEMONSTRATION OVERRIDE, loudly labelled in the plan and every trace line
  python -m scripts.execution_bridge_demo --persist               # also store plan + trace in data/execution_bridge.db

No Gemini, MCP or PayPal call is made. States are simulated; the real Sandbox payment is never retried or captured.
"""
from __future__ import annotations

import argparse
import sys

from execution_bridge.executor import PolicyExecutor
from execution_bridge.plan import BridgeError, PlanRefused, compile_plan, load_persisted_result, validate_result
from execution_bridge.scenarios import SCENARIOS
from execution_bridge.store import BridgeStore


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-id"), ap.add_argument("--agent-db"), ap.add_argument("--bridge-db")
    ap.add_argument("--allow-incomplete-source", action="store_true"), ap.add_argument("--persist", action="store_true")
    a = ap.parse_args(argv)
    try:
        res = load_persisted_result(a.run_id, a.agent_db)
        gate = validate_result(res, allow_incomplete_source=a.allow_incomplete_source)
        for c in gate.checks:
            print(f"  [{'PASS' if c.ok else 'WAIVED' if (gate.override and c.waivable) else 'FAIL'}] {c.name}: {c.detail}")
        plan = compile_plan(res, allow_incomplete_source=a.allow_incomplete_source)
    except PlanRefused as e:
        print(f"REFUSED: {e}")
        return 1
    except BridgeError as e:
        print(f"ERROR: {e}")
        return 2
    print(f"PLAN {plan.plan_id} ({plan.plan_version}) status={plan.status} policy={plan.policy_name} [{plan.policy_hash}] source_run={plan.source_run_id[:8]} ({plan.source_run_state})")
    for w in plan.warnings:
        print("  WARNING:", w)
    ex = PolicyExecutor(plan)
    for name, state, event in SCENARIOS:
        d = ex.step(state, event)
        print(f"  SIMULATED · {name:<48} -> {d.action.value:<21} {d.rule}" + (f" (+{d.delay_minutes:g} min)" if d.delay_minutes else ""))
    if a.persist:
        n = BridgeStore(a.bridge_db).save(plan, "simulated-demo", ex.trace)
        print(f"persisted plan + {n} decisions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
