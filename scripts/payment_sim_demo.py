"""Phase 2 demo: drives the payment-simulation MCP tools through an independent MCP client.
Run:  python -m scripts.payment_sim_demo [--db PATH]   (SIMULATED data only; not PayPal)"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES = [{"base": "immediate_retry"}, {"base": "exponential_backoff"}, {"base": "intelligent_recovery"}]


def row(name, s, e):
    rr, cr = s["recovery_rate"], s["completion_rate"]
    return (f"{name:21s} recov={rr['value']:.3f} [{rr['ci95_wilson'][0]:.3f},{rr['ci95_wilson'][1]:.3f}]  compl={cr['value']:.4f}  "
            f"rev=${s['per_trial_means']['recovered_revenue']:>8,.0f}  t={s['avg_recovery_time_minutes']['value']:6.1f}m  "
            f"retries={s['per_trial_means']['total_retries']:>6.0f}  unnec={s['per_trial_means']['unnecessary_retries']:>6.0f}  "
            f"cost=${s['per_trial_means']['total_processing_cost']:>6.0f}  score={e['score_mean']:+.4f} feasible={e['feasible']}")


async def main(db: str | None) -> None:
    env = {**os.environ, **({"INOEMA_PAYMENT_DB": db} if db else {})}
    t = StdioTransport(command=sys.executable, args=["-m", "spikes.mcp_sim.server"], cwd=ROOT, env=env)
    async with Client(t, timeout=600) as c:
        call = lambda n, a: c.call_tool(n, a)  # noqa: E731
        for label, constraints in (("DEFAULT hard constraints (no terminal/buyer-class retries, idempotent, <=5 retries)", {}),
                                   ("NO hard constraints (pure score ranking)", {
                                       "max_retries": None, "max_recovery_window_minutes": None, "forbid_terminal_retries": False,
                                       "require_idempotency": False, "require_buyer_approval": False})):
            eid = (await call("create_payment_experiment", {"name": label[:40], "constraints": constraints})).structured_content["experiment_id"]
            scn = (await call("generate_payment_scenarios", {"experiment_id": eid, "scenarios": [
                {"preset": "baseline", "n_transactions": 10000}, {"preset": "network_instability", "n_transactions": 10000}]})).structured_content["scenarios"]
            print(f"\n=== {label} ===")
            for sc in scn:
                r = (await call("compare_recovery_policies", {"experiment_id": eid, "scenario_id": sc["scenario_id"],
                                                              "policies": POLICIES, "n_trials": 20, "master_seed": 42})).structured_content
                print(f"-- scenario {sc['name']} ({r['n_transactions']} txns x {r['n_trials']} trials, seed {r['master_seed']}) best={r['best_policy']}")
                for rk in r["ranking"]:
                    print("  ", row(rk["policy"], r["policies"][rk["policy"]]["summary"], rk))
                for n, d in r["paired_score_difference_vs_best"].items():
                    print(f"   best - {n}: {d['mean_difference']:+.4f} CI95 {[round(x, 4) for x in d['ci95']]}")
        first = r  # noqa: F841
        # reproducibility
        a = (await call("compare_recovery_policies", {"experiment_id": eid, "scenario_id": scn[0]["scenario_id"], "policies": POLICIES, "n_trials": 5, "master_seed": 7})).structured_content
        b = (await call("compare_recovery_policies", {"experiment_id": eid, "scenario_id": scn[0]["scenario_id"], "policies": POLICIES, "n_trials": 5, "master_seed": 7})).structured_content
        d = (await call("compare_recovery_policies", {"experiment_id": eid, "scenario_id": scn[0]["scenario_id"], "policies": POLICIES, "n_trials": 5, "master_seed": 8})).structured_content
        print(f"\nreproducible (seed 7 twice identical): {a['policies'] == b['policies']};  seed 8 differs: {a['policies'] != d['policies']}")
        tr = (await call("get_payment_transaction_trace", {"experiment_id": eid, "scenario_id": scn[0]["scenario_id"],
                                                           "policy": {"base": "intelligent_recovery"}, "master_seed": 42,
                                                           "outcome_filter": "RECOVERED"})).structured_content
        print(f"\nTRACE txn #{tr['transaction_index']} ({tr['transaction']['failure_category']}, ${tr['transaction']['amount']}):")
        for e in tr["events"]:
            print("  ", e)
        stored = (await call("get_payment_experiment_results", {"experiment_id": eid})).structured_content
        print(f"\nstored runs: {len(stored['runs'])}, provenance={stored['provenance']}, label={stored['label']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db")
    asyncio.run(main(ap.parse_args().db))
