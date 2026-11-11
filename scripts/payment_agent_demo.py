"""Autonomous payment-recovery optimisation demo.

  python -m scripts.payment_agent_demo                 # GENUINE run: needs GEMINI_API_KEY (reads .env)
  python -m scripts.payment_agent_demo --scripted      # pipeline demo with a SCRIPTED TEST DOUBLE -- NOT Gemini

Flow: merchant objective -> Gemini plans/chooses tools -> real MCP simulation calls -> multiple adaptive iterations ->
deterministic evaluator re-runs the shortlist -> sensitivity analysis -> deterministic selection -> decision report.
SIMULATED data under ASSUMED distributions; no PayPal calls are made.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

from agent.payments.agent import PaymentAgent
from agent.payments.models import AgentConfig, MissingCredentialsError
from agent.payments.testing import ScriptedBackend, last_function_responses
from spikes.envutil import load_env

DEFAULT_OBJECTIVE = ("Optimize payment recovery for a merchant processing 10,000 transactions. Maximize recovered revenue while "
                     "minimizing recovery cost and latency. Do not exceed two automatic retries. Never automatically retry terminal failures.")


def scripted_turns():
    """A FIXED sequence (non-autonomous). Only used by --scripted; it adapts its 2nd/3rd step to earlier results to exercise the loop."""
    def explore(contents):
        r = last_function_responses(contents)[0]["result"]
        best = [x for x in r["ranking"] if x["feasible"]][0]["policy"]
        return [("propose_policy_variations", {"policy_name": best, "dimensions": ["backoff_base_minutes", "buyer_action_timeout", "maximum_recovery_window"]})]

    def test_variants(contents):
        v = last_function_responses(contents)[0]["result"]["variants"]
        return [("compare_recovery_policies", {"policies": [{k: x[k] for k in x if k != "change"} for x in v[:4]], "n_trials": 8, "master_seed": 0})]

    return [[("compare_recovery_policies", {"policies": [{"base": "intelligent_recovery"},
                                                         {"name": "intel_2_retries", "base": "intelligent_recovery", "retry_limit": 2}],
                                            "n_trials": 8, "master_seed": 0})],
            explore, test_variants,
            [("run_sensitivity_analysis", {"policies": [{"base": "intelligent_recovery", "name": "intel_5"}, {"name": "intel_2_retries", "base": "intelligent_recovery", "retry_limit": 2}],
                                           "parameters": [{"parameter": "buyer_response_probability"}, {"parameter": "network_instability"}], "n_trials": 4})],
            [("finish_optimization", {"shortlist": ["intel_2_retries"], "rationale": "scripted"})]]


async def main(args: argparse.Namespace) -> int:
    load_env()
    over = dict(max_iterations=args.max_iterations, exploration_trials=args.trials, final_trials=args.final_trials)
    if args.scripted:
        print("=" * 78 + "\nLLM MODE: SCRIPTED TEST DOUBLE -- THIS IS *NOT* GEMINI. No LLM is involved in decisions below.\n" + "=" * 78)
        cfg = AgentConfig(api_key="", model=args.model or "n/a", **over)
        agent = PaymentAgent(cfg, backend=ScriptedBackend(scripted_turns()), progress=print)
    else:
        try:
            cfg = AgentConfig.from_env(**({"model": args.model} if args.model else {}), **over)
        except MissingCredentialsError as e:
            print(f"ERROR: {e}\nSet GEMINI_API_KEY in .env (see .env.example), or run with --scripted for a clearly-labelled non-Gemini pipeline demo.", file=sys.stderr)
            return 2
        print("=" * 78 + f"\nLLM MODE: GENUINE GEMINI API  model={cfg.model}\n" + "=" * 78)
        agent = PaymentAgent(cfg, progress=print)
    print(f"\nMERCHANT OBJECTIVE: {args.objective}\n")
    report = await agent.run(args.objective)
    print("\n" + report.to_markdown())
    if args.json_out:
        json.dump(report.to_dict(), open(args.json_out, "w"), indent=2, default=str)
        print(f"\n(full JSON report written to {args.json_out}; execution_id={report.execution_id}; agent DB: {agent.store.path})")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scripted", action="store_true", help="use a scripted test double instead of Gemini (NOT a Gemini run)")
    ap.add_argument("--objective", default=DEFAULT_OBJECTIVE)
    ap.add_argument("--model", default=None, help="override GEMINI_MODEL")
    ap.add_argument("--max-iterations", type=int, default=10)
    ap.add_argument("--trials", type=int, default=10, help="Monte Carlo trials per exploration experiment")
    ap.add_argument("--final-trials", type=int, default=20)
    ap.add_argument("--json-out", default=None)
    sys.exit(asyncio.run(main(ap.parse_args())))
