# Policy Execution Bridge

Connects the optimization result (the agent's selected policy) to an explicit, auditable execution plan and a deterministic executor.
Offline and deterministic: **no Gemini, MCP or PayPal call is made by any of this code.**

```
persisted optimization run ──► gate ──► ExecutionPlan (versioned) ──► PolicyExecutor(state, failure event) ──► RETRY | WAIT | REQUIRE_BUYER_ACTION | STOP
   (storage, read-only)      (refuse)                                      │ every decision traced with source run, policy, hash, plan id, rule, rationale
                                                                           └─► paypal_gate: only a REAL, APPROVED, not-completed payment + explicit confirm + injected ResumeCapture
```

## What it does
* `execution_bridge/plan.py` – `load_persisted_result` (read-only), `validate_result` (gate), `compile_plan`. The gate refuses unless: a final report exists; the run is **COMPLETE**
  (genuine Gemini, finished by the model); the selected policy is feasible in the final deterministic evaluation; confidence is `HIGH`/`MEDIUM`; every objective constraint is one of the five
  supported ones (`max_retries`, `max_recovery_window_minutes`, `forbid_terminal_retries`, `require_idempotency`, `require_buyer_approval`); and the policy satisfies them.
  The compiled plan (`inoema-execution-plan/1`) holds retry limit, per-retry delays, recovery window, buyer-action requirement/timeout, per-category rules, constraints, the source run id, the policy hash and a deterministic plan id.
* `execution_bridge/executor.py` – pure `decide(state, event)`. Order: never touch a completed/captured payment → voided → pending buyer action (WAIT or timeout STOP) → terminal failures never retried →
  the engine's `next_action` for retry budget/window/buyer action → retries blocked without an idempotency key. `step()` also records a trace line.
* `execution_bridge/paypal_gate.py` – decides what may reach PayPal: simulated/read-only states never; completed payments never; `WAIT`/`REQUIRE_BUYER_ACTION` are read-only; a `RETRY` only for a `REAL_SANDBOX`, `APPROVED`, keyed payment with `confirm=True` and an injected `ResumeCapture.run`. The existing capture implementation is unchanged.
* `execution_bridge/store.py` – optional persistence to its own `data/execution_bridge.db` (plans + decisions). No existing evidence database is touched.
* Dashboard: **Execution Plan** section (read-only).

## Current data (honest status)
The only genuine Gemini run on disk is **interrupted (`llm_error`) with confidence `LOW`**, so the gate **refuses** it by default (the dashboard shows the failing checks).
A *demonstration override* waives **only** the completeness and confidence checks (never feasibility or constraints); the plan is then labelled `OVERRIDDEN_INCOMPLETE_SOURCE` and every warning is shown.
The selected policy in that run (`intel_ab_exp_b45_m2`) is real; its decisions below are produced by the deterministic executor from simulated payment states.

## Commands
```bash
.venv/bin/python -m scripts.execution_bridge_demo                           # gate only: shows the refusal for the interrupted run (exit 1)
.venv/bin/python -m scripts.execution_bridge_demo --allow-incomplete-source # labelled override: compile + run the 9 SIMULATED scenarios
.venv/bin/python -m scripts.execution_bridge_demo --allow-incomplete-source --persist   # also store the plan + trace in data/execution_bridge.db
python -m scripts.run_payments_dashboard --demo                             # then open the "Execution Plan" section
```

## What is NOT done
* Policy decisions are not wired into the live PayPal flow: the real Sandbox payment (`7P347414VM4362155`) was resumed manually and is shown only as protected from retry (`NEVER_RECAPTURE`). Nothing was retried or captured.
* No decisions were produced by live PayPal failures; failure events are simulated inputs we chose.
* A complete, medium/high-confidence genuine Gemini run is still needed for an un-overridden plan.
* Buyer notification is not sent (the plan only requires the buyer action); `terminal_failure_handling` flagging is recorded in the trace, not routed to a review queue.
* The Execution Plan page does not persist anything (use `--persist`).
