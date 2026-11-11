# INOEMA Payments Dashboard

The Payments workspace extends the existing INOEMA Streamlit app (`dashboard/app.py`) and is now its **default workspace**. It reuses the shipped identity (palette, CSS, pills,
kickers, Plotly layout from `dashboard/components.py`); it adds no new design system and no new dependencies. The original *Simulation
Control Center* is unchanged and one click away (sidebar → Workspace), or open it directly with `?workspace=simulation` / `INOEMA_DEFAULT_WORKSPACE=simulation`.
(There is no particle component in this repository's Streamlit app, so none is reused.) See `docs/FINAL_DASHBOARD_REVIEW.md` for the Phase 5.1 review.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env            # then fill in only what you need
python -m scripts.run_payments_dashboard          # opens in the Payments workspace (the default)
python -m scripts.run_payments_dashboard --demo   # ...in Demonstration mode
streamlit run dashboard/app.py                    # same default; the original app: sidebar -> "Simulation Control Center" (or ?workspace=simulation)
```

| Configuration | Purpose |
|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `GEMINI_MODEL` | required to start the optimisation agent; set the model explicitly (the default constant may be stale for new keys) |
| `PAYPAL_CLIENT_ID`, `PAYPAL_CLIENT_SECRET` | PayPal **Sandbox** REST app credentials; Sandbox-only guard is always active |
| `INOEMA_PAYMENT_DB`, `INOEMA_AGENT_DB`, `INOEMA_PAYPAL_DB` | override the three SQLite files (defaults in `data/`) |
| `INOEMA_DEFAULT_WORKSPACE=simulation`, `INOEMA_DEMO_MODE=1`, `INOEMA_DASHBOARD_AUTOCHECK=0` | startup behaviour (see `.env.example`); Payments is the default workspace |

## Three data sources, never mixed

Every metric carries exactly one label, and the code refuses to render a metric group that mixes sources.

| Label | Meaning | Store |
|---|---|---|
| ◆ **SIMULATED** | simulation-engine output under ASSUMED distributions (not PayPal, not real-world performance) | `payment_sim_*` (`data/payment_simulations.db`) |
| ✦ **AGENT-DERIVED** | chosen/explained by the Gemini agent and deterministic evaluator, from simulated results | `payment_agent_*` (`data/payment_agent_runs.db`) |
| ● **SANDBOX · ACTUAL** | real PayPal Sandbox API observations (single observations, no rates) | `sandbox_*` (`data/paypal_sandbox.db`) |
| ○ **SYSTEM** | live status of a local component | — |

## Navigation (sidebar *Workspace* → *INOEMA Payments*; sections along the top)

1. **Overview** — component status (simulation engine: runs a real 300-txn Monte Carlo; MCP server: starts it and discovers tools via an independent client; Gemini and PayPal: configuration shown, *not* probed automatically — explicit "Test" buttons spend one API/OAuth call), latest experiment, selected policy, feasibility, recovery rate (95% CI), recovered revenue, average recovery latency, experiment count, and Sandbox counters — in three separately-labelled groups.
2. **Policy Optimization** — merchant form: transactions, workload scenario, order-value distribution (lognormal median/sigma), maximum retries, recovery window, latency / cost / unnecessary-attempt weights, hard constraints, free-text guidance. A live preview shows the constraints the evaluator will enforce (built by the agent's own parser; unrecognised constraint phrases are warned as *not enforced*). After an explicit confirmation, **Start Gemini optimization agent** runs the real `PaymentAgent` in a background thread (MCP server as a subprocess); the page polls every 2 s and shows the live tool-call trace and progress log, then the full decision report (candidates, simulation results, sensitivity, deterministic selection, rejected alternatives, confidence, Gemini's explanation). One job at a time.
3. **Experiments** — pick an experiment and a stored simulation run (each is labelled with who requested it: harness, a Gemini turn, or the deterministic final evaluation). Table columns: policy, feasibility, objective score (+CI), recovery rate (+95% Wilson CI), recovered revenue, average recovery time, retry overhead, unnecessary retries, processing cost, cost per success, buyer actions, violations. Real filters (feasibility, minimum recovery rate, maximum latency, name search), sort selector, native column-header sorting, CSV export. Charts: recovery rate with CIs, objective scores, revenue-vs-latency trade-off. *Sensitivity* tab: ranked parameters, winner-at-every-point heatmap, score-vs-parameter lines, recorded assumptions. (AG Grid is deliberately **not** used: Streamlit's native dataframe provides sorting and the page provides the filters, without an extra dependency.)
4. **PayPal Sandbox** — Sandbox-only workflow behind explicit actions: *Create Sandbox order* (checkbox confirmation) → stops at **buyer approval** and shows the approval URL → *Check order state and capture* (unlocked only while the execution awaits the buyer, plus a second confirmation; the executor itself re-reads the real order and captures only if PayPal reports it approved) → verification. Shows order ID, capture ID, PayPal debug IDs, state transitions, API calls (secrets redacted) and predicted-vs-observed records. Without credentials every action is disabled; nothing is simulated or faked.
5. **Agent Execution Traces** — per execution: ID, provider, configured model, **served model** (recorded per turn; older runs show "not recorded"), termination, objective and enforced constraints, an expandable iteration history (what Gemini said and requested, each MCP tool call in plain English with its sanitised input/output, work units, simulation run IDs), candidate policies with their parameter changes vs the initial policy and vs their parent, and the final evaluator decision. Interrupted runs and non-Gemini runs are flagged explicitly.

## Demonstration mode (sidebar toggle)

Loads previously persisted, **genuine** results and labels them HISTORICAL (with execution ID, model and timestamp). Live agent and Sandbox actions are disabled. Only executions whose provider is `GEMINI_API` are used; a scripted double can never appear. It opens on the latest **COMPLETE** genuine execution (provider `GEMINI_API`, finished normally, and the model itself called `finish_optimization`) even if a newer interrupted one exists. If no complete execution exists it shows the latest genuine one under a prominent **INTERRUPTED / INCOMPLETE** banner, never as a completed run. Quota exhaustion or any API error during a live run is shown with advice and a button to switch to this mode — the dashboard never substitutes a scripted or different model.

## Errors handled

Missing credentials, quota exhaustion (with the API's suggested wait), unavailable model, rejected key, timeouts, MCP server problems, concurrent jobs, invalid form values, invalid Sandbox amounts, PayPal API errors (HTTP status + debug ID). See `dashboard/payments/errors.py`.

## Security

Keys are never stored or displayed: persisted data is scrubbed on write, and the dashboard scrubs again on read (secret-named keys, literal environment secrets, common credential shapes) and truncates long values. Job records hold a redacted config only.

## Architecture

`dashboard/payments/` — `data.py` (read-only access to the three stores; opens SQLite read-only and never creates files), `status.py`, `runner.py` (background agent jobs), `sandbox_service.py` (thin wrapper over `paypal.execution`), `errors.py`, `objective.py`, `redaction.py`, `sources.py` are Streamlit-free; `ui.py`, `charts.py`, `reports.py`, `page_*.py`, `workspace.py` render. The UI contains no simulation, scoring, agent or PayPal decision logic (a test enforces this). Backend additions for this phase are small and additive: optional `amount_median`/`amount_sigma` on scenario construction and the MCP `generate_payment_scenarios` tool, `scenario_preset`/amount fields on the agent's `ObjectiveSpec`, and a shared `DEMO_POLICY` constant in `paypal/execution.py`.

## Complete demonstration sequence

1. `python -m scripts.run_payments_dashboard --demo` → **Overview**: statuses, then the three labelled metric groups for the last genuine Gemini execution (HISTORICAL banner shows its ID, model, timestamp; if it was interrupted the banner says so).
2. **Experiments** → pick the experiment → compare the 4–6 candidate policies (filter *Feasible only*, sort by score, read CIs, see the infeasible baseline in coral) → *Sensitivity* tab.
3. **Agent Execution Traces** → open the iteration history: harness setup, each Gemini turn and the MCP tools it chose, how later experiments followed earlier results, and the final deterministic decision.
4. Turn Demonstration mode off. **Policy Optimization** → set the objective (e.g. 10,000 txns, ≤2 retries, no terminal retries) → confirm → *Start Gemini optimization agent* → watch the live trace → inspect the report (requires a Gemini key and available quota).
5. **PayPal Sandbox** (needs Sandbox credentials) → create an order → open the approval URL, approve as the Sandbox buyer → *Check order state and capture* → verify capture, debug IDs and history.

## Verification status

Offline: dashboard unit tests (data layer, labelling, redaction, errors, job runner) and headless Streamlit `AppTest` rendering of every section pass. Not verified: a pixel-level visual review in a real browser (none was available when this was built); a live Gemini run launched *from the dashboard*; live PayPal Sandbox operations from the dashboard (no credentials yet). The dashboard's Sandbox tests use a mock HTTP double that is not PayPal.
