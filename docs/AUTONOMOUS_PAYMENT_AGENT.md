# Autonomous Payment-Recovery Optimisation Agent (Phase 3)

Gemini plans experiments and proposes candidate recovery policies; every simulation runs through the FastMCP server; a deterministic
evaluator makes the final selection. All results are **SIMULATED under ASSUMED distributions** (see `docs/PAYMENT_SIMULATION_MODEL.md`) — not PayPal data,
and the agent performs **no PayPal calls**.

## 1. Architecture

```
 merchant objective (text) ──► planner.parse_objective ──► ObjectiveSpec (authoritative hard constraints)
                                                              │
 ┌──────────────────────────── agent/payments/agent.py (PaymentAgent) ───────────────────────────────┐
 │ harness: create experiment + baseline scenario + evaluate initial policy      (MCP, source=harness) │
 │ LLM loop (bounded):  Gemini ⇄ tool_adapter ⇄ MCP client ⇄ FastMCP server ⇄ engine/payments        │
 │ harness: DETERMINISTIC final phase (evaluator.py): fixed-seed re-evaluation → selection →         │
 │          repeat check → sensitivity → alternative scenarios → confidence rules                    │
 │ Gemini: writes an explanation (labelled; never used for selection)                                │
 └───────────────────────────────────────────────────────────────────────────────────────────────────┘
        persistence: storage/payment_agent_runs.py → data/payment_agent_runs.db (separate from simulation + Sandbox data)
```

| Module (`agent/payments/`) | Role |
|---|---|
| `models.py` | `AgentConfig` (env-driven; the only default model name lives here), `ObjectiveSpec`, record types, `MissingCredentialsError` |
| `planner.py` | deterministic objective parser + system/initial prompts |
| `tool_adapter.py` | **MCP ⇄ Gemini adapter** (discovery, schema conversion, execution, result compaction) |
| `llm.py` | `GeminiBackend` (google-genai, async, timeout, retry on 408/429/5xx), `LLMBackend` protocol |
| `optimizer.py` | candidate registry + deterministic `propose_variations` (local-search neighbours) |
| `evaluator.py` | pure functions: `select_winner`, `assess_confidence` |
| `decision_report.py` | structured `DecisionReport` (+ Markdown) |
| `agent.py` | orchestration, guardrails, budgets, final phase |
| `testing.py` | `ScriptedBackend` / `FailingBackend` test doubles (**not Gemini**) |

**Boundary:** nothing under `agent/payments/` imports `engine/`, `mcp_server/` or `spikes/` (enforced by `test_agent_never_imports_the_simulation_engine`). The agent works only on JSON returned by MCP tools.
The server module to launch is a string (`spikes.mcp_sim.server`, override `INOEMA_MCP_SERVER_MODULE`).

## 2. The MCP adapter (Gemini function calling ≠ MCP)

1. `McpToolAdapter.discover()` lists tools through an independent `fastmcp.Client` (stdio subprocess).
2. `to_gemini_schema()` converts each tool's JSON Schema to the Gemini `Schema` subset: inlines `$ref/$defs`; drops `additionalProperties`, `title`, `default`;
   maps `exclusiveMinimum/Maximum` → `minimum/maximum`; `anyOf[X,null]` → `nullable`; const-unions → `enum`; removes harness-owned params (`max_work_units`).
   Every declaration is validated by constructing the SDK's `types.Schema`.
3. Declarations (+ two local tools) go to Gemini with automatic function calling **disabled**; the agent runs the loop itself.
4. Each `function_call` is executed through MCP (`call_tool`), with timeout; failures become `{"error": ...}` function responses (the model sees and can adapt to them).
5. Real results return as `function_response` parts (compacted for the model's context; the **full** result is persisted). The model's own `content` (incl. any thought signatures) is appended to the history unchanged.
Hidden from the model: the legacy `simulate_release_schedule` tool and `create_payment_experiment` (the harness creates the experiment so constraints cannot be dropped).

## 3. Optimisation algorithm

*LLM-driven exploration* (bounded by `max_iterations` turns, `max_tool_calls`, `exploration_budget_units` = transactions×trials×policies): the model reads the objective,
the authoritative constraints and the initial policy's evaluation, then chooses experiments. It may call `compare_recovery_policies`, `run_payment_simulation`,
`run_sensitivity_analysis`, `generate_payment_scenarios`, `get_payment_transaction_trace`, `propose_policy_variations` (local deterministic neighbours: retry_limit ±1, backoff base ×/÷2,
multiplier, buyer timeout, window, alternate-funding / terminal-handling toggles) and finally `finish_optimization(shortlist, rationale)`. Each result carries `_harness` budget info so it can adapt.
Guardrails: foreign `experiment_id`/`scenario_id` rejected; hidden tools rejected; budget injected as `max_work_units` (server-enforced); a call limit with one grace turn to finish; LLM API failure ends exploration but not the run.

*Deterministic final phase* (not LLM-controlled; fixed `final_seed`, `final_trials`):
1. Shortlist = union of the model's shortlist, the 3 best observed-feasible candidates, and the initial policy (≤ 8).
2. One `compare_recovery_policies` on the baseline scenario. **Selection = top-ranked FEASIBLE policy** (ranking and feasibility come from the server's objective code: hard constraints first, then weighted score). Everything else gets a rejection reason (violations, or paired score gap + CI).
3. The identical request is re-run → **repeatability**.
4. `run_sensitivity_analysis` on the top 4 → **assumption sensitivity** (share of sweep points where the selected policy remains best, flip conditions).
5. Comparison across `network_instability`, `buyer_friction`, `high_failure_rate` scenarios → **scenario robustness**.
6. `assess_confidence` (rules below). Then the model writes an explanation, flagged as LLM-authored.
Given identical result JSON, steps 2 and 6 are pure functions → identical selection and confidence.

## 4. Recommendation confidence (never collapsed into one number)

| Notion | Meaning | Source |
|---|---|---|
| Repeatability | identical request reproduces identical results | step 3 |
| Statistical uncertainty | paired score margin vs runner-up, 95% CI (`CLEAR_MARGIN` / `NOT_SEPARABLE` / `NO_FEASIBLE_COMPETITOR`) — **simulation sampling error only** | step 2 |
| Assumption sensitivity | `ROBUST` ≥ 90% / `MODERATE` ≥ 60% / `FRAGILE` of sweep points keep the winner | step 4 |
| Scenario robustness | `WINS_ALL` / `WINS_SOME` / `WINS_NONE` | step 5 |

Overall label (heuristic): **HIGH** = repeatable ∧ clear margin ∧ ROBUST ∧ WINS_ALL; **MEDIUM** = repeatable ∧ (clear margin or no feasible competitor) ∧ ROBUST/MODERATE ∧ WINS_ALL/WINS_SOME;
else **LOW**; **NONE** when no candidate is feasible. Every report states: *confidence describes the simulation under assumed parameters, not real-world performance.*
Sensitivity directly changes the label (tested).

## 5. Objective parsing and enforcement

`parse_objective` extracts: transaction count, max automatic retries ("do not exceed two automatic retries"), "never automatically retry terminal failures", recovery windows ("within 6 hours"), idempotency mentions.
Unmentioned constraints use safe defaults (idempotent retries required, no auto-retry of buyer-action failures, 72 h window) and are labelled `default` in the report. Constraint-like sentences it does **not**
recognise produce warnings and are **not enforced**. Objective weights default to (revenue 1, cost 0.2, latency 0.1, unnecessary attempts 0.2) unless overridden. Gemini does not set constraints.

## 6. Configuration

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | required for a real run; without it the agent raises `MissingCredentialsError` — there is no fake fallback |
| `GEMINI_MODEL` | model id; default constant `DEFAULT_GEMINI_MODEL` in `agent/payments/models.py` (**availability for your key is not verified by the code — set it explicitly**) |
| `INOEMA_PAYMENT_DB`, `INOEMA_AGENT_DB`, `INOEMA_MCP_SERVER_MODULE` | optional paths / server module |

Tunables in `AgentConfig`: `max_iterations` (10), `max_tool_calls` (24), `exploration_budget_units` (8M), `exploration_trials` (10), `final_trials` (20), `final_seed`, timeouts, `llm_retries`, `initial_policy`.

## 6b. Safety boundaries

* Simulation only; no PayPal API calls (Sandbox execution is deliberately not wired in until the Sandbox integration is verified).
* The LLM proposes and interprets; it never computes scores, applies constraints, or makes the final selection.
* API keys: held only in `AgentConfig.api_key` (excluded from `repr` and `redacted()`); all persisted data passes through `scrub()` (redacts secret-named keys and literal key values). Test inspects raw DB bytes.
* Bounded turns, tool calls, simulation work, timeouts; tool errors are returned to the model, not raised.
* Persisted separately: simulation results (`payment_sim_*`), agent traces (`payment_agent_*`), and (elsewhere) Sandbox logs.

## 7. Persistence (`data/payment_agent_runs.db`)

`payment_agent_runs` (execution id, provider, model, config without key, objective, experiment id, termination reason, final report), `payment_agent_tool_calls` (every MCP and local call: args, full result, source gemini/harness, duration, work units),
`payment_agent_events` (objective parse, tools discovered, each LLM turn with text and requested calls, finish request, final selection, confidence), `payment_agent_candidates` (definitions + all observations).

## 8. Running it

```bash
cp .env.example .env     # set GEMINI_API_KEY (and optionally GEMINI_MODEL)
python -m scripts.payment_agent_demo                      # GENUINE Gemini run (~1-3 min depending on the model)
python -m scripts.payment_agent_demo --scripted           # pipeline demo with a scripted test double — NOT Gemini; labelled everywhere
python -m scripts.payment_agent_demo --json-out report.json
python -m pytest -q                                       # unit tests (mocked LLM only)
python -m pytest tests/integration -m gemini_live -v -s   # real Gemini API; SKIPPED without a key (a skip is not a pass)
```

## 9. Evidence classes

| Evidence | What it proves |
|---|---|
| Unit tests (`tests/test_payment_agent.py`, `tests/test_sensitivity.py`) | adapter conversion, loop, guards, limits, failures, evaluator, persistence — with a **scripted LLM** and stubbed Gemini clients; real MCP server/engine/SQLite where stated |
| `--scripted` demo | the whole pipeline end-to-end against the real MCP server with a fixed non-LLM script |
| `tests/integration/test_gemini_live.py` and the default demo | genuine Gemini calls (only when a key is configured) |

## 10. Known limitations

* No live Gemini execution has been performed in the development environment (no key was available): schema acceptance by the Gemini API, model availability, and the model's actual planning quality are **unverified**.
* NL parsing is rule-based and narrow; unrecognised constraints are not enforced.
* The agent's search is only as good as the model's proposals plus one deterministic neighbour generator; no guarantee of global optimality.
* Confidence labels use heuristic thresholds over one-at-a-time sweeps; parameter interactions and uncertainty in the assumed distributions are not covered.
* The final shortlist is a deterministic union, so the selected policy can differ from the model's favourite (reported in the notes).
* MCP server launched as a stdio subprocess per run; the server module still lives under `spikes/`.
