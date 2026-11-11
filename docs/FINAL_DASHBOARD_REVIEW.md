# Final Dashboard Review (Phase 5.1)

## 1. Default workspace changes

* **INOEMA Payments is now the default workspace** (`dashboard/app.py::_default_workspace`). The sidebar *Workspace* selector lists **INOEMA Payments** first and the original **Simulation Control Center** second; the page title is the neutral "INOEMA" because both workspaces share it.
* **The original Simulation Control Center is untouched and accessible**: sidebar → Workspace → *Simulation Control Center*, or `?workspace=simulation`, or `INOEMA_DEFAULT_WORKSPACE=simulation`. Verified in the real browser (screenshot `20_…`) and by tests.
* Updated tests: the 4 pre-existing `tests/test_dashboard.py` tests now select the legacy workspace explicitly (`_legacy_app()`), plus one new test that the default is Payments and the original is one click away. In `tests/test_dashboard_payments_ui.py` the two old "legacy is the default" tests were replaced by three tests (Payments default and listed first; original workspace accessible and unchanged; env/query override).

## 2. Demonstration mode: complete vs incomplete

* **COMPLETE** (`dashboard/payments/data.py::is_complete`) means: provider `GEMINI_API`, status `COMPLETED`, **and** termination `finished_by_llm` (the model itself called `finish_optimization`). Nothing else qualifies (`llm_error`, `max_iterations`, `model_stopped_without_finish`, failures, scripted doubles are all "not complete").
* Demonstration mode opens on the **latest COMPLETE genuine execution**, even if a newer interrupted one exists (tested). If none exists it shows the latest genuine one under a prominent amber banner — *"NO COMPLETE genuine execution is available … INTERRUPTED (llm_error) … Treat it as an incomplete historical run, not a completed autonomous optimisation"* — and the Overview metric, workflow step, Optimization page, Traces page and Sandbox policy picker all carry the same INTERRUPTED label. The Traces page "Run state" shows INTERRUPTED instead of the harness' misleading "COMPLETED" status.
* **Current data:** the only genuine execution on disk is the interrupted 10,000-transaction run (`dbfbf757…`, quota exhausted at turn 6), so Demonstration mode correctly shows the **incomplete** state (screenshots `11_…`, `12_…`). The complete 1,000-transaction run from Phase 3.1 cannot be shown: its temporary database was rotated away by pytest and only a console transcript survives. A complete run needs fresh Gemini quota (`python -m scripts.payment_agent_demo`).

## 3. Overview and workflow

* **Four explainer cards**: what INOEMA Payments does; what the AI agent optimizes; how simulation and MCP interact (Gemini → MCP tools → seeded Monte Carlo; a deterministic evaluator, not the LLM, selects); how PayPal Sandbox validates the selected policy (create → buyer approval → capture → verify; mechanics, not simulated success rates).
* **Pipeline diagram**: Merchant objective → Gemini agent → MCP tools → Simulation engine → Deterministic evaluator → PayPal Sandbox, each box coloured by its data source.
* **End-to-end workflow indicator** on every section (detailed on Overview with navigation buttons, compact elsewhere): Understand → Optimize → Compare → Validate → Audit. The current step is highlighted; each step shows a status **derived from persisted data** (DONE / COMPLETE / INTERRUPTED / VERIFIED / IN PROGRESS / NOT STARTED, with "Gemini/PayPal not configured" notes), never invented.
* **Consistency**: every section now has the same structure — workflow indicator, kicker, one title, a one-line purpose, and its data-source badge first (tested for all five).
* **Visual distinction of sources**: four colours *and* four symbols (◆ simulated teal, ✦ agent-derived deep green, ● Sandbox actual coral, ○ system grey); each metric group has a coloured accent bar plus its badge; the same colour marks that source in the workflow indicator, the pipeline diagram and its own page. The code refuses to render a mixed-source metric group.

## 4. Test results

`python -m pytest -q` (no credentials in the shell): **335 passed, 7 skipped** in 206 s (302 before this phase + 33 new). The 7 skips are the live Gemini (2) and live PayPal (5) tests: *a skip is not a pass* — those integrations were not exercised by this run.

| New/changed in this phase | Count |
|---|---|
| `tests/test_dashboard.py` (4 updated + 1 new) | +1 |
| `tests/test_dashboard_payments_data.py` (completeness logic, demo-run selection, pinned metrics, workflow states) | +11 |
| `tests/test_dashboard_payments_ui.py` (default workspace, explainer, workflow indicator, consistency, source colours, demo complete/incomplete, plus a regression test for each browser-review finding) | +21 |

## 5. Browser review evidence

**A real browser review was completed.** Playwright's *Python package* is not installed, but a Playwright-managed Chromium (`chrome-headless-shell`, "Chrome for Testing 158") was already in `~/Library/Caches/ms-playwright`, and `websockets` was already in the environment. `scripts/browser_review.py` drives that browser over the Chrome DevTools Protocol (nothing was installed), launches the **real Streamlit app on the real persisted data**, and for each step: waits for the page to settle, checks the header pill for the mode it claims, records on-page Streamlit exceptions and console errors, and takes a full-page screenshot at **1440 px wide** (desktop).

Result of the final run: **11 screenshots, 0 page exceptions, 0 console errors, all mode checks verified** (`docs/screenshots/review_summary.json`): the five sections live (`01`–`05`), the five sections in Demonstration mode (`11`–`15`), and the original Simulation Control Center (`20`).

I opened and read the screenshots (they are not just generated). **Findings and fixes** (each has a regression test):

| Finding (seen in the real browser) | Fix |
|---|---|
| My own harness clicked the demo toggle's `<label>` and it did **not** switch modes: the first "demo" screenshots were actually live mode (identical to the live ones) | the toggle works in the browser (input click); the harness now clicks the input and **refuses to label a screenshot "demo" unless the header pill says so** |
| Disabled primary buttons (Start agent, Create Sandbox order) looked fully enabled (legacy CSS overrides the disabled look) | disabled primary buttons are greyed with `not-allowed` cursor |
| Selected policy name truncated ("intel_ab_exp_b4…") | full name shown beneath long metric values |
| "Latest experiment" printed a raw Python dict | "9 comparisons · 1 single-policy run · 1 sensitivity analysis" |
| Model field showed the retired default `gemini-2.5-flash` as if recommended | explicit warning when `GEMINI_MODEL` is unset |
| Experiments table overflowed; revenue/latency/cost off-screen | short headers, fixed widths, CI as one column; extras opt-in; all required metrics visible at 1440 px |
| Overlapping unreadable labels in the revenue/latency scatter | one legend entry per policy, × for infeasible |
| Sandbox currency defaulted to **AUD**; policy came from the interrupted run without saying so | USD default; picker prefers the latest COMPLETE run and warns when the policy is from an INTERRUPTED one |
| Traces "Status: COMPLETED" contradicted the INTERRUPTED banner; Gemini's `$\le 4320$` rendered as LaTeX | "Run state" metric; dollars escaped |
| Final-evaluation table clipped; raw API error dict printed; ISO timestamp in HISTORICAL badge | compact CI column; classified readable note + sanitised raw text in an expander; formatted timestamp |

**What I inspected in final form:** `02`, `03`, `04`, `05` after their fixes. `01`, `11`, `12` were inspected after the main fixes; the last three small tweaks (timestamp format, final-evaluation CI column, error note) were regenerated and verified by tests and the exception/console checks, but `11`–`15` were not all re-inspected after them. Only a desktop viewport and one headless Chromium build were reviewed (no Safari/Firefox, tablet or mobile).

## 6. Remaining UI issues

* Traces → "Candidate policies…" table: the *Changes vs initial policy* column is clipped on the right (the full text is in the expander below it).
* The revenue/latency scatter points for near-identical policies overlap by nature (they are almost the same policy); hover and legend disambiguate.
* The big `st.metric` value for long policy names still ends with "…" (the full name is shown directly beneath).
* Experiments table: *Violations* is now opt-in via "Additional columns" (in the CSV always).
* Gemini's raw `$\le 4320$` is displayed literally (escaped), which looks slightly technical.
* The Demonstration-mode Optimization page is long (≈3,900 px) because it embeds the full decision report.
* Pre-existing and untouched: the legacy workspace still uses the deprecated `use_container_width` argument (warnings in logs) and its own "Mode Mocked"/ILLUSTRATIVE labels.
* Not verified: the live (non-demo) optimization run and the Sandbox flow *from the dashboard* with real services (no Gemini quota / PayPal credentials); a complete genuine Gemini execution is unavailable in the persisted data.

## 7. Demo launch instructions

```bash
python -m scripts.run_payments_dashboard --demo     # judge-friendly: persisted genuine history, live actions disabled
python -m scripts.run_payments_dashboard            # live workspace (needs GEMINI_API_KEY / GEMINI_MODEL; PayPal Sandbox keys for step 4)
python -m scripts.browser_review                    # re-run the automated visual review; screenshots land in docs/screenshots/
```

**Manual visual verification** (if you prefer not to use the harness): open http://localhost:8501 at ≥1440 px wide; confirm (1) it opens on *INOEMA Payments → Overview* with the four explainer cards, the pipeline strip and the five-step workflow indicator; (2) the sidebar lists Payments first and *Simulation Control Center* still opens the original app; (3) with *Demonstration mode* on, the header pill reads "Demonstration mode · historical data" and, while only the interrupted run exists, an amber "NO COMPLETE genuine execution" banner appears on Overview and Policy Optimization and Traces says "INTERRUPTED / INCOMPLETE"; (4) visit all five sections: same layout (indicator, kicker, title, purpose, source badge), teal for simulated, deep green for agent-derived, coral for Sandbox; (5) disabled buttons look greyed out; (6) Experiments shows revenue, latency, retries and cost without scrolling.
