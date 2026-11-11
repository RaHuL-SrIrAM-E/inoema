"""Autonomous payment-recovery optimisation agent.

Control flow
------------
  harness : parse objective -> create experiment (authoritative constraints) -> baseline scenario -> evaluate initial policy
  LLM loop: Gemini plans and chooses which tools to call and which candidate policies to try; every call executes
            through MCP; real results go back to Gemini (bounded turns / tool calls / simulation budget)
  harness : DETERMINISTIC final phase -- fixed-seed re-evaluation of the shortlist, selection of the best feasible
            policy, repeatability check, sensitivity analysis, alternative-scenario robustness, confidence rules
  LLM     : writes an explanation (labelled, never used for selection)

The agent imports nothing from engine/. The LLM can propose and interpret; only the evaluator selects.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from google.genai import types

from agent.payments import evaluator
from agent.payments.decision_report import DecisionReport
from agent.payments.llm import GeminiBackend, LLMBackend, LLMError
from agent.payments.models import (AgentConfig, AgentError, ObjectiveSpec, ToolCallRecord)
from agent.payments.optimizer import CandidateRegistry, ingest_tool_result, propose_variations
from agent.payments.planner import SYSTEM_PROMPT, build_initial_message, parse_objective
from agent.payments.tool_adapter import McpToolAdapter, compact_result, make_stdio_client
from storage.payment_agent_runs import AgentRunStore

SIM_TOOLS = frozenset({"run_payment_simulation", "compare_recovery_policies", "run_sensitivity_analysis"})
REQUIRED_TOOLS = frozenset({"create_payment_experiment", "generate_payment_scenarios", "run_payment_simulation",
                            "compare_recovery_policies", "run_sensitivity_analysis"})
HIDDEN_TOOLS = frozenset({"simulate_release_schedule", "create_payment_experiment"})  # legacy tool; experiment creation is harness-owned
HIDDEN_PARAMS = frozenset({"max_work_units"})  # injected by the harness (budget enforcement)

_LOCAL_DECLARATIONS = [
    types.FunctionDeclaration(
        name="propose_policy_variations",
        description="Deterministically suggest neighbouring candidate policies (one dimension changed at a time) around a policy "
                    "you have already evaluated. Returns PolicySpec-shaped dicts you can pass to compare_recovery_policies. "
                    "Does not run any simulation and does not judge feasibility.",
        parameters=types.Schema.model_validate({"type": "object", "properties": {
            "policy_name": {"type": "string", "description": "Name of an already-evaluated candidate."},
            "dimensions": {"type": "array", "items": {"type": "string", "enum": [
                "retry_limit", "backoff_base_minutes", "backoff_multiplier", "buyer_action_timeout",
                "maximum_recovery_window", "alternate_funding_action", "terminal_failure_handling"]}},
            "max_variants": {"type": "integer", "minimum": 1, "maximum": 12}}, "required": ["policy_name"]})),
    types.FunctionDeclaration(
        name="finish_optimization",
        description="Call when you are done exploring. Provide a shortlist of evaluated candidate names and your rationale. "
                    "The final selection is made by the deterministic evaluator, not by this call.",
        parameters=types.Schema.model_validate({"type": "object", "properties": {
            "shortlist": {"type": "array", "items": {"type": "string"}, "minItems": 1},
            "rationale": {"type": "string"}}, "required": ["shortlist", "rationale"]})),
]


@dataclass
class _State:
    execution_id: str
    spec: ObjectiveSpec
    registry: CandidateRegistry = field(default_factory=CandidateRegistry)
    calls: list[ToolCallRecord] = field(default_factory=list)
    experiment_id: str = ""
    scenario_ids: dict[str, str] = field(default_factory=dict)  # name -> id
    baseline_scenario: str = ""
    used_units: int = 0
    final_phase_units: int = 0
    llm_function_calls: int = 0
    turns: int = 0
    proposed_variations: set[str] = field(default_factory=set)
    finish: Optional[dict[str, Any]] = None
    termination: str = ""
    notes: list[str] = field(default_factory=list)
    initial_name: str = ""
    limit_turns: int = 0  # consecutive turns in which the tool-call limit rejected a call
    limit_hit_this_turn: bool = False


def _brief_args(args: dict[str, Any]) -> str:
    out = []
    for k, v in args.items():
        if k in ("experiment_id", "scenario_id", "max_work_units"):
            continue
        if k == "policies":
            v = [(p.get("name") or p.get("base")) for p in v]
        elif k == "policy" and isinstance(v, dict):
            v = v.get("name") or v.get("base")
        out.append(f"{k}={v}")
    return ", ".join(out)[:220]


def _brief_result(tool: str, data: Any, ok: bool) -> str:
    if not ok or not isinstance(data, dict):
        return str(data)[:200]
    if tool == "compare_recovery_policies":
        return "ranking: " + " > ".join(f"{r['policy']}({'F' if r['feasible'] else 'INFEASIBLE'},{r['score_mean']:+.3f})" for r in data["ranking"])
    if tool == "run_payment_simulation":
        e = data["evaluation"]
        return f"{data['policy_definition']['name']}: feasible={e['feasible']} score={e['score_mean']:+.3f}"
    if tool == "run_sensitivity_analysis":
        r = data["robustness"]
        return f"{r['label']} (winner stable in {r['winner_stable_share']:.0%} of {r['n_points']} points); most sensitive: {data['most_sensitive_parameters']}"
    return "ok"


class PaymentAgent:
    def __init__(self, config: AgentConfig, backend: Optional[LLMBackend] = None,
                 client_factory: Optional[Callable[[], Any]] = None, store: Optional[AgentRunStore] = None,
                 progress: Optional[Callable[[str], None]] = None):
        self.cfg = config
        self._say = progress or (lambda msg: None)
        self.backend = backend if backend is not None else GeminiBackend(config)  # raises MissingCredentialsError without a key
        self.client_factory = client_factory or (lambda: make_stdio_client(config.server_module, config.db_path, config.tool_timeout_s))
        self.store = store

    # ------------------------------------------------------------------ MCP call helper
    async def _call(self, st: _State, ad: McpToolAdapter, tool: str, args: dict[str, Any], source: str, iteration: int) -> ToolCallRecord:
        self._say(f"  [{source}{'' if iteration < 0 else f' it{iteration}'}] -> MCP {tool}({_brief_args(args)})")
        res = await ad.call(tool, args)
        data = res.data
        units = int(data.get("work_units", 0)) if res.ok and isinstance(data, dict) else 0
        rec = ToolCallRecord(len(st.calls) + 1, iteration, source, tool, args, res.ok, data, res.duration_ms, units)
        st.calls.append(rec)
        if iteration >= 0:
            st.used_units += units
        else:
            st.final_phase_units += units
        if res.ok and iteration >= 0:  # final-phase (fixed-seed) results are not exploration observations
            for name in ingest_tool_result(st.registry, tool, args, data, rec.seq, "baseline" if source == "harness" else "gemini"):
                if name in st.proposed_variations:
                    st.registry.candidates[name].origin = "variation"
        self.store.add_tool_call(st.execution_id, rec.to_dict())
        return rec

    def _remaining(self, st: _State) -> int:
        return max(0, self.cfg.exploration_budget_units - st.used_units)

    # ------------------------------------------------------------------ public entry
    async def run(self, objective_text: str, **overrides: Any) -> DecisionReport:
        spec = parse_objective(objective_text, **overrides)
        store = self.store = self.store or AgentRunStore(self.cfg.agent_db_path, secrets=[self.cfg.api_key])
        eid = store.start_run(self.backend.provider, self.cfg.model, self.cfg.redacted(), objective_text, spec.to_dict())
        st = _State(eid, spec)
        store.add_event(eid, "objective_parsed", spec.to_dict())
        try:
            async with McpToolAdapter(self.client_factory, HIDDEN_TOOLS, HIDDEN_PARAMS, self.cfg.tool_timeout_s) as ad:
                await ad.discover()
                missing = REQUIRED_TOOLS - set(ad.tools)
                if missing:
                    raise AgentError(f"MCP server is missing required tools: {sorted(missing)}")
                store.add_event(eid, "tools_discovered", sorted(ad.tools))
                report = await self._run(st, ad)
        except Exception as e:
            store.finish_run(eid, "FAILED", st.termination or "error", None, error=f"{type(e).__name__}: {e}")
            raise
        store.save_candidates(eid, st.registry.to_dict())
        store.finish_run(eid, "COMPLETED", st.termination, report.to_dict())
        return report

    # ------------------------------------------------------------------ orchestration
    async def _run(self, st: _State, ad: McpToolAdapter) -> DecisionReport:
        cfg, spec, store = self.cfg, st.spec, self.store

        # -- harness setup: authoritative experiment + scenario + initial policy -------------------------
        exp = await self._call(st, ad, "create_payment_experiment", {"name": "autonomous-agent", "objective": spec.objective_payload(),
                                                                       "constraints": spec.constraints_payload()}, "harness", 0)
        if not exp.ok:
            raise AgentError(f"could not create experiment: {exp.result}")
        st.experiment_id = exp.result["experiment_id"]
        store.set_experiment(st.execution_id, st.experiment_id)
        sc = await self._call(st, ad, "generate_payment_scenarios", {"experiment_id": st.experiment_id,
                              "scenarios": [{"preset": "baseline", "n_transactions": spec.n_transactions, "name": "baseline"}]}, "harness", 0)
        if not sc.ok:
            raise AgentError(f"could not create scenario: {sc.result}")
        st.baseline_scenario = sc.result["scenarios"][0]["scenario_id"]
        st.scenario_ids["baseline"] = st.baseline_scenario
        init = await self._call(st, ad, "run_payment_simulation", {"experiment_id": st.experiment_id, "scenario_id": st.baseline_scenario,
                                "policy": {"base": cfg.initial_policy}, "n_trials": cfg.exploration_trials, "master_seed": 0}, "harness", 0)
        if not init.ok:
            raise AgentError(f"could not evaluate initial policy: {init.result}")
        st.initial_name = init.result["policy_definition"]["name"]
        initial_view = compact_result("run_payment_simulation", init.result)

        await self._llm_loop(st, ad, initial_view)
        return await self._finalize(st, ad)

    # ------------------------------------------------------------------ LLM exploration loop
    async def _llm_loop(self, st: _State, ad: McpToolAdapter, initial_view: dict[str, Any]) -> None:
        cfg, store = self.cfg, self.store
        decls = ad.declarations() + _LOCAL_DECLARATIONS
        gen_cfg = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT + "\nUse the same master_seed across experiments so scores are comparable.",
                                              tools=[types.Tool(function_declarations=decls)],
                                              automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        contents = [types.Content(role="user", parts=[types.Part(text=build_initial_message(
            st.spec, cfg, st.experiment_id, st.baseline_scenario, initial_view, [d.name for d in decls]))])]

        for iteration in range(1, cfg.max_iterations + 1):
            st.turns = iteration
            try:
                resp = await self.backend.generate(contents, gen_cfg)
            except LLMError as e:
                st.termination = "llm_error"
                st.notes.append(f"LLM error ended exploration: {e}")
                store.add_event(st.execution_id, "llm_error", str(e))
                return
            cand = resp.candidates[0] if resp.candidates else None
            if cand is None or cand.content is None or not cand.content.parts:
                st.termination = "empty_llm_response"
                store.add_event(st.execution_id, "llm_turn", {"iteration": iteration, "empty": True})
                return
            contents.append(cand.content)
            text = " ".join(p.text for p in cand.content.parts if p.text and not getattr(p, "thought", False)).strip()
            calls = list(resp.function_calls or [])
            self._say(f"[LLM turn {iteration}] {text[:600] if text else '(no text)'}")
            for c in calls:
                self._say(f"  LLM requests function: {c.name}({_brief_args(dict(c.args or {}))})")
            store.add_event(st.execution_id, "llm_turn", {"iteration": iteration, "text": text,
                                                         "function_calls": [{"name": c.name, "args": dict(c.args or {})} for c in calls]})
            if not calls:
                st.termination = "model_stopped_without_finish"
                return

            parts: list[types.Part] = []
            st.limit_hit_this_turn = False
            for fc in calls:
                payload = await self._handle_function_call(st, ad, fc, iteration)
                parts.append(types.Part(function_response=types.FunctionResponse(name=fc.name, id=fc.id, response=payload)))
            contents.append(types.Content(role="user", parts=parts))
            if st.finish is not None:
                st.termination = "finished_by_llm"
                return
            st.limit_turns = st.limit_turns + 1 if st.limit_hit_this_turn else 0
            if st.limit_turns >= 2:  # told to finish, kept calling tools: stop (one grace turn is allowed)
                st.termination = "max_tool_calls"
                return
        st.termination = "max_iterations"

    async def _handle_function_call(self, st: _State, ad: McpToolAdapter, fc: types.FunctionCall, iteration: int) -> dict[str, Any]:
        cfg = self.cfg
        name, args = fc.name, dict(fc.args or {})
        st.llm_function_calls += 1
        meta = lambda: {"budget_remaining_work_units": self._remaining(st),  # noqa: E731
                        "tool_calls_remaining": max(0, cfg.max_tool_calls - st.llm_function_calls),
                        "turns_remaining": cfg.max_iterations - iteration}

        def local(rec_ok: bool, result: Any) -> None:
            r = ToolCallRecord(len(st.calls) + 1, iteration, "gemini", name, args, rec_ok, result, 0.0)
            st.calls.append(r)
            self.store.add_tool_call(st.execution_id, r.to_dict())

        if st.llm_function_calls > cfg.max_tool_calls and name != "finish_optimization":
            err = f"tool-call limit ({cfg.max_tool_calls}) reached; call finish_optimization now"
            st.limit_hit_this_turn = True
            local(False, err)
            return {"error": err, "_harness": meta()}

        if name == "finish_optimization":
            names = [n for n in args.get("shortlist", []) if isinstance(n, str)]
            unknown = [n for n in names if n not in st.registry.candidates]
            st.finish = {"shortlist": [n for n in names if n in st.registry.candidates], "rationale": str(args.get("rationale", ""))[:2000],
                         "unknown_names": unknown}
            self._say(f"  [LLM finish] shortlist={st.finish['shortlist']}  rationale: {st.finish['rationale'][:300]}")
            self.store.add_event(st.execution_id, "finish_requested", st.finish)
            local(True, st.finish)
            return {"result": {"status": "accepted", "ignored_unknown_candidates": unknown,
                               "next": "deterministic evaluator will now re-evaluate the shortlist"}, "_harness": meta()}

        if name == "propose_policy_variations":
            pname = args.get("policy_name")
            if pname not in st.registry.candidates:
                err = f"unknown candidate {pname!r}; known: {sorted(st.registry.candidates)}"
                local(False, err)
                return {"error": err, "_harness": meta()}
            variants = propose_variations(st.registry.definition(pname), args.get("dimensions"), int(args.get("max_variants", 8)))
            st.proposed_variations.update(v["name"] for v in variants)
            self._say(f"  [local] propose_policy_variations({pname}) -> {[v['name'] for v in variants]}")
            local(True, variants)
            return {"result": {"variants": variants}, "_harness": meta()}

        # MCP tools: guardrails then execute
        err = self._guard(st, ad, name, args)
        if err:
            local(False, err)
            return {"error": err, "_harness": meta()}
        rec = await self._call(st, ad, name, args, "gemini", iteration)
        if not rec.ok:
            return {"error": rec.result, "_harness": meta()}
        if name == "generate_payment_scenarios":
            for s in rec.result["scenarios"]:
                st.scenario_ids[s["name"]] = s["scenario_id"]
        return {"result": compact_result(name, rec.result), "_harness": meta()}

    def _guard(self, st: _State, ad: McpToolAdapter, name: str, args: dict[str, Any]) -> Optional[str]:
        if name not in ad.tools or name in ad.hidden_tools:
            return f"tool {name!r} is not available"
        if args.get("experiment_id", st.experiment_id) != st.experiment_id:
            return "experiment_id does not belong to this run"
        args["experiment_id"] = st.experiment_id
        if name != "generate_payment_scenarios" and name != "get_payment_experiment_results":
            sid = args.setdefault("scenario_id", st.baseline_scenario)
            if sid not in st.scenario_ids.values():
                return f"unknown scenario_id {sid!r}; known scenarios: {st.scenario_ids}"
        if name in SIM_TOOLS:
            if self._remaining(st) <= 0:
                return "simulation budget exhausted; call finish_optimization"
            args["max_work_units"] = self._remaining(st)
        return None

    # ------------------------------------------------------------------ deterministic final phase
    async def _finalize(self, st: _State, ad: McpToolAdapter) -> DecisionReport:
        cfg, reg, store = self.cfg, st.registry, self.store
        shortlist = list(st.finish["shortlist"]) if st.finish else []
        for n in reg.top_feasible(3) + [st.initial_name]:
            if n not in shortlist:
                shortlist.append(n)
        # priority: initial first, then best observed feasible, then the rest (compare accepts <= 8 policies)
        rest = [n for n in shortlist if n != st.initial_name]
        order = {n: i for i, n in enumerate(reg.top_feasible(len(reg.candidates)))}
        rest.sort(key=lambda n: (order.get(n, 10**6), n))
        names = [st.initial_name] + rest[:7]
        self._say(f"[deterministic evaluator] final fixed-seed evaluation of {names}")
        st.notes.append(f"final shortlist (deterministic union of LLM shortlist, top-3 observed feasible, initial policy): {names}")

        final = repeat = sens = None
        sc_checks: list[dict[str, Any]] = []
        final_args: dict[str, Any] = {}
        if len(names) >= 2:
            final_args = {"experiment_id": st.experiment_id, "scenario_id": st.baseline_scenario,
                          "policies": [reg.definition(n) for n in names], "n_trials": cfg.final_trials, "master_seed": cfg.final_seed}
            rec = await self._call(st, ad, "compare_recovery_policies", dict(final_args), "harness", -1)
            if not rec.ok:
                raise AgentError(f"final evaluation failed: {rec.result}")
            final = rec.result
            rep = await self._call(st, ad, "compare_recovery_policies", dict(final_args), "harness", -1)
            repeat = rep.result if rep.ok else None
        else:
            st.notes.append("fewer than two candidates available; no final comparison possible")

        selection = evaluator.select_winner(final) if final else evaluator.Selection(None, (), (), 0)
        winner = selection.winner

        if final and winner:
            top = [r["policy"] for r in final["ranking"]][:4]
            sres = await self._call(st, ad, "run_sensitivity_analysis", {
                "experiment_id": st.experiment_id, "scenario_id": st.baseline_scenario,
                "policies": [reg.definition(n) for n in top], "n_trials": cfg.robustness_trials, "master_seed": cfg.final_seed,
                "sweep_n_transactions": cfg.robustness_n_transactions}, "harness", -1)
            sens = sres.result if sres.ok else None
            if not sres.ok:
                st.notes.append(f"sensitivity analysis failed: {sres.result}")
            gen = await self._call(st, ad, "generate_payment_scenarios", {"experiment_id": st.experiment_id, "scenarios": [
                {"preset": p, "n_transactions": cfg.robustness_n_transactions, "name": p}
                for p in ("network_instability", "buyer_friction", "high_failure_rate")]}, "harness", -1)
            if gen.ok:
                for s in gen.result["scenarios"]:
                    st.scenario_ids[s["name"]] = s["scenario_id"]
                    cmp_ = await self._call(st, ad, "compare_recovery_policies", {
                        "experiment_id": st.experiment_id, "scenario_id": s["scenario_id"], "policies": [reg.definition(n) for n in names],
                        "n_trials": cfg.robustness_trials, "master_seed": cfg.final_seed}, "harness", -1)
                    if cmp_.ok:
                        sel_ = evaluator.select_winner(cmp_.result)
                        sc_checks.append({"scenario": s["name"], "winner": sel_.winner, "n_feasible": sel_.n_feasible})
                    else:
                        st.notes.append(f"scenario {s['name']} comparison failed: {cmp_.result}")
            else:
                st.notes.append(f"could not generate robustness scenarios: {gen.result}")

        conf = evaluator.assess_confidence(winner, final, repeat, sens, sc_checks) if final else evaluator.assess_confidence(None, {}, None, None, [])
        store.add_event(st.execution_id, "final_selection", {"winner": winner, "shortlist": names, "rejected": [r.to_dict() for r in selection.rejected]})
        store.add_event(st.execution_id, "confidence", conf.to_dict())

        explanation = await self._explain(st, selection, conf, final) if cfg.include_llm_explanation and final else None
        return self._build_report(st, names, final, selection, conf, sens, explanation)

    async def _explain(self, st: _State, selection: evaluator.Selection, conf: evaluator.ConfidenceAssessment, final: dict[str, Any]) -> Optional[str]:
        import json
        facts = {"selected": selection.winner, "ranking": compact_result("compare_recovery_policies", final)["ranking"],
                 "rejected": [r.to_dict() for r in selection.rejected], "confidence": conf.to_dict(), "objective": st.spec.to_dict()}
        prompt = ("You are explaining a payment-recovery decision to a merchant. The selection below was made by a deterministic evaluator; "
                  "do NOT change it or invent numbers. In under 200 words: state the choice, why it beat the alternatives, and what the "
                  "confidence assessment does and does not mean. Remind the reader results are simulated under assumed distributions.\n\n"
                  + json.dumps(facts, default=str)[:12000])
        try:
            resp = await self.backend.generate([types.Content(role="user", parts=[types.Part(text=prompt)])],
                                               types.GenerateContentConfig(system_instruction="Be precise and concise."))
            cand = resp.candidates[0] if resp.candidates else None
            return " ".join(p.text for p in cand.content.parts if p.text).strip() if cand and cand.content else None
        except Exception as e:  # noqa: BLE001
            st.notes.append(f"LLM explanation unavailable: {type(e).__name__}: {str(e)[:200]}")
            return None

    # ------------------------------------------------------------------ report
    def _build_report(self, st: _State, names: list[str], final: Optional[dict[str, Any]], selection: evaluator.Selection,
                      conf: evaluator.ConfidenceAssessment, sens: Optional[dict[str, Any]], explanation: Optional[str]) -> DecisionReport:
        reg, cfg = st.registry, self.cfg
        init_cand = reg.candidates[st.initial_name]
        io = init_cand.observations[0] if init_cand.observations else None
        initial = {"name": st.initial_name, "definition": init_cand.definition,
                   **({"feasible": io.feasible, "score": round(io.score_mean, 4), "recovery_rate": round(io.recovery_rate, 4),
                       "violations": io.violations} if io else {})}
        sim_calls = [c for c in st.calls if c.tool in SIM_TOOLS and c.ok]
        sim_cfg = [{"seq": c.seq, "source": c.source, "tool": c.tool, "phase": "final" if c.iteration < 0 else "exploration",
                    "scenario_id": c.args.get("scenario_id"), "n_trials": c.args.get("n_trials"), "master_seed": c.args.get("master_seed"),
                    "n_policies": len(c.args.get("policies", [c.args.get("policy")] if c.args.get("policy") else [])),
                    "work_units": c.work_units} for c in sim_calls]
        feas = []
        for c in reg.candidates.values():
            for o in c.observations:
                feas.append({"candidate": c.name, "seq": o.seq, "feasible": o.feasible, "score": round(o.score_mean, 4), "violations": o.violations})
        fe: dict[str, Any] = {}
        selected = None
        if final:
            comp = compact_result("compare_recovery_policies", final)
            fe = {"scenario": "baseline", "n_trials": final["n_trials"], "n_transactions": final["n_transactions"], "master_seed": final["master_seed"],
                  "ranking": [{"rank": r["rank"], "policy": r["policy"], "feasible": r["feasible"], "score": r["score"],
                               "recovery_rate": r["recovery_rate"], "recovery_rate_ci95": r["recovery_rate_ci95"],
                               "avg_recovery_minutes": r["avg_recovery_minutes"], "unnecessary_retries": r["unnecessary_retries"],
                               "violations": r["violations"]} for r in comp["ranking"]]}
            if selection.winner:
                row = next(r for r in fe["ranking"] if r["policy"] == selection.winner)
                selected = {"name": selection.winner, "definition": reg.definition(selection.winner), "metrics": row}
        sf = conf.assumption_sensitivity if sens is not None else {"label": "NOT_ASSESSED"}
        gem_calls = [c for c in st.calls if c.source == "gemini"]
        return DecisionReport(
            execution_id=st.execution_id, llm={"provider": self.backend.provider, "model": cfg.model},
            merchant_objective=st.spec.to_dict(), initial_policy=initial,
            candidates_explored=[{"name": c.name, "origin": c.origin, "n_observations": len(c.observations),
                                  "ever_feasible": c.ever_feasible(), "best_feasible_score": (round(c.best_feasible_score(), 4) if c.best_feasible_score() is not None else None),
                                  "definition": c.definition} for c in reg.candidates.values()],
            experiments={"simulation_calls": len(sim_calls), "by_gemini": sum(c.source == "gemini" for c in sim_calls),
                         "by_harness": sum(c.source == "harness" for c in sim_calls), "tool_calls_total": len(st.calls)},
            simulation_configurations=sim_cfg, final_evaluation=fe, feasibility_decisions=feas,
            sensitivity_findings=sf, selected_policy=selected, rejected_alternatives=[r.to_dict() for r in selection.rejected],
            confidence=conf.to_dict(), termination_reason=st.termination or "unknown",
            budget={"exploration_used": st.used_units, "exploration_limit": cfg.exploration_budget_units, "final_phase_units": st.final_phase_units},
            llm_activity={"function_calls": st.llm_function_calls, "turns": st.turns,
                          "mcp_tools_called_by_llm": sorted({c.tool for c in gem_calls if c.tool in SIM_TOOLS | {"generate_payment_scenarios", "get_payment_experiment_results", "get_payment_transaction_trace"}}),
                          "llm_shortlist": (st.finish or {}).get("shortlist"), "llm_rationale": (st.finish or {}).get("rationale")},
            llm_explanation=explanation, notes=st.notes)
