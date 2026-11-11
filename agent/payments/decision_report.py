"""Structured decision report. Every number comes from MCP tool results; the optional LLM explanation is
clearly labelled and never used for selection."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

ASSUMPTIONS_AND_LIMITATIONS = (
    "All distributions (failure rates, category mix, outage persistence, buyer response, costs) are ASSUMPTIONS, not empirical statistics.",
    "Results describe simulated merchant-side recovery behaviour; they are not PayPal data and do not model PayPal's authorization or risk systems.",
    "Confidence intervals quantify simulation sampling error only, not uncertainty about the assumed parameters.",
    "Sensitivity sweeps are one-at-a-time over illustrative ranges; interactions between parameters are not explored.",
    "The failure-category rules were cross-checked only against PayPal's public docs, not against live Sandbox behaviour.",
    "Constraints are enforced from the objective text via explicit parsing rules; unrecognised constraint phrases are listed as warnings and NOT enforced.",
    "The recommendation has not been validated against the PayPal Sandbox; no PayPal execution is performed by this agent.",
)


@dataclass
class DecisionReport:
    execution_id: str
    llm: dict[str, Any]
    merchant_objective: dict[str, Any]
    initial_policy: dict[str, Any]
    candidates_explored: list[dict[str, Any]]
    experiments: dict[str, Any]
    simulation_configurations: list[dict[str, Any]]
    final_evaluation: dict[str, Any]
    feasibility_decisions: list[dict[str, Any]]
    sensitivity_findings: dict[str, Any]
    selected_policy: Optional[dict[str, Any]]
    rejected_alternatives: list[dict[str, Any]]
    confidence: dict[str, Any]
    termination_reason: str
    budget: dict[str, Any]
    llm_activity: dict[str, Any]
    llm_explanation: Optional[str] = None
    assumptions_and_limitations: tuple[str, ...] = ASSUMPTIONS_AND_LIMITATIONS
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["assumptions_and_limitations"] = list(self.assumptions_and_limitations)
        return d

    def to_markdown(self) -> str:
        L: list[str] = []
        sel, conf = self.selected_policy, self.confidence
        L.append(f"# Decision report  (execution {self.execution_id})")
        L.append(f"LLM: **{self.llm['provider']}** model `{self.llm['model']}` | termination: `{self.termination_reason}` | "
                 f"LLM function calls: {self.llm_activity['function_calls']} over {self.llm_activity['turns']} turn(s)")
        L.append("\n> SIMULATED results under ASSUMED distributions - not PayPal data, not real-world performance.\n")
        L.append("## Merchant objective")
        L.append(self.merchant_objective["text"])
        L.append("Constraints enforced: " + ", ".join(f"{k}={v}" for k, v in self.merchant_objective["constraints"].items()))
        for w in self.merchant_objective["warnings"]:
            L.append(f"- WARNING: {w}")
        ip = self.initial_policy
        L.append(f"\n## Initial policy: `{ip['name']}`  feasible={ip.get('feasible')}  score={ip.get('score')}  recovery_rate={ip.get('recovery_rate')}")
        L.append(f"\n## Experiments: {self.experiments['simulation_calls']} simulation calls "
                 f"({self.experiments['by_gemini']} by LLM, {self.experiments['by_harness']} by harness); "
                 f"{len(self.candidates_explored)} candidates explored; budget used {self.budget['exploration_used']}/{self.budget['exploration_limit']} units")
        L.append("\n## Candidates explored")
        L.append("| candidate | origin | evaluations | ever feasible | best feasible score |\n|---|---|---|---|---|")
        for c in self.candidates_explored:
            L.append(f"| {c['name']} | {c['origin']} | {c['n_observations']} | {c['ever_feasible']} | {c['best_feasible_score']} |")
        fe = self.final_evaluation
        L.append(f"\n## Final deterministic evaluation (scenario `{fe.get('scenario')}`, {fe.get('n_trials')} trials x {fe.get('n_transactions')} txns, seed {fe.get('master_seed')})")
        L.append("| rank | policy | feasible | score | recovery rate [95% CI] | avg recovery min | unnecessary retries |\n|---|---|---|---|---|---|---|")
        for r in fe.get("ranking", []):
            L.append(f"| {r['rank']} | {r['policy']} | {r['feasible']} | {r['score']} | {r['recovery_rate']} {r['recovery_rate_ci95']} | {r['avg_recovery_minutes']} | {r['unnecessary_retries']} |")
        L.append("\n## Selected policy")
        if sel:
            L.append(f"**{sel['name']}** - selected deterministically as the top-scoring FEASIBLE candidate. Definition: {sel['definition']}")
        else:
            L.append("**None** - no candidate satisfied the hard constraints.")
        L.append("\n## Rejected alternatives")
        for r in self.rejected_alternatives:
            L.append(f"- `{r['policy']}` [{r['kind']}]: {'; '.join(r['reasons'])}"
                     + (f" (score gap {r['score_gap_to_winner']:+.4f}, CI {r['gap_ci95']})" if r.get("score_gap_to_winner") is not None else ""))
        sf = self.sensitivity_findings
        L.append("\n## Sensitivity findings")
        if sf.get("label") == "NOT_ASSESSED":
            L.append("Not assessed.")
        else:
            L.append(f"Winner stable in {sf['winner_stable_share']:.0%} of {sf['n_points']} sweep points ({sf['label']}); most sensitive parameters: "
                     f"{sf.get('most_sensitive_parameters') or 'none flip the winner'}")
            for f in sf.get("flip_conditions", [])[:6]:
                L.append(f"- flips to `{f['new_winner']}` when {f['parameter']} = {f['value']}")
        L.append("\n## Recommendation confidence (simulation only)")
        L.append(f"**{conf['label']}** - " + "; ".join(conf["reasons"]))
        L.append(f"- repeatability: {conf['repeatability']}")
        L.append(f"- statistical uncertainty: {conf['statistical_uncertainty'].get('label')} {conf['statistical_uncertainty'].get('margin_ci95', '')}")
        L.append(f"- assumption sensitivity: {conf['assumption_sensitivity'].get('label')}")
        L.append(f"- scenario robustness: {conf['scenario_robustness'].get('label')}")
        L.append(f"- {conf['disclaimer']}")
        if self.llm_explanation:
            L.append(f"\n## LLM explanation (written by {self.llm['provider']}; NOT used for selection)\n{self.llm_explanation}")
        L.append("\n## Assumptions and limitations")
        L.extend(f"- {a}" for a in self.assumptions_and_limitations)
        for n in self.notes:
            L.append(f"- NOTE: {n}")
        return "\n".join(L)
