"""Deterministic final selection and recommendation-confidence assessment.

Pure functions over MCP result JSON. The LLM does not participate: selection is the top FEASIBLE row of
a fixed-seed comparison computed (scores, hard constraints) by the simulation server's objective code.
Given identical result JSON, the outputs here are identical.

Four distinct notions are kept separate (never collapsed into one number):
  * repeatability          - re-running the identical request reproduces identical results
  * statistical uncertainty- paired score margin vs the runner-up, with 95% CI (simulation sampling error only)
  * assumption sensitivity - does the winner survive changes to the ASSUMED world / weights?
  * scenario robustness    - does it win under other workload scenarios?
None of these is confidence in real-world performance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from agent.payments.models import SIM_DISCLAIMER


@dataclass(frozen=True)
class Rejection:
    policy: str
    kind: str  # 'INFEASIBLE' | 'LOWER_SCORE'
    reasons: tuple[str, ...]
    score: float
    score_gap_to_winner: Optional[float] = None
    gap_ci95: Optional[tuple[float, float]] = None
    gap_significant: Optional[bool] = None

    def to_dict(self) -> dict[str, Any]:
        return {"policy": self.policy, "kind": self.kind, "reasons": list(self.reasons), "score": self.score,
                "score_gap_to_winner": self.score_gap_to_winner, "gap_ci95": list(self.gap_ci95) if self.gap_ci95 else None,
                "gap_significant": self.gap_significant}


@dataclass(frozen=True)
class Selection:
    winner: Optional[str]
    ranking: tuple[dict[str, Any], ...]
    rejected: tuple[Rejection, ...]
    n_feasible: int


def select_winner(final: dict[str, Any]) -> Selection:
    """Best feasible policy from a compare_recovery_policies result; every other candidate gets a reason."""
    ranking = final["ranking"]
    feasible = [r for r in ranking if r["feasible"]]
    winner = feasible[0]["policy"] if feasible else None
    diffs = final.get("paired_score_difference_vs_best", {})
    rejected = []
    for r in ranking:
        if r["policy"] == winner:
            continue
        if not r["feasible"]:
            rejected.append(Rejection(r["policy"], "INFEASIBLE", tuple(r["violations"]), r["score_mean"]))
        else:
            d = diffs.get(r["policy"], {})
            ci = d.get("ci95")
            rejected.append(Rejection(r["policy"], "LOWER_SCORE", ("feasible but scored lower than the winner",), r["score_mean"],
                                      d.get("mean_difference"), tuple(ci) if ci else None, d.get("ci_excludes_zero")))
    return Selection(winner, tuple(ranking), tuple(rejected), len(feasible))


@dataclass(frozen=True)
class ConfidenceAssessment:
    label: str  # 'HIGH' | 'MEDIUM' | 'LOW' | 'NONE'
    reasons: tuple[str, ...]
    repeatability: dict[str, Any]
    statistical_uncertainty: dict[str, Any]
    assumption_sensitivity: dict[str, Any]
    scenario_robustness: dict[str, Any]
    disclaimer: str = SIM_DISCLAIMER

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "reasons": list(self.reasons), "repeatability": self.repeatability,
                "statistical_uncertainty": self.statistical_uncertainty, "assumption_sensitivity": self.assumption_sensitivity,
                "scenario_robustness": self.scenario_robustness, "disclaimer": self.disclaimer}


def results_identical(a: dict[str, Any], b: dict[str, Any]) -> bool:
    """Two compare results describe identical simulation outcomes (ignores run ids)."""
    return a["ranking"] == b["ranking"] and a["policies"] == b["policies"]


def assess_confidence(winner: Optional[str], final: dict[str, Any], repeat: Optional[dict[str, Any]],
                      sensitivity: Optional[dict[str, Any]], scenario_checks: list[dict[str, Any]]) -> ConfidenceAssessment:
    """Rule-based, deterministic. The thresholds are heuristics, documented in docs/AUTONOMOUS_PAYMENT_AGENT.md."""
    if winner is None:
        empty = {"verified": None}
        return ConfidenceAssessment("NONE", ("no feasible candidate satisfied the hard constraints",), empty, empty, empty, empty)

    reasons: list[str] = []
    # repeatability
    rep_ok = bool(repeat is not None and results_identical(final, repeat))
    repeatability = {"verified": rep_ok if repeat is not None else None,
                     "detail": "identical request re-run; ranking and all metrics matched" if rep_ok else
                               ("not checked" if repeat is None else "re-run DIFFERED (non-determinism!)")}

    # statistical uncertainty: paired margin winner vs runner-up (feasible only)
    feasible = [r for r in final["ranking"] if r["feasible"]]
    wrow = feasible[0]
    if len(feasible) < 2:
        stat = {"label": "NO_FEASIBLE_COMPETITOR", "winner_score_ci95": wrow["score_ci95"],
                "detail": "only one candidate satisfied the hard constraints, so superiority over feasible alternatives is untested"}
    else:
        runner = feasible[1]["policy"]
        d = final["paired_score_difference_vs_best"][runner]
        stat = {"label": "CLEAR_MARGIN" if d["ci_excludes_zero"] else "NOT_SEPARABLE", "runner_up": runner,
                "margin_mean": d["mean_difference"], "margin_ci95": d["ci95"], "winner_score_ci95": wrow["score_ci95"],
                "n_trials": final["n_trials"], "detail": "paired score difference over simulation trials; sampling error only"}

    # assumption sensitivity: share of sweep points where THIS winner is still the winner
    if sensitivity is None:
        sens = {"label": "NOT_ASSESSED"}
    else:
        pts = sensitivity["points"]
        stable = [p for p in pts if p["winner"] == winner]
        share = len(stable) / len(pts) if pts else 1.0
        flips = [{"parameter": p["parameter"], "value": p["value"], "new_winner": p["winner"]} for p in pts if p["winner"] != winner]
        sens = {"label": "ROBUST" if share >= 0.9 else "MODERATE" if share >= 0.6 else "FRAGILE", "winner_stable_share": share,
                "n_points": len(pts), "flip_conditions": flips,
                "most_sensitive_parameters": sensitivity["most_sensitive_parameters"],
                "detail": "share of one-at-a-time sweep points (ASSUMED ranges) where the selected policy remains best"}

    # scenario robustness
    if scenario_checks:
        wins = [c for c in scenario_checks if c["winner"] == winner]
        rob = {"label": "WINS_ALL" if len(wins) == len(scenario_checks) else "WINS_SOME" if wins else "WINS_NONE",
               "share": len(wins) / len(scenario_checks), "scenarios": scenario_checks}
    else:
        rob = {"label": "NOT_ASSESSED", "share": None, "scenarios": []}

    # combine (deterministic rules)
    if not rep_ok:
        reasons.append("repeatability not verified")
    if stat["label"] == "NOT_SEPARABLE":
        reasons.append("margin over the runner-up is not statistically separable from zero")
    if stat["label"] == "NO_FEASIBLE_COMPETITOR":
        reasons.append("no feasible competitor to compare against")
    if sens["label"] in ("MODERATE", "FRAGILE"):
        reasons.append(f"recommendation changes under some assumption variations ({sens['label']})")
    if sens["label"] == "NOT_ASSESSED":
        reasons.append("assumption sensitivity not assessed")
    if rob["label"] in ("WINS_SOME", "WINS_NONE"):
        reasons.append("selected policy does not win every alternative scenario")
    if rob["label"] == "NOT_ASSESSED":
        reasons.append("scenario robustness not assessed")

    high = (rep_ok and stat["label"] == "CLEAR_MARGIN" and sens["label"] == "ROBUST" and rob["label"] == "WINS_ALL")
    medium = (rep_ok and stat["label"] in ("CLEAR_MARGIN", "NO_FEASIBLE_COMPETITOR") and sens["label"] in ("ROBUST", "MODERATE")
              and rob["label"] in ("WINS_ALL", "WINS_SOME"))
    label = "HIGH" if high else "MEDIUM" if medium else "LOW"
    if label == "HIGH":
        reasons.append("repeatable, clear margin, robust to assumption sweeps, wins all alternative scenarios (within the simulation)")
    return ConfidenceAssessment(label, tuple(reasons), repeatability, stat, sens, rob)
