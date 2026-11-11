"""Data-source labelling. EVERY metric shown in the Payments workspace carries exactly one of these sources, and the UI never
combines values from different sources into one number."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class DataSource(str, Enum):
    SIMULATED = "SIMULATED"  # produced by the INOEMA payment simulation engine under assumed distributions
    AGENT_DERIVED = "AGENT_DERIVED"  # decided/explained by the Gemini agent + deterministic evaluator (over simulated data)
    SANDBOX_ACTUAL = "SANDBOX_ACTUAL"  # observed from real PayPal Sandbox API calls
    SYSTEM = "SYSTEM"  # live status of a local component (not a payment metric)


TEAL, DEEP, CORAL, GREY = "#2CB6A4", "#102C2B", "#F07868", "#65716F"
SOURCE_STYLE: dict[DataSource, dict[str, str]] = {
    DataSource.SIMULATED: {"label": "SIMULATED", "color": TEAL, "mark": "◆",
                           "hint": "Simulation engine output under ASSUMED distributions. Not PayPal data; not real-world performance."},
    DataSource.AGENT_DERIVED: {"label": "AGENT-DERIVED", "color": DEEP, "mark": "✦",
                               "hint": "Chosen or explained by the Gemini agent / deterministic evaluator, from simulated results."},
    DataSource.SANDBOX_ACTUAL: {"label": "SANDBOX · ACTUAL", "color": CORAL, "mark": "●",
                                "hint": "Observed from real PayPal SANDBOX API calls. Single observations, never blended with simulation."},
    DataSource.SYSTEM: {"label": "SYSTEM", "color": GREY, "mark": "○", "hint": "Live status of a local component."},
}


@dataclass(frozen=True)
class Metric:
    label: str
    value: Any
    source: DataSource
    caption: str = ""
    delta: Optional[str] = None

    @property
    def display(self) -> str:
        if self.value is None:
            return "—"
        return str(self.value)


def badge_html(source: DataSource, extra: str = "") -> str:
    st_ = SOURCE_STYLE[source]
    return (f"<span title=\"{st_['hint']}\" style=\"display:inline-block; font-size:0.68rem; font-weight:700; letter-spacing:0.05em; "
            f"color:{st_['color']}; border:1px solid {st_['color']}55; background:{st_['color']}12; border-radius:999px; padding:0.08rem 0.55rem;\">"
            f"{st_['mark']}&nbsp;{st_['label']}{(' · ' + extra) if extra else ''}</span>")
