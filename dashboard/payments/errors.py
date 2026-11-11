"""Turn backend exceptions / error texts into something a merchant can act on. No Streamlit imports."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from agent.payments.models import MissingCredentialsError


@dataclass(frozen=True)
class ErrorInfo:
    kind: str  # MISSING_CREDENTIALS | QUOTA_EXHAUSTED | AUTH | MODEL_UNAVAILABLE | TIMEOUT | MCP_ERROR | BUSY | UNKNOWN
    title: str
    detail: str
    advice: str
    retry_after: Optional[str] = None


def _retry_after(text: str) -> Optional[str]:
    m = re.search(r"~?\s*([0-9.]+)\s*h\b", text)
    if m:
        return f"about {float(m.group(1)):.1f} hours"
    m = re.search(r"retryDelay['\": ]+(\d+(?:\.\d+)?)s", text) or re.search(r"waiting\s+(\d+(?:\.\d+)?)s", text)
    if m:
        secs = float(m.group(1))
        return f"about {secs / 3600:.1f} hours" if secs >= 3600 else f"about {secs / 60:.0f} minutes" if secs >= 120 else f"{secs:.0f} seconds"
    return None


def classify_error(err: "BaseException | str") -> ErrorInfo:
    if isinstance(err, MissingCredentialsError):
        return ErrorInfo("MISSING_CREDENTIALS", "Gemini is not configured", str(err),
                         "Set GEMINI_API_KEY (and optionally GEMINI_MODEL) in .env, then restart the dashboard. No fake model is ever used as a fallback.")
    text = err if isinstance(err, str) else f"{type(err).__name__}: {err}"
    low = text.lower()
    if "resource_exhausted" in low or "quota" in low or "rate limit" in low or " 429" in low or "429 " in low:
        return ErrorInfo("QUOTA_EXHAUSTED", "Gemini quota exhausted", text[:1200],
                         "The API key's quota is used up (free-tier keys allow only a few requests per day per model). Wait for the quota to reset, use a billed key, "
                         "or switch to Demonstration mode to inspect the last genuine execution. The dashboard will not substitute a scripted model.", _retry_after(text))
    if "no longer available" in low or "not_found" in low or "404" in low:
        return ErrorInfo("MODEL_UNAVAILABLE", "Gemini model unavailable for this key", text[:1200],
                         "Set GEMINI_MODEL to a model your key can use (the API's error message may recommend one). The dashboard never picks another model for you.")
    if "api key not valid" in low or "api_key_invalid" in low or "permission_denied" in low or "unauthenticated" in low or " 401" in low or " 403" in low:
        return ErrorInfo("AUTH", "Gemini rejected the credentials", text[:1200], "Check GEMINI_API_KEY (it may be invalid, revoked or restricted).")
    if "deadline_exceeded" in low or "timeout" in low or "timed out" in low or " 504" in low:
        return ErrorInfo("TIMEOUT", "Request timed out", text[:1200], "The model or network did not answer in time. Try again, or configure a different model.")
    if "mcp" in low or "tool" in low and "server" in low:
        return ErrorInfo("MCP_ERROR", "MCP simulation server problem", text[:1200], "Check that the simulation server module starts: `python -m spikes.mcp_sim.server`.")
    return ErrorInfo("UNKNOWN", "Unexpected error", text[:1200], "See the agent trace and logs for details.")
