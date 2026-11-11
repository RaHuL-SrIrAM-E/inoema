"""Display-time redaction for traces and tool I/O. Defence in depth: the persistence layers already scrub secrets at write time;
this layer scrubs again at read time (secret-named keys, literal env secrets, common credential shapes) and truncates long values."""
from __future__ import annotations

import os
import re
from typing import Any

from storage.payment_agent_runs import REDACTED, scrub

_ENV_SECRETS = ("GEMINI_API_KEY", "GOOGLE_API_KEY", "PAYPAL_CLIENT_SECRET", "PAYPAL_CLIENT_ID")
_PATTERNS = [re.compile(p) for p in (r"AIza[0-9A-Za-z_\-]{20,}", r"AQ\.[A-Za-z0-9_\-]{20,}", r"Bearer\s+[A-Za-z0-9._\-]{8,}",
                                     r"A21[A-Za-z0-9_\-]{20,}")]
_EXTRA_SECRET_KEYS = re.compile(r"(access_token|id_token|refresh_token|client_secret|client_id|password|nonce)", re.I)


def env_secrets() -> list[str]:
    return [v for k in _ENV_SECRETS if (v := os.environ.get(k)) and len(v) >= 6]


def _mask_patterns(s: str) -> str:
    for p in _PATTERNS:
        s = p.sub(REDACTED, s)
    return s


def sanitize(obj: Any, *, extra_secrets: tuple[str, ...] = (), max_str: int = 4000) -> Any:
    """Return a copy of `obj` that is safe to display."""
    secrets = [*env_secrets(), *extra_secrets]

    def walk(o: Any) -> Any:
        if isinstance(o, dict):
            return {k: (REDACTED if isinstance(k, str) and _EXTRA_SECRET_KEYS.search(k) else walk(v)) for k, v in scrub(o, secrets).items()}
        if isinstance(o, (list, tuple)):
            return [walk(v) for v in o]
        if isinstance(o, str):
            o = _mask_patterns(o)
            return o if len(o) <= max_str else o[:max_str] + f"… [truncated {len(o) - max_str} chars]"
        return o

    return walk(scrub(obj, secrets))
