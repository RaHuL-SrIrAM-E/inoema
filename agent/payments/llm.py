"""Gemini backend (google-genai SDK). Thin on purpose: the agent depends on the LLMBackend protocol so unit
tests can supply SDK-typed scripted responses, while production uses GeminiBackend against the real API."""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable, Optional, Protocol

from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from agent.payments.models import AgentConfig, AgentError, MissingCredentialsError

RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
MAX_ERROR_CHARS = 2000


def _find_detail(node: Any, suffix: str) -> Optional[dict]:
    """Depth-first search of an API error payload for a google.rpc detail whose @type ends with `suffix`."""
    if isinstance(node, dict):
        if str(node.get("@type", "")).endswith(suffix):
            return node
        for v in node.values():
            r = _find_detail(v, suffix)
            if r is not None:
                return r
    elif isinstance(node, list):
        for v in node:
            r = _find_detail(v, suffix)
            if r is not None:
                return r
    return None


def rate_limit_info(e: Exception) -> tuple[Optional[float], Optional[str]]:
    """(server-advised retry delay in seconds, quota description) parsed from a Gemini API error, if present."""
    payload = getattr(e, "details", None) or getattr(e, "response_json", None)
    delay = None
    info = _find_detail(payload, "RetryInfo")
    if info and isinstance(info.get("retryDelay"), str):
        try:
            delay = float(info["retryDelay"].rstrip("s"))
        except ValueError:
            delay = None
    quota = _find_detail(payload, "QuotaFailure")
    desc = None
    if quota and quota.get("violations"):
        v = quota["violations"][0]
        desc = f"{v.get('description', '')} [quotaId={v.get('quotaId')}, limit={v.get('quotaValue')}, model={(v.get('quotaDimensions') or {}).get('model')}]"
    return delay, desc


class LLMError(AgentError):
    pass


class LLMBackend(Protocol):
    provider: str

    async def generate(self, contents: list[types.Content], config: types.GenerateContentConfig) -> types.GenerateContentResponse: ...


class GeminiBackend:
    provider = "GEMINI_API"

    def __init__(self, cfg: AgentConfig, client: Optional[genai.Client] = None,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep):
        if client is None:
            if not cfg.api_key:
                raise MissingCredentialsError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=cfg.api_key, http_options=types.HttpOptions(timeout=int(cfg.llm_timeout_s * 1000)))
        self.client, self.model, self.retries, self._sleep = client, cfg.model, cfg.llm_retries, sleep
        self.timeout_s = cfg.llm_timeout_s
        self.max_rate_limit_wait_s = cfg.max_rate_limit_wait_s

    async def generate(self, contents: list[types.Content], config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                return await asyncio.wait_for(
                    self.client.aio.models.generate_content(model=self.model, contents=contents, config=config), self.timeout_s + 5)
            except genai_errors.APIError as e:
                last = e
                if getattr(e, "code", None) not in RETRYABLE_CODES:
                    raise LLMError(f"Gemini API error {getattr(e, 'code', '?')}: {str(e)[:MAX_ERROR_CHARS]}") from e
                delay, quota = rate_limit_info(e)
                if delay is not None:
                    if delay > self.max_rate_limit_wait_s:  # e.g. a per-day quota: retrying now is pointless
                        raise LLMError(f"Gemini rate limit/quota exhausted ({getattr(e, 'code', '?')}); server advises waiting {delay:.0f}s "
                                       f"(~{delay / 3600:.1f} h), longer than the {self.max_rate_limit_wait_s:.0f}s this client will wait. "
                                       f"{quota or ''} Raw: {str(e)[:MAX_ERROR_CHARS]}") from e
                    if attempt < self.retries:
                        await self._sleep(delay + 1.0)  # honour the server-advised delay
                        continue
            except asyncio.TimeoutError as e:
                last = e
            if attempt < self.retries:
                await self._sleep(min(2 ** attempt, 8))
        raise LLMError(f"Gemini API failed after {self.retries + 1} attempts: {type(last).__name__}: {str(last)[:MAX_ERROR_CHARS]}")
