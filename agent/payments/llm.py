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

    async def generate(self, contents: list[types.Content], config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                return await asyncio.wait_for(
                    self.client.aio.models.generate_content(model=self.model, contents=contents, config=config), self.timeout_s + 5)
            except genai_errors.APIError as e:
                last = e
                if getattr(e, "code", None) not in RETRYABLE_CODES:
                    raise LLMError(f"Gemini API error {getattr(e, 'code', '?')}: {str(e)[:300]}") from e
            except asyncio.TimeoutError as e:
                last = e
            if attempt < self.retries:
                await self._sleep(min(2 ** attempt, 8))
        raise LLMError(f"Gemini API failed after {self.retries + 1} attempts: {type(last).__name__}: {str(last)[:300]}")
