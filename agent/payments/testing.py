"""Test doubles for the LLM. NOT Gemini.

ScriptedBackend returns SDK-typed responses (google.genai.types) from a script, so unit tests and the
`--scripted` demo exercise the real adapter/loop code without a network call. Any report produced with it
carries llm.provider == 'SCRIPTED_TEST_DOUBLE' and must never be described as Gemini output.
"""
from __future__ import annotations

from typing import Any, Callable, Union

from google.genai import types

Call = tuple[str, dict[str, Any]]
Turn = Union[str, list[Call], Callable[[list[types.Content]], Union[str, list[Call]]]]


def function_call_response(calls: list[Call], text: str = "") -> types.GenerateContentResponse:
    parts = ([types.Part(text=text)] if text else []) + [types.Part.from_function_call(name=n, args=a) for n, a in calls]
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])


def text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))])


def last_function_responses(contents: list[types.Content]) -> list[dict[str, Any]]:
    """Payloads of the most recent function_response parts (lets scripted turns adapt to earlier results)."""
    for c in reversed(contents):
        got = [p.function_response.response for p in (c.parts or []) if p.function_response is not None]
        if got:
            return got
    return []


class ScriptedBackend:
    provider = "SCRIPTED_TEST_DOUBLE"

    def __init__(self, turns: list[Turn], explanation: str = "(scripted explanation - not an LLM)"):
        self.turns, self.explanation = list(turns), explanation
        self.seen: list[list[types.Content]] = []
        self.configs: list[types.GenerateContentConfig] = []

    async def generate(self, contents: list[types.Content], config: types.GenerateContentConfig) -> types.GenerateContentResponse:
        self.seen.append(list(contents))
        self.configs.append(config)
        if not config.tools:  # explanation call
            return text_response(self.explanation)
        if not self.turns:
            return text_response("(script exhausted)")
        turn = self.turns.pop(0)
        if callable(turn):
            turn = turn(contents)
        return text_response(turn) if isinstance(turn, str) else function_call_response(turn)


class FailingBackend:
    provider = "SCRIPTED_TEST_DOUBLE"

    def __init__(self, exc: Exception):
        self.exc = exc

    async def generate(self, contents, config):
        raise self.exc
