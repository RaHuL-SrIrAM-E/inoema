"""OFFLINE: the live Gemini tests must be impossible to run by accident. This only inspects the skip conditions; it never calls Gemini."""
import importlib

import pytest


def _skip_conditions(monkeypatch, *, opt_in: bool, key: bool):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "INOEMA_ALLOW_LIVE_GEMINI"):
        monkeypatch.delenv(k, raising=False)
    if key:
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-not-used")
    if opt_in:
        monkeypatch.setenv("INOEMA_ALLOW_LIVE_GEMINI", "1")
    mod = importlib.reload(importlib.import_module("tests.integration.test_gemini_live"))
    return [m.args[0] for m in mod.pytestmark if m.name == "skipif"]


def test_live_gemini_tests_are_skipped_even_with_a_key_unless_explicitly_opted_in(monkeypatch):
    assert any(_skip_conditions(monkeypatch, opt_in=False, key=True)), "live Gemini tests would RUN with only a key present"


def test_live_gemini_tests_are_skipped_with_opt_in_but_no_key(monkeypatch):
    assert any(_skip_conditions(monkeypatch, opt_in=True, key=False))


def test_live_gemini_tests_run_only_with_opt_in_and_a_key(monkeypatch):
    assert not any(_skip_conditions(monkeypatch, opt_in=True, key=True))


@pytest.fixture(autouse=True)
def _restore_module(monkeypatch):
    yield
    monkeypatch.undo()
    importlib.reload(importlib.import_module("tests.integration.test_gemini_live"))
