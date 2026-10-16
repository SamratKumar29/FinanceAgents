"""Offline test fixtures: no network, no .env, scripted LLM client."""
import os
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "financeagents" / "ui" / "app.py")

os.environ["FINANCEAGENTS_SKIP_DOTENV"] = "1"
for _k in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL", "FALLBACK_MODELS", "LLM_MAX_ATTEMPTS", "LLM_TIMEOUT_S"):
    os.environ.pop(_k, None)


@pytest.fixture(autouse=True)
def fresh_market_cache(monkeypatch):
    from financeagents.data import market as market_data

    market_data.clear_cache()
    monkeypatch.setattr(market_data, "sleep", lambda s: None)
    yield
    market_data.clear_cache()


@pytest.fixture(autouse=True)
def no_web_search(monkeypatch):
    """Web search would hit the network; tests that need it patch tools.web_search explicitly."""
    from financeagents.data import tools
    from financeagents.errors import DataUnavailableError

    def offline(query, max_results=5):
        raise DataUnavailableError("web search disabled in tests")

    monkeypatch.setattr(tools, "web_search", offline)


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def guard(*args, **kwargs):
        raise RuntimeError("network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", guard)
    monkeypatch.setattr(socket, "create_connection", guard)


def response(content, model="m", finish_reason="stop"):
    msg = SimpleNamespace(content=content)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)], model=model,
                           usage=None)


class FakeLLMClient:
    """Mimics ``client.chat.completions.create``. ``script`` maps model -> list of outcomes.

    An outcome is an Exception (raised), a string (content), or a ready response object.
    Every request is recorded in ``calls``.
    """

    def __init__(self, script=None, default="OK"):
        self.script = {k: list(v) for k, v in (script or {}).items()}
        self.default = default
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        queue = self.script.get(model)
        outcome = queue.pop(0) if queue else self.default
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, str):
            return response(outcome, model=model)
        return outcome

    def factory(self, _settings):
        return self


@pytest.fixture
def fake_llm():
    return FakeLLMClient


@pytest.fixture
def patch_llm(monkeypatch):
    """Route every ``llm.complete`` call through a FakeLLMClient."""
    from financeagents import llm

    def install(client):
        monkeypatch.setattr(llm, "make_client", client.factory)
        original = llm.complete

        def patched(settings, messages, **kw):
            kw["client_factory"] = client.factory
            return original(settings, messages, **kw)

        from financeagents import pipeline, resolver
        from financeagents.portfolio import service

        for m in (llm, pipeline, resolver, service):
            if hasattr(m, "complete"):
                monkeypatch.setattr(m, "complete", patched)
        return client

    return install
