import json
import logging

import pytest
from conftest import FakeLLMClient, response
from fixture_yf import FixtureTicker

from financeagents import llm, tracing
from financeagents import pipeline as graph
from financeagents import resolver as ticker_resolver
from financeagents.data import market as market_data
from financeagents.llm import Settings
from financeagents.state import initial_state

SENTINEL = "sk-or-v1-SENTINELKEY0123456789"


@pytest.fixture
def traced_env(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", FixtureTicker)
    usage = type("U", (), {"model_dump": lambda self: {"prompt_tokens": 100, "completion_tokens": 20,
                                                        "total_tokens": 120}})()
    extract = response('{"company_name": "Apple", "ticker_if_known": "AAPL", "context": "earnings"}', model="m")
    extract.usage = usage
    brief = response("Apple trades at a P/E near its history.", model="m")
    brief.usage = usage
    client = FakeLLMClient({"m": [extract, brief]})
    original = llm.complete

    def patched(s, m, **kw):
        kw["client_factory"] = client.factory
        return original(s, m, **kw)

    for mod in (graph, ticker_resolver):
        monkeypatch.setattr(mod, "complete", patched)
    monkeypatch.setattr(ticker_resolver, "search_yahoo", lambda name: [])
    return client


def test_trace_records_every_llm_call_step_and_totals(traced_env):
    settings = Settings(api_key=SENTINEL, model="m")
    out = graph.run_analysis(graph.build_graph(settings), initial_state("How is AAPL doing before earnings?"))
    tr = out["trace"]
    assert len(tr["run_id"]) == 12
    assert [c["task"] for c in tr["llm_calls"]] == ["ticker_extraction", "synthesis"]
    assert tr["total_tokens"] == 240 and tr["models_used"] == ["m"]
    names = {s["name"] for s in tr["steps"]}
    assert {"resolve_ticker", "fetch_info", "market_data", "risk_metrics", "financials", "analyst_view",
            "synthesis"} <= names
    assert all(s["ms"] >= 0 for s in tr["steps"]) and tr["total_latency_ms"] > 0


def test_failed_llm_call_is_traced(monkeypatch):
    from test_llm import status_error

    client = FakeLLMClient({"m": [status_error(429)], "m2": [status_error(503)]})
    with tracing.start_trace() as tr, pytest.raises(llm.LLMUnavailableError):
        llm.complete(Settings(api_key="k", model="m", fallback_models=("m2",)), [], client_factory=client.factory,
                     task="synthesis")
    call = tr.summary()["llm_calls"][0]
    assert call["error"] == "all models unavailable" and len(call["attempts"]) == 2


def test_api_key_never_appears_in_logs_or_trace(traced_env, caplog):
    handler = logging.StreamHandler()
    handler.setFormatter(tracing.JsonFormatter())
    records = []
    handler.emit = lambda rec: records.append(handler.format(rec))
    logging.getLogger().addHandler(handler)
    try:
        with caplog.at_level(logging.INFO):
            settings = Settings(api_key=SENTINEL, model="m")
            out = graph.run_analysis(graph.build_graph(settings), initial_state("AAPL"))
            logging.getLogger("x").warning("provider said: invalid api_key=%s Bearer %s", SENTINEL, SENTINEL)
            with tracing.start_trace() as tr, pytest.raises(RuntimeError):
                with tr.step("boom"):
                    raise RuntimeError(f"Authorization: Bearer {SENTINEL}")
    finally:
        logging.getLogger().removeHandler(handler)
    blob = json.dumps(out, default=str) + json.dumps(tr.summary()) + "\n".join(records)
    assert SENTINEL not in blob and "SENTINELKEY" not in blob
    assert any("[REDACTED]" in r for r in records)
    assert all(json.loads(r)["level"] for r in records)


def test_settings_repr_hides_key():
    assert SENTINEL not in repr(Settings(api_key=SENTINEL))
