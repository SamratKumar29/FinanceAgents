"""Tests against recorded Yahoo Finance payloads (tests/fixtures, see scripts/record_fixtures.py)."""
import json
import math

import pytest
from conftest import FakeLLMClient
from fixture_yf import FixtureTicker, load

import graph
import ticker_resolver
import tools
from errors import DataUnavailableError
from llm import Settings
from state import initial_state

TICKERS = ["AAPL", "TTWO", "NVDA", "RELIANCE.NS", "INFY.NS"]


@pytest.fixture(autouse=True)
def fixture_yf(monkeypatch):
    FixtureTicker.instances.clear()
    monkeypatch.setattr(tools.yf, "Ticker", FixtureTicker)
    monkeypatch.setattr(ticker_resolver.yf, "Ticker", FixtureTicker)


def _scalars(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _scalars(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _scalars(v)
    else:
        yield obj


@pytest.mark.parametrize("ticker", TICKERS)
def test_recorded_payloads_are_typed(ticker):
    info = tools.fetch_info(ticker)
    for p in (tools.get_market_data(ticker, info=info), tools.get_financials(ticker, info=info),
              tools.get_analyst_estimates(ticker, info=info), tools.compute_risk(ticker)):
        assert p["as_of"]
        for v in _scalars(p):
            assert v is None or isinstance(v, (int, float, str, bool))
            assert not (isinstance(v, float) and math.isnan(v))
            assert v not in ("nan", "N/A")


@pytest.mark.parametrize("ticker,benchmark", [("RELIANCE.NS", "^NSEI"), ("INFY.NS", "^NSEI"), ("AAPL", "^GSPC")])
def test_market_aware_beta_on_recorded_data(ticker, benchmark):
    r = tools.compute_risk(ticker)
    assert r["benchmark"] == benchmark
    assert isinstance(r["beta"], float) and r["beta_observations"] >= 200
    assert r["observations"] >= 200 and r["start_date"] < r["end_date"]


def test_invalid_ticker_is_data_unavailable():
    with pytest.raises(DataUnavailableError):
        tools.get_market_data("ZZZZINVALID", info={})
    assert ticker_resolver.lookup_info("ZZZZINVALID") is None


def test_full_analysis_on_fixtures_fetches_info_once_and_grounds_prompt(monkeypatch):
    client = FakeLLMClient(default="BRIEF")
    import llm

    original = llm.complete
    monkeypatch.setattr(graph, "complete", lambda s, m, **kw: original(s, m, client_factory=client.factory, **kw))
    settings = Settings(api_key="k", model="m")

    def resolver(query, s):
        return ticker_resolver.resolve_ticker(query, None, search=lambda name: [])

    out = graph.build_graph(settings, resolver=resolver).invoke(initial_state("What about Reliance Industries?"))
    assert out["ticker"] == "RELIANCE.NS" and out["data_gaps"] == []
    assert sum(t.info_reads for t in FixtureTicker.instances if t.symbol == "RELIANCE.NS") == 2  # resolver + run
    payload = json.loads(client.calls[0]["messages"][1]["content"].split("\n", 1)[1])
    price = payload["tool_data"]["market_data"]["current_price"]
    assert price == load("RELIANCE.NS")["info"].get("currentPrice")
    assert payload["tool_data"]["risk_metrics"]["benchmark"] == "^NSEI"
