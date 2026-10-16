"""ML-01: deterministic allocator properties, on volatilities from recorded Yahoo fixtures."""
import itertools
import json

import pytest
from conftest import FakeLLMClient
from fixture_yf import FixtureTicker

from financeagents.data import market as market_data
from financeagents.errors import DataUnavailableError
from financeagents.llm import Settings
from financeagents.portfolio import service as pa
from financeagents.portfolio.allocator import (
    BUCKET_WEIGHTS,
    allocate_slots,
    build_allocation,
    load_universe,
    validate_allocation,
)

UNIVERSE = {i.ticker: i for i in load_universe()}


def fixture_vol(ticker):
    return pa.yahoo_volatility(ticker)


@pytest.fixture(autouse=True)
def fixture_yf(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", FixtureTicker)


def profile(**kw):
    p = {"base_currency": "USD", "budget": 10_000.0, "budget_type": "lumpsum", "risk_tolerance": "moderate",
         "investment_horizon": "5 years", "num_positions": 5, "preferred_markets": ["us"]}
    p.update(kw)
    return p


def test_every_universe_instrument_has_fixture_volatility():
    missing = [t for t in UNIVERSE if fixture_vol(t) is None]
    assert missing == []


MARKET_OPTIONS = [["us"], ["india"], ["us", "india"], []]
CASES = list(itertools.product(BUCKET_WEIGHTS, MARKET_OPTIONS, [3, 4, 5, 8, 12, 15], [999.99, 100_000.0]))


@pytest.mark.parametrize("risk,markets,n,budget", CASES)
def test_allocation_invariants(risk, markets, n, budget):
    out = pa.suggest_portfolio(profile(risk_tolerance=risk, preferred_markets=markets, num_positions=n,
                                       budget=budget), None, volatility=fixture_vol)
    pos = out["positions"]
    validate_allocation(pos, budget, markets)  # tickers in universe, markets, sums, non-negative
    assert all(p["allocation_pct"] >= 0 and p["amount"] >= 0 for p in pos)
    assert sum(p["allocation_pct"] for p in pos) == pytest.approx(100, abs=0.01)
    assert sum(p["amount"] for p in pos) == pytest.approx(budget, abs=0.01)
    assert len(pos) <= n
    by_bucket = {}
    for p in pos:
        by_bucket[p["category"]] = by_bucket.get(p["category"], 0) + p["allocation_pct"]
    for bucket, w in BUCKET_WEIGHTS[risk].items():
        assert by_bucket.get(bucket, 0) == pytest.approx(w, abs=0.02)
    assert out["rationale_source"] == "template" and all(p["rationale"] for p in pos)


def test_requested_count_respected_when_universe_allows():
    for n in (3, 5, 8):
        out = pa.suggest_portfolio(profile(num_positions=n), None, volatility=fixture_vol)
        assert len(out["positions"]) == n


def test_deterministic_and_no_llm_needed():
    a = pa.suggest_portfolio(profile(), None, volatility=fixture_vol)
    b = pa.suggest_portfolio(profile(), None, volatility=fixture_vol)
    assert a == b


def test_inverse_volatility_within_bucket():
    pos = build_allocation("conservative", 4, ["us"], fixture_vol)
    etfs = [p for p in pos if p["category"] == "etf"]
    assert len(etfs) == 2
    low, high = sorted(etfs, key=lambda p: p["volatility_pct"])
    assert low["allocation_pct"] > high["allocation_pct"]
    assert low["allocation_pct"] * low["volatility_pct"] == pytest.approx(
        high["allocation_pct"] * high["volatility_pct"], rel=1e-3)


def test_unavailable_instruments_are_skipped_and_weight_renormalised():
    def vol(t):
        return None if UNIVERSE[t].bucket == "speculative" else fixture_vol(t)

    pos = build_allocation("moderate", 5, ["us"], vol)
    assert {p["category"] for p in pos} == {"etf", "growth"}
    assert sum(p["allocation_pct"] for p in pos) == pytest.approx(100)


def test_no_data_at_all_raises_typed_error():
    with pytest.raises(DataUnavailableError):
        pa.suggest_portfolio(profile(), None, volatility=lambda t: None)


def test_allocate_slots():
    w = BUCKET_WEIGHTS["aggressive"]
    assert allocate_slots(10, w, {"etf": 9, "growth": 9, "speculative": 9}) == {"etf": 2, "growth": 5, "speculative": 3}
    assert sum(allocate_slots(3, w, {"etf": 9, "growth": 9, "speculative": 9}).values()) == 3
    assert allocate_slots(10, w, {"etf": 1, "growth": 2, "speculative": 1}) == {"etf": 1, "growth": 2, "speculative": 1}


def test_llm_writes_rationales_only(monkeypatch):
    base = pa.suggest_portfolio(profile(), None, volatility=fixture_vol)
    texts = {p["ticker"]: f"About {p['ticker']}." for p in base["positions"]}
    client = FakeLLMClient(default=json.dumps(texts))
    original = pa.complete
    monkeypatch.setattr(pa, "complete", lambda s, m, **kw: original(s, m, client_factory=client.factory, **kw))
    out = pa.suggest_portfolio(profile(), Settings(api_key="k", model="m"), volatility=fixture_vol)
    assert out["rationale_source"] == "llm"
    assert [p["rationale"] for p in out["positions"]] == [texts[p["ticker"]] for p in out["positions"]]
    strip = [{k: v for k, v in p.items() if k != "rationale"} for p in out["positions"]]
    assert strip == [{k: v for k, v in p.items() if k != "rationale"} for p in base["positions"]]


def test_llm_failure_still_yields_valid_allocation(monkeypatch):
    client = FakeLLMClient(default="not json")
    original = pa.complete
    monkeypatch.setattr(pa, "complete", lambda s, m, **kw: original(s, m, client_factory=client.factory, **kw))
    out = pa.suggest_portfolio(profile(), Settings(api_key="k", model="m"), volatility=fixture_vol)
    assert out["rationale_source"] == "template"
    validate_allocation(out["positions"], 10_000.0, ["us"])


def test_extract_json_object_is_not_greedy():
    assert pa.extract_json_object('Here: {"a": 1} and {"b": 2}') == {"a": 1}


def test_summary_reports_actual_mix_and_asset_classes():
    out = pa.suggest_portfolio(profile(risk_tolerance="conservative", num_positions=5), None, volatility=fixture_vol)
    assert "etf 60%" in out["summary"] and "bond" in out["summary"]
    assert {p["asset_class"] for p in out["positions"]} <= {"equity", "bond", "gold"}


def test_summary_admits_missing_bucket():
    def vol(t):
        return None if UNIVERSE[t].bucket == "speculative" else fixture_vol(t)

    out = pa.suggest_portfolio(profile(num_positions=5), None, volatility=vol)
    assert "No eligible instruments for: speculative" in out["summary"] and "speculative" not in out["summary"].split(
        "No eligible")[0]
