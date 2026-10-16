import math

import numpy as np
import pandas as pd
import pytest
from conftest import APP_PATH

from financeagents import cli as main
from financeagents.data import market as market_data
from financeagents.data import tools
from financeagents.errors import DataUnavailableError, LLMAuthError


class FakeTicker:
    def __init__(self, symbol, info=None, empty=False):
        self.symbol = symbol
        self.info = info if info is not None else {
            "longName": "Apple Inc.", "currency": "USD", "financialCurrency": "USD", "currentPrice": 200,
            "trailingPE": float("nan"), "marketCap": "3E12", "earningsQuarterlyGrowth": 0.1, "revenueGrowth": None,
        }
        self.empty = empty
        idx = pd.bdate_range("2025-01-01", periods=260)
        frame = pd.DataFrame({"Total Revenue": [1e9], "Net Income": [np.nan]}).T
        frame.columns = [pd.Timestamp("2025-09-30")]
        self.income_stmt = frame
        self.balance_sheet = pd.DataFrame()
        self.cashflow = None
        self._hist = pd.DataFrame({"Close": 100 + np.arange(260, dtype=float)}, index=idx)

    def history(self, period="1y"):
        return pd.DataFrame({"Close": []}) if self.empty else self._hist


def _walk(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


@pytest.fixture
def fake_yf(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", lambda s: FakeTicker(s))


def test_payloads_typed_with_as_of_and_no_nan_strings(fake_yf):
    info = tools.fetch_info("AAPL")
    payloads = [tools.get_market_data("AAPL", info=info), tools.compute_risk("AAPL"),
                tools.get_financials("AAPL", info=info), tools.get_analyst_estimates("AAPL", info=info)]
    for p in payloads:
        assert p["as_of"]
        for v in _walk(p):
            assert v is None or isinstance(v, (int, float, str, bool))
            assert v not in ("nan", "N/A", "None")
            if isinstance(v, float):
                assert not math.isnan(v)
    md, risk, fin, est = payloads
    assert md["pe_ratio_trailing"] is None and md["market_cap"] == 3e12 and md["financial_currency"] == "USD"
    assert risk["benchmark"] == "^GSPC" and risk["observations"] == 259
    assert fin["income_statement"]["net_income"] is None and fin["balance_sheet"] is None
    assert "earnings_estimate_current_quarter" not in est
    assert est["trailing_quarterly_earnings_growth_yoy_pct"] == 10.0


def test_empty_history_raises_data_unavailable(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", lambda s: FakeTicker(s, empty=True))
    with pytest.raises(DataUnavailableError):
        tools.get_market_data("XXXX", info={})
    with pytest.raises(DataUnavailableError):
        tools.compute_risk("XXXX")


def test_same_currency_conversion_needs_no_network():
    assert tools.convert_currency(10, "usd", "USD")["converted"] == 10


def test_cli_exits_nonzero_with_user_message(monkeypatch, capsys):
    def boom(query, settings):
        raise LLMAuthError("401 from provider")

    monkeypatch.setattr(main, "run_analysis", boom)
    assert main.main(["apple?"]) == 1
    assert "LLM authentication failed" in capsys.readouterr().err


def test_streamlit_key_field_is_never_prefilled(monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_API_KEY", "SERVER-SENTINEL-KEY")
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    key_input = next(t for t in at.text_input if t.label == "API Key")
    assert key_input.value == ""
    assert "SERVER-SENTINEL-KEY" not in str(at.get("text_input"))
    assert any("server key" in c.value for c in at.caption)


def test_streamlit_onboarding_generates_portfolio_once(monkeypatch):
    pytest.importorskip("streamlit.testing.v1")
    from streamlit.testing.v1 import AppTest

    from financeagents.portfolio import service as portfolio_allocator

    calls = []

    def fake_suggest(profile, settings):
        calls.append(dict(profile))
        return {"positions": [{"ticker": "NIFTYBEES.NS", "company": "Nifty ETF", "allocation_pct": 100.0,
                               "amount": profile["budget"], "rationale": "r", "risk_level": "low"}],
                "summary": "s", "disclaimer": "d", "total_amount": profile["budget"], "currency": "INR",
                "served_model": "fake"}

    monkeypatch.setattr(portfolio_allocator, "suggest_portfolio", fake_suggest)
    monkeypatch.setenv("LLM_API_KEY", "SERVER-SENTINEL-KEY")
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    next(b for b in at.button if "Investment Mode" in b.label).click().run()
    answers = ["India", "5 lakh", "lumpsum", "retirement", "moderate", "10 years", "4", "india", "none"]
    for a in answers:
        at.chat_input[0].set_value(a).run()
        assert not at.exception
    assert any("Positions: 4" in m.value for m in at.markdown), "num_positions question must be asked"
    assert len(calls) == 1 and calls[0]["budget"] == 500_000
    at.run()  # any later rerun (e.g. clicking Download) must not regenerate
    at.run()
    assert len(calls) == 1


def test_all_nan_closes_are_data_unavailable(monkeypatch):
    class NanTicker(FakeTicker):
        def history(self, period="1y"):
            return pd.DataFrame({"Close": [np.nan, np.nan]}, index=pd.bdate_range("2026-01-01", periods=2))

    monkeypatch.setattr(market_data.yf, "Ticker", lambda s: NanTicker(s))
    with pytest.raises(DataUnavailableError):
        tools.get_market_data("X", info={})
    with pytest.raises(DataUnavailableError):
        tools.convert_currency(1, "USD", "INR")


def test_first_usable_value_falls_back_when_primary_is_none():
    assert tools._first({"currentPrice": None, "regularMarketPrice": 12.5}, "currentPrice", "regularMarketPrice") == 12.5


def test_minor_currency_units(monkeypatch):
    class FX(FakeTicker):
        def history(self, period="1y"):
            return pd.DataFrame({"Close": [1.25]}, index=pd.bdate_range("2026-01-01", periods=1))

    monkeypatch.setattr(market_data.yf, "Ticker", lambda s: FX(s))
    assert tools.convert_currency(1000, "GBp", "GBP")["converted"] == 10.0
    assert tools.convert_currency(1000, "GBp", "USD")["converted"] == pytest.approx(12.5)
