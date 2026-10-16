import pytest
from fixture_yf import FixtureTicker
from pydantic import ValidationError

from financeagents.data import market as market_data
from financeagents.data import tools
from financeagents.schemas import AnalystView, MarketSnapshot, RiskMetrics

TICKERS = ["AAPL", "KO", "ITC.NS", "RELIANCE.NS", "BND", "NIFTYBEES.NS"]


@pytest.fixture(autouse=True)
def fx(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", FixtureTicker)


@pytest.mark.parametrize("ticker", TICKERS)
def test_fixture_payloads_validate(ticker):
    info = tools.fetch_info(ticker)
    md = tools.get_market_data(ticker, info=info)
    MarketSnapshot.model_validate(md)
    AnalystView.model_validate(tools.get_analyst_estimates(ticker, info=info))
    RiskMetrics.model_validate(tools.compute_risk(ticker))


def test_dividend_yield_is_percent():
    # KO pays roughly 2-3% a year; a fraction (0.02) would mean the unit assumption broke.
    assert 1.0 < tools.get_market_data("KO")["dividend_yield_pct"] < 6.0


def test_unexpected_or_non_numeric_fields_rejected():
    md = tools.get_market_data("AAPL")
    with pytest.raises(ValidationError):
        MarketSnapshot.model_validate({**md, "surprise": 1})
    with pytest.raises(ValidationError):
        MarketSnapshot.model_validate({**md, "current_price": "N/A"})
