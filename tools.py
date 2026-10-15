"""
tools.py -- Market data, risk, financials, news and FX helpers used by the analysis pipeline.

Every function returns plain JSON-serialisable dicts with numbers (or ``None``), never
"nan"/"N/A" strings, and raises ``DataUnavailableError`` instead of returning error text.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Optional

import yfinance as yf

from analytics.risk import market_for_ticker, risk_report
from errors import DataUnavailableError

try:
    from ddgs import DDGS
except ImportError:  # older package name
    from duckduckgo_search import DDGS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _num(value: Any) -> Optional[float]:
    """Coerce to float; None for missing/NaN/non-numeric."""
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else f


def _str(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value.strip() else None


def fetch_info(ticker: str) -> dict:
    """Fetch ``Ticker.info`` once; callers pass the result to the functions below."""
    try:
        info = yf.Ticker(ticker).info
    except Exception as e:  # yfinance raises a variety of HTTP/parse errors
        raise DataUnavailableError(f"Could not fetch info for {ticker}: {e}") from e
    return info or {}


def get_market_data(ticker: str, info: Optional[dict] = None) -> dict:
    """Price, valuation, 52-week range and 6-month performance."""
    info = fetch_info(ticker) if info is None else info
    try:
        hist = yf.Ticker(ticker).history(period="6mo")
    except Exception as e:
        raise DataUnavailableError(f"Could not fetch price history for {ticker}: {e}") from e
    if hist is None or hist.empty:
        raise DataUnavailableError(f"No price data for '{ticker}'; it may be delisted or invalid.")
    close = hist["Close"].dropna()
    change_6mo = (close.iloc[-1] / close.iloc[0] - 1) * 100 if len(close) > 1 else None
    return {
        "ticker": ticker,
        "as_of": _now(),
        "last_close_date": close.index[-1].date().isoformat(),
        "company": _str(info.get("longName")) or _str(info.get("shortName")),
        "sector": _str(info.get("sector")),
        "industry": _str(info.get("industry")),
        "currency": _str(info.get("currency")),
        "financial_currency": _str(info.get("financialCurrency")),
        "current_price": _num(info.get("currentPrice", info.get("regularMarketPrice"))),
        "market_cap": _num(info.get("marketCap")),
        "pe_ratio_trailing": _num(info.get("trailingPE")),
        "pe_ratio_forward": _num(info.get("forwardPE")),
        "52_week_low": _num(info.get("fiftyTwoWeekLow")),
        "52_week_high": _num(info.get("fiftyTwoWeekHigh")),
        # Units vary across yfinance versions; passed through as reported by Yahoo.
        "dividend_yield_as_reported": _num(info.get("dividendYield")),
        "6month_price_change_pct": None if change_6mo is None else round(float(change_6mo), 2),
        "business_summary": (_str(info.get("longBusinessSummary")) or "")[:500] or None,
    }


def compute_risk(ticker: str) -> dict:
    """1-year volatility, Sharpe, max drawdown and beta vs the ticker's home-market benchmark."""
    market = market_for_ticker(ticker)
    try:
        prices = yf.Ticker(ticker).history(period="1y")["Close"]
    except Exception as e:
        raise DataUnavailableError(f"Could not fetch price history for {ticker}: {e}") from e
    if prices is None or prices.dropna().empty:
        raise DataUnavailableError(f"No price data for '{ticker}'.")
    try:
        bench = yf.Ticker(market.benchmark).history(period="1y")["Close"]
    except Exception:
        bench = None  # beta becomes None; other metrics still valid
    try:
        report = risk_report(prices, bench, market)
    except ValueError as e:
        raise DataUnavailableError(f"Cannot compute risk for {ticker}: {e}") from e
    return {"ticker": ticker, "as_of": _now(), **report}


def get_financials(ticker: str, info: Optional[dict] = None) -> dict:
    """Latest annual income statement, balance sheet and cash flow lines."""
    try:
        stock = yf.Ticker(ticker)
        income, balance, cashflow = stock.income_stmt, stock.balance_sheet, stock.cashflow
    except Exception as e:
        raise DataUnavailableError(f"Could not fetch financials for {ticker}: {e}") from e

    def latest(frame, fields: dict[str, str]) -> Optional[dict]:
        if frame is None or frame.empty:
            return None
        col = frame.iloc[:, 0]
        period = frame.columns[0]
        out = {"period_end": period.date().isoformat() if hasattr(period, "date") else str(period)}
        out.update({k: _num(col.get(src)) for k, src in fields.items()})
        return out

    data = {
        "ticker": ticker,
        "as_of": _now(),
        "financial_currency": _str((info or {}).get("financialCurrency")),
        "income_statement": latest(income, {
            "total_revenue": "Total Revenue", "net_income": "Net Income",
            "operating_income": "Operating Income", "gross_profit": "Gross Profit", "eps_diluted": "Diluted EPS",
        }),
        "balance_sheet": latest(balance, {
            "total_assets": "Total Assets", "total_debt": "Total Debt",
            "total_equity": "Stockholders Equity", "cash_and_equivalents": "Cash And Cash Equivalents",
        }),
        "cash_flow": latest(cashflow, {
            "operating_cash_flow": "Operating Cash Flow", "free_cash_flow": "Free Cash Flow",
            "capital_expenditure": "Capital Expenditure",
        }),
    }
    if not any(data[k] for k in ("income_statement", "balance_sheet", "cash_flow")):
        raise DataUnavailableError(f"No financial statements for '{ticker}'.")
    return data


def get_analyst_estimates(ticker: str, info: Optional[dict] = None) -> dict:
    """Analyst price targets, consensus, trailing growth and valuation ratios."""
    info = fetch_info(ticker) if info is None else info
    return {
        "ticker": ticker,
        "as_of": _now(),
        "currency": _str(info.get("currency")),
        "analyst_target_mean": _num(info.get("targetMeanPrice")),
        "analyst_target_high": _num(info.get("targetHighPrice")),
        "analyst_target_low": _num(info.get("targetLowPrice")),
        "analyst_recommendation": _str(info.get("recommendationKey")),
        "analyst_count": _num(info.get("numberOfAnalystOpinions")),
        # Yahoo's earningsQuarterlyGrowth / revenueGrowth are trailing YoY growth, not estimates.
        "trailing_quarterly_earnings_growth_yoy": _num(info.get("earningsQuarterlyGrowth")),
        "revenue_growth_yoy": _num(info.get("revenueGrowth")),
        "peg_ratio": _num(info.get("pegRatio", info.get("trailingPegRatio"))),
        "price_to_book": _num(info.get("priceToBook")),
        "price_to_sales": _num(info.get("priceToSalesTrailing12Months")),
    }


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """Top DuckDuckGo results as ``{title, url, snippet}``. Content is untrusted."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        raise DataUnavailableError(f"Web search failed: {e}") from e
    return [
        {"title": r.get("title"), "url": r.get("href"), "snippet": (r.get("body") or "")[:300]}
        for r in results
    ]


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert using the latest Yahoo FX close."""
    src, dst = from_currency.upper().strip(), to_currency.upper().strip()
    if src == dst:
        return {"amount": amount, "from": src, "to": dst, "rate": 1.0, "converted": amount, "as_of": _now()}
    try:
        hist = yf.Ticker(f"{src}{dst}=X").history(period="5d")
    except Exception as e:
        raise DataUnavailableError(f"Could not fetch FX rate {src}/{dst}: {e}") from e
    if hist is None or hist.empty:
        raise DataUnavailableError(f"No FX rate for {src}/{dst}.")
    rate = float(hist["Close"].dropna().iloc[-1])
    return {"amount": amount, "from": src, "to": dst, "rate": round(rate, 6),
            "converted": round(amount * rate, 2), "as_of": _now()}
