"""
schemas.py -- Typed tool payloads. Every tool output is validated against one of these models, so
numbers are floats or None (never "nan"/"N/A" strings) and each payload carries ``as_of``.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict

Num = Optional[float]


class _Payload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ticker: str
    as_of: str


class MarketSnapshot(_Payload):
    last_close_date: str
    company: Optional[str]
    sector: Optional[str]
    industry: Optional[str]
    currency: Optional[str]
    financial_currency: Optional[str]
    current_price: Num
    market_cap: Num
    pe_ratio_trailing: Num
    pe_ratio_forward: Num
    week52_low: Num
    week52_high: Num
    dividend_yield_pct: Num  # yfinance >= 1.x reports dividendYield in percent (AAPL 0.31 = 0.31%)
    price_change_6mo_pct: Num
    business_summary: Optional[str]


class RiskMetrics(_Payload):
    annualized_volatility_pct: float
    sharpe_ratio: Num
    max_drawdown_pct: float
    beta: Num
    benchmark: str
    benchmark_name: str
    beta_observations: int
    risk_free_rate: float
    risk_free_rate_source: str
    observations: int
    start_date: str
    end_date: str


class Statement(BaseModel):
    model_config = ConfigDict(extra="allow")  # line items vary by statement

    period_end: str


class Financials(_Payload):
    financial_currency: Optional[str]
    income_statement: Optional[Statement]
    balance_sheet: Optional[Statement]
    cash_flow: Optional[Statement]


class AnalystView(_Payload):
    currency: Optional[str]
    analyst_target_mean: Num
    analyst_target_high: Num
    analyst_target_low: Num
    analyst_recommendation: Optional[str]
    analyst_count: Num
    trailing_quarterly_earnings_growth_yoy_pct: Num
    revenue_growth_yoy_pct: Num
    peg_ratio: Num
    price_to_book: Num
    price_to_sales: Num


def validated(model: type[BaseModel], data: dict) -> dict:
    return model.model_validate(data).model_dump()
