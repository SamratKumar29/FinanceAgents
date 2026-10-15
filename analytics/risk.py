"""
analytics/risk.py -- Pure, market-aware risk metrics over daily close prices.

All functions take pandas Series indexed by timestamps and do no I/O.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Market:
    code: str
    benchmark: str
    benchmark_name: str
    # Annual risk-free rate. These are *configured assumptions*, not live data:
    # override with RISK_FREE_RATE_<CODE> (e.g. RISK_FREE_RATE_US=0.042).
    default_risk_free_rate: Optional[float]


MARKETS: dict[str, Market] = {
    "US": Market("US", "^GSPC", "S&P 500", 0.04),
    "IN": Market("IN", "^NSEI", "NIFTY 50", 0.065),
    "UK": Market("UK", "^FTSE", "FTSE 100", None),
    "HK": Market("HK", "^HSI", "Hang Seng", None),
    "KR": Market("KR", "^KS11", "KOSPI", None),
    "FR": Market("FR", "^FCHI", "CAC 40", None),
}

_SUFFIX_TO_MARKET = {".NS": "IN", ".BO": "IN", ".L": "UK", ".HK": "HK", ".KS": "KR", ".PA": "FR"}


def market_for_ticker(ticker: str) -> Market:
    upper = (ticker or "").upper()
    for suffix, code in _SUFFIX_TO_MARKET.items():
        if upper.endswith(suffix):
            return MARKETS[code]
    return MARKETS["US"]


def risk_free_rate(market: Market, env=None) -> tuple[float, str]:
    """Return (rate, source description)."""
    env = os.environ if env is None else env
    override = env.get(f"RISK_FREE_RATE_{market.code}")
    if override:
        return float(override), f"RISK_FREE_RATE_{market.code} environment override"
    if market.default_risk_free_rate is not None:
        return market.default_risk_free_rate, "configured default assumption (not live data)"
    return 0.0, "not configured for this market; 0 assumed"


def to_daily(prices: pd.Series) -> pd.Series:
    """Index by calendar date (tz dropped) so different exchanges align on the same day."""
    s = prices.dropna().copy()
    idx = pd.DatetimeIndex(s.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    s.index = idx.normalize()
    return s[~s.index.duplicated(keep="last")].sort_index()


def daily_returns(prices: pd.Series) -> pd.Series:
    return to_daily(prices).pct_change(fill_method=None).dropna()


def annualized_volatility(returns: pd.Series) -> float:
    if len(returns) < 2:
        raise ValueError("need at least 2 returns")
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(returns: pd.Series, rf_annual: float) -> Optional[float]:
    vol = annualized_volatility(returns)
    if vol == 0:
        return None
    return float((returns.mean() * TRADING_DAYS - rf_annual) / vol)


def max_drawdown(returns: pd.Series) -> float:
    """Most negative peak-to-trough decline of the cumulative return path (e.g. -0.25)."""
    wealth = pd.concat([pd.Series([1.0]), (1 + returns.reset_index(drop=True)).cumprod()], ignore_index=True)
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def beta(returns: pd.Series, bench_returns: pd.Series, min_obs: int = 20) -> tuple[Optional[float], int]:
    """Beta from one covariance matrix (consistent ddof). Returns (beta or None, overlapping obs)."""
    joined = pd.concat([returns, bench_returns], axis=1, join="inner").dropna()
    n = len(joined)
    if n < min_obs:
        return None, n
    cov = np.cov(joined.iloc[:, 0], joined.iloc[:, 1], ddof=1)
    if cov[1, 1] == 0:
        return None, n
    return float(cov[0, 1] / cov[1, 1]), n


def risk_report(prices: pd.Series, bench_prices: Optional[pd.Series], market: Market, env=None) -> dict:
    returns = daily_returns(prices)
    if len(returns) < 2:
        raise ValueError("insufficient price history")
    rf, rf_source = risk_free_rate(market, env)
    b, overlap = (None, 0)
    if bench_prices is not None and len(bench_prices):
        b, overlap = beta(returns, daily_returns(bench_prices))
    sharpe = sharpe_ratio(returns, rf)
    return {
        "annualized_volatility_pct": round(annualized_volatility(returns) * 100, 2),
        "sharpe_ratio": None if sharpe is None else round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown(returns) * 100, 2),
        "beta": None if b is None else round(b, 2),
        "benchmark": market.benchmark,
        "benchmark_name": market.benchmark_name,
        "beta_observations": overlap,
        "risk_free_rate": rf,
        "risk_free_rate_source": rf_source,
        "observations": len(returns),
        "start_date": returns.index[0].date().isoformat(),
        "end_date": returns.index[-1].date().isoformat(),
    }
