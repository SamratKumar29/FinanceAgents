import numpy as np
import pandas as pd
import pytest

from financeagents.analytics.risk import (
    MARKETS,
    annualized_volatility,
    beta,
    daily_returns,
    market_for_ticker,
    max_drawdown,
    risk_free_rate,
    risk_report,
    sharpe_ratio,
)


def prices_from_returns(returns, start="2025-01-01", tz=None):
    idx = pd.bdate_range(start, periods=len(returns) + 1, tz=tz)
    return pd.Series(100 * np.cumprod(np.r_[1.0, 1 + np.asarray(returns)]), index=idx)


def test_beta_of_levered_benchmark_is_exact():
    rng = np.random.default_rng(0)
    bench = pd.Series(rng.normal(0, 0.01, 250))
    b, n = beta(2 * bench, bench)
    assert b == pytest.approx(2.0, abs=1e-9) and n == 250


def test_constant_series_has_zero_vol_and_no_sharpe():
    r = pd.Series([0.001] * 50)
    assert annualized_volatility(r) == pytest.approx(0.0, abs=1e-15)
    assert sharpe_ratio(pd.Series([0.0] * 50), 0.04) is None


def test_known_drawdown_path():
    # 100 -> 120 -> 90 -> 108 : max drawdown = 90/120 - 1 = -25%
    r = pd.Series([0.2, 90 / 120 - 1, 108 / 90 - 1])
    assert max_drawdown(r) == pytest.approx(-0.25)


def test_drawdown_counts_drop_on_first_day():
    assert max_drawdown(pd.Series([-0.1, 0.05])) == pytest.approx(-0.1)


def test_market_selection_by_suffix():
    assert market_for_ticker("RELIANCE.NS").benchmark == "^NSEI"
    assert market_for_ticker("INFY.BO").code == "IN"
    assert market_for_ticker("AAPL").benchmark == "^GSPC"
    assert market_for_ticker("1211.HK").benchmark == "^HSI"


def test_risk_free_override_and_unconfigured_market():
    assert risk_free_rate(MARKETS["US"], {"RISK_FREE_RATE_US": "0.05"})[0] == 0.05
    rate, src = risk_free_rate(MARKETS["UK"], {})
    assert rate == 0.0 and "not configured" in src


def test_cross_timezone_alignment_overlaps():
    """NSE and NYSE bars stamped in different zones must align on calendar dates."""
    rng = np.random.default_rng(1)
    bench_r = rng.normal(0, 0.01, 260)
    stock = prices_from_returns(1.5 * bench_r, tz="Asia/Kolkata")
    bench = prices_from_returns(bench_r, tz="America/New_York")
    rep = risk_report(stock, bench, MARKETS["IN"], env={})
    assert rep["beta_observations"] >= 200
    assert rep["beta"] == pytest.approx(1.5, abs=0.01)
    assert rep["benchmark"] == "^NSEI" and rep["risk_free_rate"] == 0.065
    for key in ("observations", "start_date", "end_date", "risk_free_rate_source"):
        assert key in rep


def test_short_history_rejected_and_missing_benchmark_gives_none_beta():
    with pytest.raises(ValueError):
        risk_report(prices_from_returns([0.01]), None, MARKETS["US"], env={})
    rep = risk_report(prices_from_returns([0.01, -0.02, 0.03]), None, MARKETS["US"], env={})
    assert rep["beta"] is None and rep["observations"] == 3


def test_daily_returns_drop_duplicate_dates_and_nans():
    s = pd.Series([1.0, np.nan, 2.0, 2.0], index=pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03",
                                                                  "2025-01-03"]))
    assert daily_returns(s).tolist() == [1.0]


def _yields(values, end="2026-10-15"):
    return pd.Series(values, index=pd.bdate_range(end=end, periods=len(values), tz="America/Chicago"))


def test_live_rate_used_when_fresh_and_labelled():
    from financeagents.analytics.risk import live_rate_from_yield

    live = live_rate_from_yield(_yields([4.0, 4.088]), "^IRX", today=pd.Timestamp("2026-10-16"))
    assert live[0] == pytest.approx(0.04088) and "^IRX close 4.088% on 2026-10-15 (live)" == live[1]
    assert risk_free_rate(MARKETS["US"], {}, live) == live


def test_stale_or_absurd_live_rate_falls_back_to_default():
    from financeagents.analytics.risk import live_rate_from_yield

    assert live_rate_from_yield(_yields([4.0]), "^IRX", today=pd.Timestamp("2026-12-01")) is None
    assert live_rate_from_yield(_yields([400.0]), "^IRX", today=pd.Timestamp("2026-10-16")) is None
    assert live_rate_from_yield(None, "^IRX") is None
    rate, src = risk_free_rate(MARKETS["US"], {}, None)
    assert rate == 0.04 and "default" in src


def test_env_override_beats_live_rate():
    rate, src = risk_free_rate(MARKETS["US"], {"RISK_FREE_RATE_US": "0.05"}, (0.041, "live"))
    assert rate == 0.05 and "override" in src


def test_india_has_no_live_symbol():
    assert MARKETS["IN"].rf_symbol is None and MARKETS["US"].rf_symbol == "^IRX"


def test_malformed_risk_free_override_is_config_error():
    from financeagents.errors import ConfigError

    with pytest.raises(ConfigError):
        risk_free_rate(MARKETS["US"], {"RISK_FREE_RATE_US": "four percent"})
