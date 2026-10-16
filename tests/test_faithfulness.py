import pytest
from fixture_yf import FixtureTicker

from financeagents.analytics.faithfulness import check_brief, extract_claims
from financeagents.data import market as market_data
from financeagents.data import tools


@pytest.fixture
def aapl_payload(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", FixtureTicker)
    info = tools.fetch_info("AAPL")
    return {"ticker": "AAPL", "tool_data": {
        "market_data": tools.get_market_data("AAPL", info=info),
        "risk_metrics": tools.compute_risk("AAPL"),
        "analyst_view": tools.get_analyst_estimates("AAPL", info=info),
    }}


def faithful_brief(p):
    md, risk, av = (p["tool_data"][k] for k in ("market_data", "risk_metrics", "analyst_view"))
    upside = (av["analyst_target_mean"] / md["current_price"] - 1) * 100
    return (
        f"## Apple (AAPL)\n"
        f"1. Shares trade at ${md['current_price']:.2f}, a market cap of about ${md['market_cap'] / 1e12:.2f} trillion "
        f"and a trailing P/E of {md['pe_ratio_trailing']:.1f}.\n"
        f"2. Over 6 months the price moved {md['price_change_6mo_pct']:.1f}%.\n"
        f"3. The mean analyst target of ${av['analyst_target_mean']:.0f} implies {upside:.1f}% upside.\n"
        f"- 1-year volatility is {risk['annualized_volatility_pct']:.1f}% with a max drawdown of "
        f"{abs(risk['max_drawdown_pct']):.1f}% and beta {risk['beta']:.2f} vs the S&P 500 in Q3 2026.\n\n"
        "This is general informational analysis, not personalized financial advice -- consult 1 advisor."
    )


def test_faithful_brief_has_no_unsupported_numbers(aapl_payload):
    res = check_brief(faithful_brief(aapl_payload), aapl_payload)
    assert res["numbers_checked"] >= 9
    assert res["unsupported"] == [] and res["unsupported_rate"] == 0


def test_fabricated_numbers_are_flagged(aapl_payload):
    fabricated = "Revenue grew 47.3% last quarter and the dividend yield is 9.8%. EPS will reach $31.40.\n"
    brief = faithful_brief(aapl_payload).replace("\nThis is general", "\n" + fabricated + "This is general")
    res = check_brief(brief, aapl_payload)
    flagged = {u["raw"] for u in res["unsupported"]}
    assert {"47.3%", "9.8%", "$31.40"} <= flagged
    assert res["unsupported_rate"] > 0


def test_extraction_skips_years_quarters_time_spans_list_markers_and_disclaimer():
    text = ("1. In Q3 2026 over 52-week and 6 month windows, price was 12.5.\n"
            "This is general informational analysis, 99 percent")
    assert [c["raw"] for c in extract_claims(text)] == ["12.5"]


@pytest.mark.parametrize("raw,value", [("$3.2 trillion", 3.2e12), ("₹2,50,000", 250000), ("4.5 crore", 4.5e7),
                                       ("12%", 12), ("1.5bn", 1.5e9)])
def test_units(raw, value):
    assert extract_claims(f"value {raw} here")[0]["value"] == pytest.approx(value)


def test_numbers_from_query_count_as_supported():
    assert check_brief("Ahead of GTA 6 the stock rose 5.2%.", {"x": 5.2}, query="GTA 6")["unsupported"] == []


@pytest.mark.parametrize("text,payload,flagged", [
    ("A 6\u2011month change and a 52\u2011week range of 10.5", {"x": 10.5}, []),           # non-breaking hyphens
    ("P/E above 35x and P/B > 45x", {"pe": 38.2, "pb": 51.0}, []),                          # bounds, true
    ("P/E above 35x", {"pe": 12.0}, ["35"]),                                                # bound, false
    ("debt is about 18% of assets", {"bs": {"period_end": "2026-03-31", "debt": 3.98, "assets": 21.78}}, []),           # same-section ratio
    ("revenue grew 5.9%", {"av": {"revenue_growth_yoy_pct": 105.9}}, ["5.9%"]),             # the NVDA misread
    ("about 15% downside to 170", {"price": 201.23, "target_low": 170.0}, []),              # approx derived gap
    ("exactly 15.0% downside", {"price": 201.23, "target_low": 170.0}, ["15.0%"]),          # precise but wrong
    ("S&P 500 typically ~15-20%", {"vol": 33.21}, ["15", "20%"]),                           # outside knowledge
])
def test_live_run_cases(text, payload, flagged):
    assert [u["raw"] for u in check_brief(text, payload)["unsupported"]] == flagged
