"""
tools.py -- LangGraph @tool definitions for the agentic CIO agent.

Each tool wraps a specific capability (market data, risk, web search, etc.)
and is decorated with @tool for automatic binding to the LLM's tool-calling interface.

The CIO agent can call any of these tools during its ReAct reasoning loop.
"""
import re
import json
import numpy as np
import yfinance as yf
from langchain_core.tools import tool
try:
    from ddgs import DDGS
except ImportError:
    from duckduckgo_search import DDGS


# ---------------------------------------------------------------------------
# Tool 1: Ticker Search / Resolution
# ---------------------------------------------------------------------------
@tool
def search_ticker(query: str) -> str:
    """Resolve a company name or natural language reference to a stock ticker symbol.
    Use this when the user mentions a company name (like 'Take Two', 'Nvidia')
    instead of a ticker symbol. Returns the ticker, company name, and exchange."""
    from ticker_resolver import resolve_ticker
    result = resolve_ticker(query)
    if result:
        return json.dumps(result.to_dict(), indent=2)
    return f"Could not resolve a ticker for '{query}'. Try using the explicit ticker symbol (e.g., TTWO, RELIANCE.NS)."


# ---------------------------------------------------------------------------
# Tool 2: Market Data
# ---------------------------------------------------------------------------
@tool
def get_market_data(ticker: str) -> str:
    """Get current market data for a stock ticker: price, P/E ratio, market cap,
    52-week range, analyst target, 6-month performance, and business summary.
    Use this to understand a company's current valuation and recent performance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="6mo")

        if hist.empty:
            return f"No price data found for '{ticker}'. It may be delisted or invalid."

        change_6mo = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100

        data = {
            "ticker": ticker,
            "company": info.get("longName", info.get("shortName", "N/A")),
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "current_price": info.get("currentPrice", info.get("regularMarketPrice", "N/A")),
            "currency": info.get("currency", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio_trailing": info.get("trailingPE", "N/A"),
            "pe_ratio_forward": info.get("forwardPE", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "analyst_target_mean": info.get("targetMeanPrice", "N/A"),
            "analyst_target_high": info.get("targetHighPrice", "N/A"),
            "analyst_target_low": info.get("targetLowPrice", "N/A"),
            "dividend_yield": info.get("dividendYield", "N/A"),
            "6month_price_change_pct": round(change_6mo, 2),
            "business_summary": str(info.get("longBusinessSummary", "N/A"))[:500],
        }
        return json.dumps(data, indent=2, default=str)
    except Exception as e:
        return f"Error fetching market data for '{ticker}': {str(e)}"


# ---------------------------------------------------------------------------
# Tool 3: Risk Metrics
# ---------------------------------------------------------------------------
@tool
def compute_risk(ticker: str) -> str:
    """Compute quantitative risk metrics for a stock: annualized volatility,
    Sharpe ratio, maximum drawdown, and beta vs S&P 500.
    Use this to assess the risk profile of an investment."""
    try:
        BENCHMARK = "^GSPC"
        RISK_FREE_RATE = 0.065

        stock_hist = yf.Ticker(ticker).history(period="1y")["Close"]
        bench_hist = yf.Ticker(BENCHMARK).history(period="1y")["Close"]

        if stock_hist.empty:
            return f"No price data found for '{ticker}'."

        returns = stock_hist.pct_change().dropna()
        bench_returns = bench_hist.pct_change().dropna()

        volatility = round(float(returns.std() * np.sqrt(252) * 100), 2)
        sharpe = round(float((returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252))), 2)

        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = round(float(drawdown.min() * 100), 2)

        aligned_stock, aligned_bench = returns.align(bench_returns, join="inner")
        if len(aligned_stock) > 10:
            beta = round(float(np.cov(aligned_stock, aligned_bench)[0][1] / np.var(aligned_bench)), 2)
        else:
            beta = "insufficient data"

        data = {
            "ticker": ticker,
            "annualized_volatility_pct": volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_drawdown,
            "beta_vs_sp500": beta,
            "risk_free_rate_used": RISK_FREE_RATE,
            "period": "1 year",
        }
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"Error computing risk metrics for '{ticker}': {str(e)}"


# ---------------------------------------------------------------------------
# Tool 4: Financial Statements
# ---------------------------------------------------------------------------
@tool
def get_financials(ticker: str) -> str:
    """Get key financial statement data for a stock: revenue, net income,
    total assets, total debt, free cash flow. Use this for fundamental analysis."""
    try:
        stock = yf.Ticker(ticker)

        income = stock.income_stmt
        balance = stock.balance_sheet
        cashflow = stock.cashflow

        data = {"ticker": ticker}

        # Latest income statement
        if income is not None and not income.empty:
            latest = income.iloc[:, 0]
            data["income_statement"] = {
                "period": str(income.columns[0].date()) if hasattr(income.columns[0], 'date') else str(income.columns[0]),
                "total_revenue": str(latest.get("Total Revenue", "N/A")),
                "net_income": str(latest.get("Net Income", "N/A")),
                "operating_income": str(latest.get("Operating Income", "N/A")),
                "gross_profit": str(latest.get("Gross Profit", "N/A")),
                "eps_diluted": str(latest.get("Diluted EPS", "N/A")),
            }

        # Latest balance sheet
        if balance is not None and not balance.empty:
            latest = balance.iloc[:, 0]
            data["balance_sheet"] = {
                "total_assets": str(latest.get("Total Assets", "N/A")),
                "total_debt": str(latest.get("Total Debt", "N/A")),
                "total_equity": str(latest.get("Stockholders Equity", "N/A")),
                "cash_and_equivalents": str(latest.get("Cash And Cash Equivalents", "N/A")),
            }

        # Latest cash flow
        if cashflow is not None and not cashflow.empty:
            latest = cashflow.iloc[:, 0]
            data["cash_flow"] = {
                "operating_cash_flow": str(latest.get("Operating Cash Flow", "N/A")),
                "free_cash_flow": str(latest.get("Free Cash Flow", "N/A")),
                "capital_expenditure": str(latest.get("Capital Expenditure", "N/A")),
            }

        return json.dumps(data, indent=2, default=str)
    except Exception as e:
        return f"Error fetching financials for '{ticker}': {str(e)}"


# ---------------------------------------------------------------------------
# Tool 5: Web Search (DuckDuckGo)
# ---------------------------------------------------------------------------
@tool
def web_search(query: str) -> str:
    """Search the web for real-time information using DuckDuckGo.
    Use this to find news, catalysts, events, earnings dates, product launches,
    or any context that affects a stock's outlook. Returns top 5 results."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return f"No web results found for '{query}'."

        output = []
        for i, r in enumerate(results, 1):
            output.append(f"{i}. **{r.get('title', 'N/A')}**")
            output.append(f"   URL: {r.get('href', 'N/A')}")
            output.append(f"   {r.get('body', 'N/A')[:200]}")
            output.append("")

        return "\n".join(output)
    except Exception as e:
        return f"Web search error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 6: Currency Conversion
# ---------------------------------------------------------------------------
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using live exchange rates.
    Use this when suggesting investments in a different currency than the user's preference.
    Example: convert_currency(10000, "USD", "INR")"""
    from_currency = from_currency.upper().strip()
    to_currency = to_currency.upper().strip()

    if from_currency == to_currency:
        return json.dumps({"amount": amount, "from": from_currency, "to": to_currency, "converted": amount, "rate": 1.0})

    try:
        pair = f"{from_currency}{to_currency}=X"
        rate_data = yf.Ticker(pair).history(period="1d")
        if rate_data.empty:
            return f"Could not fetch exchange rate for {from_currency} to {to_currency}."
        rate = float(rate_data["Close"].iloc[-1])
        converted = round(amount * rate, 2)
        return json.dumps({
            "original_amount": amount,
            "from_currency": from_currency,
            "to_currency": to_currency,
            "exchange_rate": round(rate, 4),
            "converted_amount": converted,
        })
    except Exception as e:
        return f"Currency conversion error: {str(e)}"


# ---------------------------------------------------------------------------
# Tool 7: Analyst Estimates
# ---------------------------------------------------------------------------
@tool
def get_analyst_estimates(ticker: str) -> str:
    """Get analyst price targets, earnings estimates, and recommendation summary
    for a stock. Use this to understand market consensus on a stock."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        data = {
            "ticker": ticker,
            "analyst_target_mean": info.get("targetMeanPrice", "N/A"),
            "analyst_target_high": info.get("targetHighPrice", "N/A"),
            "analyst_target_low": info.get("targetLowPrice", "N/A"),
            "analyst_recommendation": info.get("recommendationKey", "N/A"),
            "analyst_count": info.get("numberOfAnalystOpinions", "N/A"),
            "earnings_estimate_current_quarter": str(info.get("earningsQuarterlyGrowth", "N/A")),
            "revenue_estimate_current_quarter": str(info.get("revenueGrowth", "N/A")),
            "peg_ratio": info.get("pegRatio", "N/A"),
            "price_to_book": info.get("priceToBook", "N/A"),
            "price_to_sales": info.get("priceToSalesTrailing12Months", "N/A"),
        }
        return json.dumps(data, indent=2, default=str)
    except Exception as e:
        return f"Error fetching analyst estimates for '{ticker}': {str(e)}"


# ---------------------------------------------------------------------------
# All tools list (for easy binding)
# ---------------------------------------------------------------------------
ALL_TOOLS = [
    search_ticker,
    get_market_data,
    compute_risk,
    get_financials,
    web_search,
    convert_currency,
    get_analyst_estimates,
]

TOOL_NAMES = [t.name for t in ALL_TOOLS]