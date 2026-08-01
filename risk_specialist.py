"""
risk_specialist.py -- computes real quantitative risk metrics from
historical price data via yfinance: annualized volatility, Sharpe ratio,
max drawdown, and beta against a benchmark. The LLM interprets these
numbers in plain language; it does not compute or guess at them.
"""
import numpy as np
import yfinance as yf
from openai import OpenAI
import config

SYSTEM_PROMPT = """You are a Risk Management Specialist. You are given
computed quantitative risk metrics -- explain what they mean in plain
language and what risks they imply for someone considering this position.
Do not recompute or second-guess the numbers, interpret them. Do not
issue a confident buy/sell call -- that is the CIO's job."""

BENCHMARK_TICKER = "^GSPC"  # S&P 500 -- swap for "^NSEI" (NIFTY 50) for an Indian benchmark
RISK_FREE_RATE = 0.065      # approx. India 10Y G-Sec yield -- adjust if you want US-based instead


def compute_risk_metrics(ticker: str) -> dict:
    stock_hist = yf.Ticker(ticker).history(period="1y")["Close"]
    bench_hist = yf.Ticker(BENCHMARK_TICKER).history(period="1y")["Close"]

    if stock_hist.empty:
        return {"error": f"No data found for '{ticker}'."}

    returns = stock_hist.pct_change().dropna()
    bench_returns = bench_hist.pct_change().dropna()

    volatility_annual = returns.std() * np.sqrt(252) * 100
    sharpe = (returns.mean() * 252 - RISK_FREE_RATE) / (returns.std() * np.sqrt(252))

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min() * 100

    aligned_stock, aligned_bench = returns.align(bench_returns, join="inner")
    if len(aligned_stock) > 10:
        beta = np.cov(aligned_stock, aligned_bench)[0][1] / np.var(aligned_bench)
    else:
        beta = None

    return {
        "annualized_volatility_pct": round(volatility_annual, 2),
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "beta_vs_benchmark": round(beta, 2) if beta is not None else "insufficient data",
    }


def run(ticker: str) -> str:
    metrics = compute_risk_metrics(ticker)
    if "error" in metrics:
        return metrics["error"]

    client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Explain what these risk metrics mean for {ticker}:\n\n{metrics}"},
        ],
    )
    return response.choices[0].message.content
