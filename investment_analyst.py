"""
investment_analyst.py -- pulls real market data via yfinance and produces
a research summary: recent performance, fundamentals, and context.
No invented numbers -- everything the LLM discusses comes from real data
fetched first.
"""
import yfinance as yf
from openai import OpenAI
import config

SYSTEM_PROMPT = """You are an Investment Analyst. You research companies
and investment opportunities using real market data provided to you.
Base your analysis ONLY on the data given -- do not invent figures.
Present facts and context clearly. Do not issue a confident buy/sell
recommendation -- that synthesis is the CIO's job, not yours."""


def fetch_data(ticker: str) -> str:
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = stock.history(period="6mo")
    if hist.empty:
        return f"No data found for ticker '{ticker}'. It may be an invalid or delisted symbol."

    change_6mo = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
    return f"""
Ticker: {ticker}
Company: {info.get('longName', 'N/A')}
Sector: {info.get('sector', 'N/A')} / Industry: {info.get('industry', 'N/A')}
Current price: {info.get('currentPrice', 'N/A')}
6-month price change: {change_6mo:.1f}%
Market cap: {info.get('marketCap', 'N/A')}
P/E ratio (trailing): {info.get('trailingPE', 'N/A')}
52-week range: {info.get('fiftyTwoWeekLow', 'N/A')} - {info.get('fiftyTwoWeekHigh', 'N/A')}
Analyst target (mean): {info.get('targetMeanPrice', 'N/A')}
Business summary: {str(info.get('longBusinessSummary', 'N/A'))[:500]}
"""


def run(ticker: str) -> str:
    raw_data = fetch_data(ticker)
    if raw_data.startswith("No data found"):
        return raw_data

    client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this data and summarize the investment context:\n\n{raw_data}"},
        ],
    )
    return response.choices[0].message.content
