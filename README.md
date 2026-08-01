# FinanceAgents

AI-powered investment analysis team — agentic reasoning, real market data, and personalized portfolios.

## What It Does

**Analysis Mode** — Ask about any stock in natural language. The system resolves company names to tickers, fetches real market data, computes risk metrics, searches the web for catalysts, and produces a balanced investment brief.

**Investment Mode** — Complete a short onboarding profile (budget, goals, risk tolerance) and get a personalized portfolio with specific stock/ETF recommendations and allocation amounts. Download your profile as JSON to resume next session.

## Architecture

```
User Input → Router → Ticker Resolver → CIO Agent (ReAct Loop) → Final Brief
                                                  ↓
                              ┌───────────┬──────────┬──────────┐
                              ▼           ▼          ▼          ▼
                         Market Data  Risk Metrics  Financials  Web Search
```

## 7 Agent Tools

| Tool | What It Does |
|---|---|
| search_ticker | LLM + yfinance Search resolves company names to tickers |
| get_market_data | Price, P/E, market cap, 52-week range, analyst targets |
| compute_risk | Volatility, Sharpe ratio, max drawdown, beta |
| get_financials | Income statement, balance sheet, cash flow |
| web_search | DuckDuckGo for news, catalysts, events |
| convert_currency | Live INR/USD exchange rates |
| get_analyst_estimates | Analyst targets, PEG, P/B, P/S ratios |

## Quick Start

```bash
git clone https://github.com/SamratKumar29/FinanceAgents.git
cd FinanceAgents
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your API key
streamlit run app.py
```

Or via CLI:
```bash
python main.py "Should I invest in Apple?"
python main.py --mode investment
```

## Example

Query: "I was thinking of investing in Take Two before GTA 6 launch. Is it advisable?"

- Resolves Take Two → TTWO (NASDAQ)
- Analyst consensus: Strong buy, target $284 (17% upside)
- Risk: 31.6% volatility, -27.7% max drawdown
- Context: GTA 6 identified as primary catalyst

## Tech Stack

LangGraph · OpenRouter/MiMo v2.5 · yfinance · DuckDuckGo · NumPy · Streamlit

## Disclaimer

For educational and informational purposes only. Not personalized financial advice.