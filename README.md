# FinanceAgents

A multi-agent financial analysis team built with **LangGraph** — a Chief Investment Officer (CIO) supervisor that dispatches an Investment Analyst and a Risk Management Specialist in **parallel**, waits for both, then synthesizes one coherent brief.

---

## Architecture

```
                        ┌─────────────────────┐
                        │      User Query      │
                        │  "RELIANCE.NS?"      │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │    CIO Dispatcher    │
                        │   (cio_dispatch)     │
                        └──┬───────────────┬──┘
                           │               │
                  Send()   │               │   Send()
                  parallel │               │   parallel
                           │               │
            ┌──────────────▼──┐     ┌──────▼──────────────┐
            │ Investment      │     │ Risk Management     │
            │ Analyst         │     │ Specialist          │
            │ (yfinance data  │     │ (volatility, Sharpe,│
            │  → LLM summary) │     │  drawdown, beta →   │
            │                 │     │  LLM interpretation)│
            └──────────────┬──┘     └──────┬──────────────┘
                           │               │
                        ┌──▼───────────────▼──┐
                        │   CIO Synthesizer    │
                        │   (cio_synthesize)   │
                        │                      │
                        │  Merges both into    │
                        │  one balanced brief  │
                        │  + disclaimer        │
                        └──────────┬──────────┘
                                   │
                        ┌──────────▼──────────┐
                        │     Final Brief      │
                        └─────────────────────┘
```

### How it works

1. **CIO Dispatcher** receives the query and uses LangGraph's `Send()` to fan out to both specialists **in parallel**. Their analyses are independent — no reason to run them sequentially.
2. **Investment Analyst** pulls real market data from yfinance (price, P/E, 6-month change, 52-week range, analyst target) and feeds it to an LLM for a structured research summary.
3. **Risk Specialist** computes quantitative risk metrics from 1 year of price history (annualized volatility, Sharpe ratio, max drawdown, beta vs S&P 500) and has an LLM interpret them in plain language.
4. **CIO Synthesizer** only runs once **both** branches complete (fan-in). It merges both analyses into one balanced brief with trade-offs, and must end with a clear disclaimer.

This is the **supervisor with parallel fan-out** pattern — the current (2026) LangGraph approach for multi-agent orchestration where tasks are independent.

---

## Project Structure

```
FinanceAgents/
├── venv/                    # Python virtual environment
├── state.py                 # Shared state schema + MAX_LOOPS safety cap
├── config.py                # LLM connection settings (OpenRouter)
├── investment_analyst.py    # Agent 1: market data + research
├── risk_specialist.py       # Agent 2: risk metrics + interpretation
├── graph.py                 # CIO supervisor + graph wiring (core file)
├── main.py                  # CLI entry point
├── app.py                   # Streamlit chat UI
├── requirements.txt         # Pinned dependencies
├── .env                     # API keys (not committed)
├── .env.example             # Template for .env
└── README.md                # This file
```

---

## Setup

```bash
# Clone and enter the project
cd FinanceAgents

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure your API key
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### Required: Get an OpenRouter API key

1. Sign up at [openrouter.ai](https://openrouter.ai)
2. Create an API key
3. Add it to `.env` as `LLM_API_KEY`

---

## Usage

### CLI

```bash
python main.py "Should I look into RELIANCE.NS?"
python main.py "Analyze RVNL.NS"
python main.py "What about INFY.NS?"
```

### Streamlit UI

```bash
streamlit run app.py
```

Opens a chat interface in your browser at `http://localhost:8501`.

---

## Example Output

```
Analyzing: RELIANCE.NS

============================================================
FINAL BRIEF
============================================================
### Synthesized Investment Brief: Reliance Industries Limited (RELIANCE.NS)

#### 1. Investment Case (Analyst Summary)
Reliance Industries is a blue-chip Indian conglomerate with a dominant market
position. The stock has underperformed recently, trading near the lower end of
its 52-week range. Its trailing P/E ratio of 23.7 is moderate. The average
analyst price target (~₹1,681.7) suggests considerable potential upside.

#### 2. Risk Profile (Risk Specialist Summary)
- Volatility: 20.1% (moderate)
- Sharpe Ratio: -0.49 (poor risk-adjusted returns)
- Max Drawdown: -20.58% (significant downside risk)

#### 3. Key Trade-offs
Potential upside from diversified growth vs. high volatility and poor
historical risk-adjusted returns. Suitability depends on individual
risk tolerance and investment horizon.

This is general informational analysis, not personalized financial
advice -- consult a licensed financial advisor before making investment
decisions.
============================================================
(Agents that ran: ['investment_analyst', 'risk_specialist'])
```

---

## Safety Cap

The project includes a `MAX_LOOPS` safety cap in `state.py` (default: 4). This prevents a confused LLM from silently burning through API credits in a retry loop. If the cap is hit, the system outputs a clear warning message instead of making additional API calls.

---

## Tech Stack

| Component | Technology | Version |
|---|---|---|
| Orchestration | LangGraph | 1.2.10 |
| LLM | MiMo v2.5 via OpenRouter | — |
| Market Data | yfinance | 1.5.2 |
| Risk Metrics | NumPy | 2.5.1 |
| LLM Client | OpenAI SDK | 2.52.0 |
| CLI | Python argparse | — |
| Web UI | Streamlit | 1.60.0 |

---

## ⚠️ Disclaimer

**This tool produces investment-adjacent output for informational and educational purposes only.** The CIO synthesis always includes a clear disclaimer that this is not personalized financial advice. The system presents facts and trade-offs — it does not issue confident buy/sell/hold directives. Consult a licensed financial advisor before making investment decisions.

---

## Future Roadmap

This is Phase 1 with 3 agents. The full vision includes 18 specialized agents:

| Agent | Role |
|---|---|
| Chief Investment Officer (CIO) | Investment strategy & capital allocation |
| Portfolio Manager | Buy/sell/rebalance to match strategy |
| Financial Planner (CFP) | Retirement, education, insurance planning |
| Tax Advisor (CA/CPA) | Tax optimization & structuring |
| Estate Planning Attorney | Wealth transfer, trusts, succession |
| Corporate Lawyer | Contracts, business structures, M&A |
| Risk Management Specialist | Risk identification & hedging |
| Insurance Advisor | Life, health, property insurance |
| Accountant / Controller | Cash flow, bookkeeping, reporting |
| Chief Financial Officer (CFO) | Budgeting, liquidity, financial controls |
| Investment Analyst | Research companies & opportunities |
| Private Equity Specialist | Private company investments |
| Venture Capital Specialist | Startup investment evaluation |
| Real Estate Advisor | Property acquisitions & financing |
| Trust Officer / Fiduciary | Trust administration |
| Philanthropy Advisor | Charitable giving strategy |
| Banking & Lending Specialist | Financing & credit solutions |
| Compliance Officer | Regulatory compliance |