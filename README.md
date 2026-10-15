# FinanceAgents

LLM-assisted stock research briefs and educational model allocations, grounded in Yahoo Finance data.

## What it does

**Analysis Mode**: ask about a stock in natural language ("Take Two before GTA 6", "RELIANCE.NS").
1. The ticker resolver turns the query into a ticker and checks the ticker against Yahoo Finance. It tries these in order:
   - LLM extraction, then Yahoo search
   - a static name map, matched on whole words only
   - an all-caps regex, with stopwords such as CEO, ETF and PE excluded

   When none of these gives a confident match, it returns no ticker rather than guessing.
2. A fixed pipeline fetches market data, financials and analyst views. It also computes 1-year risk metrics against the ticker's home benchmark (S&P 500 for US stocks, NIFTY 50 for `.NS`/`.BO`). If you gave context, it also runs a web search.
3. One LLM call writes a balanced brief from that JSON. Missing data sections are listed as gaps, not invented. Web results are marked as untrusted content.

**Investment Mode**: a validated questionnaire (shared by the CLI and the UI), then an *educational model allocation*:
- The LLM proposes the positions.
- Code then checks the position count, ticker format and market preference.
- Code normalises the weights so they sum to 100% and the amounts sum exactly to the budget.

## Architecture

```
query ─► resolve_ticker ─► analyze ─► brief
                            │
              fetch_info (once) · market data · risk (analytics/risk.py)
              financials · analyst view · web search (only if context given)
```

| Module | Responsibility |
|---|---|
| `llm.py` | Per-session immutable `Settings`, server key locked to `LLM_BASE_URL`, `complete()` with ordered model fallback |
| `errors.py` | Typed errors. The UI and CLI show `user_message`; error text is never returned as content |
| `graph.py` | Two-node LangGraph pipeline |
| `ticker_resolver.py` | Ticker resolution and validation |
| `tools.py` | yfinance and DuckDuckGo wrappers that return typed dicts with `as_of` |
| `analytics/risk.py` | Pure functions for volatility, Sharpe, max drawdown and beta, plus a market table |
| `onboarding.py` | Questionnaire parsers (lakh/crore/k/million amounts, enums, country → currency) |
| `session.py` | Profile JSON export and import, validated with pydantic and size-capped |
| `portfolio_allocator.py` | LLM proposal followed by code validation and normalisation |

### Design decisions
- **A fixed pipeline instead of a ReAct tool-calling agent.** The earlier tool-calling loop discarded tool results and never ran with the configured models (see `report.md` F-02). A fixed pipeline works with any chat model and costs one synthesis call. It also makes it traceable which tool produced every number.
- **Client-side model fallback.** `LLM_MODEL` is tried first, then `FALLBACK_MODELS`, in order. The next model is tried on a 429, a timeout, a connection error, a 5xx, empty content or output that fails validation. 401, 402, 403 and 400 stop the chain immediately. `LLM_MAX_ATTEMPTS` caps how many models are tried per request.
- **The server key is protected.** The Streamlit key field is never pre-filled. Settings are per session. The server key is only sent to the operator-configured base URL; a custom URL requires the user's own key.

## Quick start

```bash
python3.12 -m venv venv && source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
cp .env.example .env        # add LLM_API_KEY (OpenRouter by default)
streamlit run app.py
```

CLI:
```bash
python main.py "Should I invest in Apple?"
python main.py --ticker RELIANCE.NS
python main.py --mode investment --save my_profile.json
```
The CLI exits with a non-zero code on any LLM or data error.

## Configuration

See `.env.example`: `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`, `FALLBACK_MODELS`, `LLM_MAX_ATTEMPTS`, `LLM_TIMEOUT_S`, `RISK_FREE_RATE_<US|IN|...>`. The default risk-free rates (US 4%, IN 6.5%) are fixed assumptions, not live data. Override them for accurate Sharpe ratios.

## Tests

```bash
pytest          # offline: network is blocked, and the LLM and yfinance are faked
ruff check .
```
CI (`.github/workflows/ci.yml`) runs both on every push and pull request.

## Limitations
- Portfolio tickers and weights come from the LLM. They are checked for structure, but not against market data and not by a deterministic allocator (roadmap item ML-01).
- There is no evaluation harness yet, so resolver accuracy and brief faithfulness are not measured.
- There is no caching or retry for yfinance or DuckDuckGo calls.
- Yahoo Finance data is provided for personal use under Yahoo's terms. It can be delayed or incomplete.

## Disclaimer

For educational and informational purposes only. This is not personalized financial advice.
