# FinanceAgents

[![CI](https://github.com/SamratKumar29/FinanceAgents/actions/workflows/ci.yml/badge.svg)](https://github.com/SamratKumar29/FinanceAgents/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

LLM-assisted stock research briefs and educational model allocations, grounded in Yahoo Finance data.
Every number in a brief is checked against the data the model was given. Every run is traced. An offline
demo needs no API key.

> **Educational use only. This is not personalized financial advice.**

---

## Contents
- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Project layout](#project-layout)
- [Architecture](#architecture)
- [Configuration](#configuration)
- [Development](#development)
- [Evaluation](#evaluation)
- [Limitations](#limitations)
- [Scope, license, disclaimer](#scope-and-vision)

## What it does

**Analysis Mode**: ask about a stock in natural language ("Take Two before GTA 6", "RELIANCE.NS").
1. **Resolve the ticker.** The resolver tries these stages in order and checks every candidate against Yahoo:
   - LLM extraction, then Yahoo search;
   - a static name map, matched on whole words only;
   - an all-caps regex, with stopwords such as CEO, ETF and PE excluded.

   When no stage gives a confident match, it returns no ticker rather than guessing.
2. **Fetch the data in parallel:**
   - market snapshot, financial statements and analyst view;
   - 1-year risk against the ticker's home benchmark (S&P 500 for US stocks, NIFTY 50 for `.NS`/`.BO`);
   - a web search, only if you gave context.
3. **Write and check the brief.** One LLM call writes a balanced brief from that JSON. The numbers in it are then checked against the data, and anything unsupported is flagged. Web results are marked as untrusted content.

**Investment Mode**: a validated questionnaire (shared by CLI and UI), then an *educational model allocation*:
- **Instruments:** a curated universe of 35 US/India ETFs and large caps, each checked on Yahoo.
- **Weights:** fixed bucket weights per risk tolerance (e.g. moderate: 40% ETF / 40% growth / 20% speculative). Within each bucket, weights are inverse to 1-year volatility.
- **Checks in code:** weights sum to 100%, amounts sum exactly to the budget, only allowed markets, no duplicates.
- **Explanations:** the summary reports the actual bucket and asset-class (equity/bond/gold) split.
- **LLM role:** it only writes one-sentence rationales. If it fails, template text is used and the allocation is unchanged.

## Quick start

Requires Python 3.11 or 3.12 and `make`.

```bash
git clone https://github.com/SamratKumar29/FinanceAgents.git && cd FinanceAgents
make setup                 # venv + pinned dependencies (requirements.lock) + editable package
make demo                  # offline demo in the browser: no API key, no network
```

To use live data and a real model:

```bash
cp .env.example .env       # set LLM_API_KEY (OpenRouter by default) and optional FALLBACK_MODELS
make run                   # Streamlit app
```

Command line (installed as `financeagents`):

```bash
venv/bin/financeagents "Should I invest in Apple?"
venv/bin/financeagents --ticker RELIANCE.NS
venv/bin/financeagents --mode investment --save my_profile.json
venv/bin/financeagents --demo "How is Infosys doing?"
```
The CLI exits non-zero on any LLM or data error.

## Project layout

```
.
├── src/financeagents/          # the installable package
│   ├── cli.py                  # `financeagents` command
│   ├── pipeline.py             # LangGraph pipeline: resolve_ticker → analyze (+ run_analysis tracing)
│   ├── resolver.py             # natural language → validated ticker
│   ├── llm.py                  # per-session Settings, server-key guard, model-fallback complete()
│   ├── config.py  errors.py  tracing.py  schemas.py  state.py
│   ├── onboarding.py  session.py  demo.py
│   ├── analytics/              # risk.py (vol, Sharpe, drawdown, beta) · faithfulness.py (number checker)
│   ├── data/                   # market.py (cached Yahoo access) · tools.py · fixtures.py + fixtures/*.json
│   ├── portfolio/              # allocator.py (rules) · service.py (suggest_portfolio) · universe.json
│   ├── ui/                     # app.py (Streamlit) · presentation.py
│   └── demo_data/              # recorded model responses for demo mode
├── tests/                      # offline pytest suite (network blocked)
├── evaluation/                 # offline + live evaluation, latency benchmark, committed results
├── scripts/                    # record_fixtures.py
├── docs/                       # vision.md · audit-report.md
├── .github/workflows/ci.yml    # lint + tests + evaluation gate
├── pyproject.toml              # package metadata, dependencies, tool config
├── requirements.lock           # exact versions the test suite was verified with
├── Makefile                    # setup / test / lint / run / demo / eval-* / bench / fixtures / clean
└── progress.md                 # implementation log against docs/audit-report.md
```

## Architecture

```
query ─► resolve_ticker ─► analyze ─────────────────────────────► brief + faithfulness + trace
                             │  (concurrent, cached, retried)
                             ├─ market snapshot      ├─ financial statements
                             ├─ risk vs benchmark    ├─ analyst view
                             └─ web search (only if the query has context)
```

### Design decisions
- **A fixed pipeline instead of a ReAct tool-calling agent.** The earlier tool-calling loop discarded tool results and never ran with the configured models (`docs/audit-report.md` F-02). A fixed pipeline works with any chat model, costs one synthesis call, and makes it traceable which tool produced every number.
- **Client-side model fallback.** `LLM_MODEL` is tried first, then `FALLBACK_MODELS`, in order.
  - The next model is tried on 404, 408, 409, 429, 5xx, timeouts, connection errors, empty content, or output that fails validation.
  - 400, 401, 402 and 403 stop the chain immediately.
  - `LLM_MAX_ATTEMPTS` caps how many models are tried per request.
- **The server key is protected.**
  - The key field is never pre-filled, and settings are per session.
  - The server key is only sent to `LLM_BASE_URL`, and only with operator-configured models.
  - Other URLs or models require the user's own key.
- **Typed data.** Every tool payload is validated by a Pydantic model: numbers or `null`, an `as_of` date, and currency fields. Growth is sent as an explicit percentage.
- **Observability.** Every run gets a `run_id` and a trace of its steps, each LLM call (model, attempts, tokens, latency) and the totals. Logs are JSON with API keys redacted.

## Configuration

All settings are environment variables; see [`.env.example`](.env.example).

| Variable | Purpose |
|---|---|
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | Provider credentials and primary model (OpenRouter by default) |
| `FALLBACK_MODELS` | Comma-separated models tried in order after the primary |
| `LLM_MAX_ATTEMPTS`, `LLM_TIMEOUT_S` | Per-request model cap and per-attempt timeout |
| `RISK_FREE_RATE_<US\|IN\|...>` | Override the risk-free rate. US defaults to the live 13-week T-bill (`^IRX`, fallback 4%); India uses a fixed 6.5% |
| `MARKET_DATA_TTL_S` | Yahoo cache lifetime in seconds (default 900) |
| `FINANCEAGENTS_DEMO=1` | Offline demo mode (same as `make demo` / `--demo`) |

## Development

```bash
make help           # all targets
make test           # offline pytest suite: network blocked, LLM faked, Yahoo data from fixtures
make lint           # ruff (make format to auto-fix)
make eval-offline   # resolver + portfolio-constraint evaluation with CI gates
make eval-live      # brief faithfulness on FREE models only; also refreshes demo responses
make bench          # synthetic data-path latency benchmark
make fixtures       # re-record Yahoo fixtures (network)
make clean          # caches and build artefacts
```

CI runs ruff, pytest and the offline evaluation gate on every push and pull request, installing from `requirements.lock`.

## Evaluation

`make eval-offline` (no network, no LLM) writes `evaluation/results/<date>_offline.{json,md}`. CI fails if the constraint pass rate drops below 100%, or if resolver accuracy falls more than 2 points below `evaluation/baseline.json`.

### Latest offline run (2026-10-16, fixtures recorded 2026-10-15)

Resolver: 99 labelled queries, deterministic stages only (no LLM, no Yahoo search).

| Metric | Baseline `32abe04` | Current |
|---|---|---|
| Accuracy | 70.7% | 83.8% |
| False positives on no-answer queries | 43.8% | 0.0% |
| Abstain rate | 30.3% | 32.3% |
| Accuracy: company_name | 60.0% | 66.7% |
| Accuracy: contextual | 100.0% | 100.0% |
| Accuracy: delisted | 0.0% | 100.0% |
| Accuracy: explicit_ticker | 96.0% | 100.0% |
| Accuracy: share_class | 0.0% | 66.7% |
| Accuracy: trap | 60.0% | 100.0% |

Portfolio constraints: 144 synthetic profiles, pass rate 100.0%.

What this measures and what it doesn't:
- The resolver numbers cover only the deterministic stages, with Yahoo lookups served from recorded fixtures. Company names outside the static map count as misses, because the LLM and Yahoo-search stages are not exercised.
- The query labels were written by the developer and have not been independently reviewed.
- The baseline is the audited resolver at commit `32abe04`, run under the same conditions.
- Error analysis: in the first run (`evaluation/results/2026-10-16_offline_iter0.md`), explicit-ticker accuracy fell to 88% because long suffixed symbols such as `HDFCBANK.NS` were missed. Fixing that brought it back to 100%, and overall accuracy went from 79.8% to 83.8%.

### Brief faithfulness (live, free models only)

`make eval-live` builds briefs for 5 tickers from fixture data, using only the `:free` models in your `.env`. The numeric claims in each brief are then checked against the data the model received.

| Run (2026-10-16) | Model served | Numbers checked | Unsupported |
|---|---|---|---|
| 1 (before fixes) | cohere/north-mini-code:free | 242 | 11 (4.5%) |
| 2 (after checker + data fixes) | cohere/north-mini-code:free | 201 | 1 (0.5%) |
| 3 (after schema + live risk-free rate) | cohere/north-mini-code:free | 209 | 1 (0.5%) |

Each run-1 flag was inspected:
- One was a real model error caused by how the data was presented. Yahoo's `revenueGrowth` of 1.059 (+105.9%) was written as "5.9%". Growth is now sent as an explicit percentage.
- One was knowledge from outside the data ("S&P 500 typically ~15-20%"), which is correctly flagged.
- The rest were checker false positives: non-breaking hyphens, bounds such as "P/E > 35×", and ratios derived from two balance-sheet values. The checker was fixed and regression tests were added.

Run 2's single flag was a unit the checker didn't recognise ("trn"), now fixed. That brief also wrote an INR amount with "$", which the checker cannot detect; the prompt now names the currency to use. Run 3's single flag is a real rounding slip: 5.64 where the data says 5.627. The briefs differ between runs, so this is not a controlled comparison.

How lenient the checker is, measured on fixture payloads: random numbers between 1 and 100 match by chance 5–9% of the time with 2 decimals, 12–15% with 1 decimal, and 50–61% as integers. It reliably catches invented precise figures, but cannot prove round numbers or currency symbols are correct.

### Data-path latency (synthetic)

`make bench` adds a simulated 200 ms delay to each Yahoo call and uses an instant fake LLM. There are 5 Yahoo calls per cold run, including the live T-bill rate.

| Configuration | Wall time | Yahoo calls |
|---|---|---|
| Sequential fetches, cold cache | ~1230 ms | 5 |
| Concurrent fetches, cold cache | ~820 ms | 5 |
| Repeat query within the cache window | ~10 ms | 0 |

These numbers show the effect of concurrency and caching only. They are not real-world latencies, and LLM latency is not included.

## Limitations
- The allocation rules (bucket weights, inverse volatility) are simple heuristics and have not been backtested. The universe is a small hand-picked list, re-verified only when `make fixtures` is run.
- The faithfulness checker flags unsupported numbers but cannot verify reasoning, round numbers or currency symbols. Numbers that appear in the user's own question count as supported.
- The risk-free rate is live for the US only. India uses a configured 6.5%, because no reliable Yahoo symbol for Indian T-bill yields was found. Set `RISK_FREE_RATE_IN` to a current value.
- The cache is per process and in memory only. There is no hosted deployment or per-user request cap yet.
- Yahoo Finance data is provided for personal use under Yahoo's terms. It can be delayed or incomplete.

## Scope and vision

The original idea of a full team of financial experts (tax, estate, insurance, and more) is kept in [docs/vision.md](docs/vision.md) as future scope, together with which roles this project covers today. The engineering audit this work was based on is in [docs/audit-report.md](docs/audit-report.md), and implementation progress is tracked in [progress.md](progress.md).

## License

[Apache License 2.0](LICENSE). See [NOTICE](NOTICE). Market data from Yahoo Finance is subject to Yahoo's terms and is not covered by this license.

## Disclaimer

For educational and informational purposes only. Nothing here is personalized financial advice. Verify all data independently and consult a licensed adviser before investing.
