# progress.md — implementation status against report.md

Last updated: 2026-10-15. Branch `main`, base commit `32abe04`. Nothing is committed yet; all changes are in the working tree.

## Completed (verified offline)
| Task | What was done |
|---|---|
| SEC-01 | Added `llm.Settings` (immutable, per session) and `resolve_settings`: the server key is only sent to the configured base URL (`LLM_BASE_URL`), and a custom URL requires the user's own key. The UI key field is never pre-filled. `config.update_config` and its module globals are removed. |
| LLM-01 | `llm.complete()` tries `[LLM_MODEL, *FALLBACK_MODELS]` in order: de-duplicated, capped by `LLM_MAX_ATTEMPTS`, with a timeout on each attempt and SDK `max_retries=0`. It moves to the next model on 429, timeout, connection error, 5xx, empty content / `finish_reason=length`, or output that fails validation. It stops on 401/403 (`LLMAuthError`), 402 (`LLMBudgetError`) and 400 (`LLMRequestError`). Each result records `served_model` and the attempt list. Not done: the OpenRouter catalog check (only needed for tool calling, which was removed). |
| COR-02 | `errors.py` defines typed errors. No code returns error strings as content. The UI shows `user_message`; the CLI exits with 1. |
| COR-01 / ARC-01 | Took the report's "acceptable alternative": removed the tool-calling loop, the router, the unreachable nodes, the unused prompts, `investment_analyst.py`, `risk_specialist.py`, the `@tool` decorators and the unused state fields. The graph is now `resolve_ticker → analyze`. |
| COR-03 | `analytics/risk.py`: pure functions, a per-market benchmark (^GSPC / ^NSEI / ^FTSE / ^HSI / ^KS11 / ^FCHI), indexes normalized to calendar dates before aligning, and beta computed from a single covariance matrix (ddof=1). Output includes benchmark, risk-free rate and its source, observation count, and start/end dates. |
| DATA-01 (partial) | Payloads use numbers or `None`. The mislabeled growth fields are renamed. Added `as_of` and `financial_currency`. `Ticker.info` is fetched once per run. Not done: Pydantic models for the payloads. |
| COR-06 | `onboarding.py` holds the parsers shared by the CLI and the UI (country → currency, k/lakh/crore/million amounts, enums, horizon in years). |
| COR-04 | Fixed in the UI: the portfolio is generated once; each upload is processed once per `file_id` and validated; the profile is passed into the analysis; the onboarding transcript persists; the `num_positions` question is asked. |
| SEC-02 / F-11 / F-14 / F-15 | Split into `load_session_from_str` and `load_session_from_path`, with Pydantic schema validation and a 64 KB cap. Web content is wrapped in `<untrusted_web_results>`. The ticker badge is passed through `html.escape`. `profile.json` is gitignored, and the CLI only writes a profile when given `--save PATH`. |
| COR-05 | Ticker resolver: whole-word static map, a stopword list, and abstaining when the match score is below threshold. Share classes are kept (BRK.B → BRK-B). The `.NS` suffix is added only when the company name points to India. ATVI is removed. Up to 3 candidates and a `source` are returned. All F-07 cases have regression tests. |
| F-08 (partial) | The allocator uses the user's `num_positions`, enforces the market preference, uses a non-greedy JSON parse, normalizes weights to 100% and amounts exactly to the budget, and falls back to the next model when output fails validation. |
| TST-01 (partial) | 105 offline tests. The network is blocked in `conftest.py`, and `FakeLLMClient` scripts LLM responses per model. Two Streamlit AppTests are included. |
| TST-01 fixtures | `scripts/record_fixtures.py` recorded 1y closes, info and statements for AAPL, TTWO, NVDA, RELIANCE.NS, INFY.NS, ^GSPC, ^NSEI and one invalid ticker (`recorded_at` 2026-10-15). `tests/fixture_yf.py` stands in for `yf.Ticker`. `tests/test_fixtures.py` runs typed-payload, market-aware beta and end-to-end tests offline. Observed: both NSE tickers and ^NSEI are stamped in Asia/Kolkata. Fixture betas: RELIANCE.NS 0.95 (246 obs vs ^NSEI), AAPL 0.66 (250 obs vs ^GSPC). |
| REP-01 (partial) / REP-02 / CI-01 (partial) | Added `pyproject.toml` (Python ≥3.11,<3.13; pytest and ruff config), bounded `requirements.txt` (dropped the unused `langchain` and `langchain-core`), `requirements-dev.txt`, `.gitattributes`, and `.github/workflows/ci.yml` running ruff and pytest. |
| DOC-01 (partial) | The README now describes the actual architecture, design decisions, configuration and limitations. `.env.example` documents the new variables. |

## Validation (latest run)
- `venv/bin/pytest`: **116 passed**, offline, in about 2.3 s.
- `venv/bin/ruff check .`: clean.
- Code searches (excluding `venv/` and `.cortex/`):
  - `chat.completions.create` appears only in `llm.py`.
  - `"LLM error"` and `value=config.OPENAI_API_KEY` return nothing.
- **Not run:** live LLM calls, live yfinance/DDG calls, and a manual browser check of the Streamlit app. RELIANCE.NS / ^NSEI alignment is now verified on recorded data (246 overlapping days).
- **Not verified:** the CI workflow has not run on GitHub yet.

- Independent review (verifier-1, read-only): SEC-01, LLM-01 and COR-02 were reviewed. Two defects were found and fixed:
  - The server key could be sent to `api.openai.com`. It is now locked to `LLM_BASE_URL`.
  - Non-numeric `LLM_MAX_ATTEMPTS` / `LLM_TIMEOUT_S` caused a raw crash. They now raise `ConfigError`.
- **Not independently reviewed yet:** COR-03, COR-04, COR-05, COR-06, ARC-01 and SEC-02. These are covered only by the author's tests.

## Remaining (priority, dependencies)
0. **(P1) Independent review** of `graph.py`, `tools.py`, `analytics/risk.py`, `ticker_resolver.py`, `onboarding.py`, `session.py` and `portfolio_allocator.py`.
1. **ML-01 (P1):** deterministic, data-validated allocator. Depends on COR-03 (done) and recorded fixtures. Tickers are still LLM-proposed and not checked against market data.
2. **TST-01 rest (P2):** recorded DuckDuckGo and Yahoo-search fixtures.
3. **ML-02 (P1):** evaluation harness (resolver accuracy set, faithfulness checker, constraint suite) and `make eval-offline`. Depends on ML-01 and the fixtures.
4. **REP-01 rest (P1):** lockfile and Makefile.
5. **REL-01 (P2):** TTL cache for the benchmark and history; retry with backoff for yfinance and DDG.
6. **OBS-01 (P2):** a trace per run with a `run_id` and token totals. The resolver's LLM call is not yet in `llm_trace`.
7. **DATA-01 rest:** Pydantic payload models. Check the units of `dividendYield`.
8. **UX-01 / DEMO-01 / PKG-01 / DIFF-\*:** not started. A request cap is required before hosting.
9. **LLM-01 live smoke test:** a documented live run showing fallback from a rate-limited model. Costs a few cents; needs the owner's approval.
10. **DOC-01 rest:** move or remove `FinalFullAgentsGoal.txt`, add a LICENSE (the owner chooses which), and an evaluation table once ML-02 exists.

## Decisions and assumptions
- Default risk-free rates (US 0.04, IN 0.065) are labeled assumptions and can be overridden with environment variables. They are not sourced live, so the owner should set current values.
- Only INR and USD portfolios are supported. Other countries are asked to choose one.
- Pre-existing CRLF working-tree changes noted in REP-02 were not present at resume time; the index is already all LF.

## Blockers
- Team subagents in this environment cannot run shell commands (no approval prompt), so the implementation was done by the lead agent directly.

## Next task to resume
Implement ML-01: a curated, Yahoo-verified universe plus a deterministic bucket + inverse-volatility allocator that uses `analytics/risk.py`, with the LLM writing rationale text only.
