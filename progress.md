# progress.md — implementation status against docs/audit-report.md

Last updated: 2026-10-16. Branch `main`. Audited baseline is `32abe04`. The owner committed batch 1 (`6714313`) and part of batch 2; everything since is uncommitted in the working tree.

## Completed (verified offline)
| Task | What was done |
|---|---|
| SEC-01 | Added `llm.Settings` (immutable, per session) and `resolve_settings`: the server key is only sent to the configured base URL (`LLM_BASE_URL`), and a custom URL requires the user's own key. The UI key field is never pre-filled. `config.update_config` and its module globals are removed. |
| LLM-01 | `llm.complete()` tries `[LLM_MODEL, *FALLBACK_MODELS]` in order: de-duplicated, capped by `LLM_MAX_ATTEMPTS`, with a timeout on each attempt and SDK `max_retries=0`. It moves to the next model on 429, timeout, connection error, 5xx, empty content / `finish_reason=length`, or output that fails validation. It stops on 401/403 (`LLMAuthError`), 402 (`LLMBudgetError`) and 400 (`LLMRequestError`). Each result records `served_model` and the attempt list. Not done: the OpenRouter catalog check (only needed for tool calling, which was removed). |
| COR-02 | `errors.py` defines typed errors. No code returns error strings as content. The UI shows `user_message`; the CLI exits with 1. |
| COR-01 / ARC-01 | Took the report's "acceptable alternative": removed the tool-calling loop, the router, the unreachable nodes, the unused prompts, `investment_analyst.py`, `risk_specialist.py`, the `@tool` decorators and the unused state fields. The graph is now `resolve_ticker → analyze`. |
| COR-03 | `analytics/risk.py`: pure functions, a per-market benchmark (^GSPC / ^NSEI / ^FTSE / ^HSI / ^KS11 / ^FCHI), indexes normalized to calendar dates before aligning, and beta computed from a single covariance matrix (ddof=1). Output includes benchmark, risk-free rate and its source, observation count, and start/end dates. |
| DATA-01 | Payloads use numbers or `None`, carry `as_of` and `financial_currency`, and are validated by Pydantic models in `schemas.py` (extra fields forbidden). Growth is sent as `*_yoy_pct`: in a live run the model misread Yahoo's fraction 1.059 as "5.9%". `dividend_yield_pct` was checked live on 2026-10-16: yfinance 1.x reports `dividendYield` in percent (KO 2.31 vs a trailing fraction of 0.0227). Keys renamed: `week52_*`, `price_change_6mo_pct`. |
| COR-06 | `onboarding.py` holds the parsers shared by the CLI and the UI (country → currency, k/lakh/crore/million amounts, enums, horizon in years). |
| COR-04 | Fixed in the UI: the portfolio is generated once; each upload is processed once per `file_id` and validated; the profile is passed into the analysis; the onboarding transcript persists; the `num_positions` question is asked. |
| SEC-02 / F-11 / F-14 / F-15 | Split into `load_session_from_str` and `load_session_from_path`, with Pydantic schema validation and a 64 KB cap. Web content is wrapped in `<untrusted_web_results>`. The ticker badge is passed through `html.escape`. `profile.json` is gitignored, and the CLI only writes a profile when given `--save PATH`. |
| COR-05 | Ticker resolver: whole-word static map, a stopword list, and abstaining when the match score is below threshold. Share classes are kept (BRK.B → BRK-B). The `.NS` suffix is added only when the company name points to India. ATVI is removed. Up to 3 candidates and a `source` are returned. All F-07 cases have regression tests. |
| ML-01 / F-08 / F-17 | `portfolio/universe.json`: 35 instruments, each checked to have ≥200 Yahoo closes on 2026-10-15. ZOMATO.NS is no longer listed and was replaced by ETERNAL.NS; LIQUIDBEES.NS was excluded because near-zero volatility would dominate inverse-volatility weights. `portfolio/allocator.py`: risk-bucket weights, largest-remainder slot split, inverse volatility within buckets, renormalisation when a bucket has no data, and `validate_allocation`. The LLM writes rationales only and falls back to templates. Output is labelled an "educational model allocation". `tests/test_portfolio.py` runs 144 property cases (3 risks × 4 market options × 6 sizes × 2 budgets) on fixture volatilities: weights ≥0, sum 100±0.01, amounts = budget±0.01, tickers in universe, markets respected, bucket weights equal configuration, deterministic with no LLM, and still valid when the LLM fails. Not done: currency-aware budget thresholds (no longer needed, since position count is user-chosen), and per-instrument price-currency conversion (amounts are allocation values in the profile currency). |
| TST-01 | Offline tests (count at the time: 105; current total under Validation). The network is blocked in `conftest.py`, and `FakeLLMClient` scripts LLM responses per model. Two Streamlit AppTests are included. |
| TST-01 fixtures | `scripts/record_fixtures.py` recorded 1y closes, info and statements for AAPL, TTWO, NVDA, RELIANCE.NS, INFY.NS, ^GSPC, ^NSEI and one invalid ticker (`recorded_at` 2026-10-15). `tests/fixture_yf.py` stands in for `yf.Ticker`. `tests/test_fixtures.py` runs typed-payload, market-aware beta and end-to-end tests offline. Observed: both NSE tickers and ^NSEI are stamped in Asia/Kolkata. Fixture betas: RELIANCE.NS 0.95 (246 obs vs ^NSEI), AAPL 0.66 (250 obs vs ^GSPC). |
| ML-02 (offline part) | `evaluation/resolver_queries.csv` (99 labelled queries: explicit_ticker, company_name, contextual, trap, share_class, delisted). `evaluation/run_offline.py` (`make eval-offline`, also run in CI) runs the deterministic resolver stages on fixtures for the current code and for baseline `32abe04` (loaded from git), plus the 144-profile constraint suite. Gates: constraints 100%, resolver accuracy ≥ `evaluation/baseline.json` (83.8) − 2. Measured 2026-10-16: baseline 70.7% → current 83.8% accuracy; no-answer false positives 43.8% → 0.0%; constraints 100%. One error-analysis iteration was recorded (iter0 79.8%: the regex missed long suffixed tickers; fixed). Recording the fixtures also found stale data: `TATAMOTORS.NS` no longer trades after the demerger (mapped to TMCV.NS / TMPV.NS); `EA` has had no closes since 2026-08-04 (unmapped); and lookup now requires prices in the last month. |
| REP-01 | `requirements.lock` (pip freeze of the verified venv, with the unused `langchain` removed). A fresh Python 3.12 venv installed from it passed the full suite at the time (263 tests) and ruff. `Makefile` targets: setup, setup-latest, test, lint, run, fixtures, eval-offline. CI installs from the lockfile. |
| REP-01 (earlier) / REP-02 / CI-01 (partial) | Added `pyproject.toml` (Python ≥3.11,<3.13; pytest and ruff config), bounded `requirements.txt` (dropped the unused `langchain` and `langchain-core`), `requirements-dev.txt`, `.gitattributes`, and `.github/workflows/ci.yml` running ruff and pytest. |
| Batch 3: LICENSE / vision | Added the Apache-2.0 `LICENSE` (official text from apache.org) and `NOTICE` (copyright "Samrat Kumar", taken from git history; **owner to confirm**). `FinalFullAgentsGoal.txt` moved to `docs/vision.md` with a current-coverage table. |
| ML-02 faithfulness | `analytics/faithfulness.py` checks the numeric claims in a brief against the payload and cited web snippets. It accounts for rounding to the decimals shown (+0.1%), "≈/about" (+5%), units (trn/bn/mn/k/crore/lakh), bounds ("> 35×"), price-gap and same-statement ratios, and skips time spans, years, quarters, list markers, index names and the disclaimer. Results are attached to every analysis and shown in the UI and CLI. Leniency measured on random numbers 1–100: 5–9% accepted with 2 decimals, 12–15% with 1 decimal, 50–61% as integers. `make eval-live` (free models only, fixture data, 5 tickers): run 1 had 11/242 unsupported, run 2 1/201, run 3 1/209. Every run-1 flag was inspected; fixes are listed in the README. |
| REL-01 | `market_data.py`: thread-safe in-process TTL cache (900 s, `MARKET_DATA_TTL_S`) for history, info and statements, plus 3 attempts with exponential backoff and jitter. DuckDuckGo gets 2 attempts. The four data fetches in `analyze` run concurrently. Tests: the benchmark is downloaded once across two queries, TTL expiry refetches, a flaky source recovers, a persistent failure raises `DataUnavailableError` after exactly 3 attempts, and backoff grows. `make bench` (synthetic, 200 ms per call): sequential ~1030 ms, concurrent ~615 ms, warm repeat ~7 ms with 0 calls. The resolver and the analysis now share one cached `info` fetch. Switched `duckduckgo-search` (now warns that it was renamed) to `ddgs` 9.16, checked live; the lockfile was regenerated. |
| OBS-01 | `tracing.py`: each run gets a `run_id`, steps with ms/ok/error (including concurrent steps, via copied contextvars), every LLM call (task, served model, attempts, usage, latency, error), `total_latency_ms`, `total_tokens` and `models_used`. `JsonFormatter` writes JSON logs and redacts `sk-…`, `Bearer …` and `api_key=…`. `graph.run_analysis` is used by both the UI and the CLI. Tests: both resolver and synthesis calls appear, failed calls are traced, and a sentinel key never appears in logs, the trace or results. |
| Live risk-free rate | US uses the live `^IRX` 13-week T-bill close (4.088% on 2026-10-15) when it is ≤10 days old and between 0 and 25%. The source and date are recorded. Precedence: env override → live → default. India has no working Yahoo symbol (tried `^IN91`, `IN91D=X`, `^INBMK`, `IN10Y=RR`), so it stays on the configured 6.5%. |
| UX-01 | `presentation.py` and the app show: the resolved company with confidence and source, alternatives when confidence is not high, the brief, the accuracy-check result, data gaps, prices as-of / currency / benchmark window / risk-free source, and sources plus a trace table in an expander. Yahoo names are rendered as plain Markdown, not HTML. An AppTest checks these elements appear. |
| DEMO-01 | `FINANCEAGENTS_DEMO=1`, `make demo` or `financeagents --demo`: fixture Yahoo data, search over fixture names, no web search, and `DemoClient` replaying the 5 free-model briefs recorded in `demo/llm_responses.json` (other tickers get a template brief). Banner: "recorded data as of 2026-10-15". Settings are hidden in demo mode. Tests: CLI analysis (recorded and template), CLI investment, and both UI modes, all with the network blocked. |
| Repo reorganisation (2026-10-16) | Installable `src/financeagents/` package with sub-packages `analytics/`, `data/` (Yahoo access plus recorded fixtures as package data), `portfolio/`, `ui/` and `demo_data/`. Tracked files were moved with `git mv` to keep history. `eval/` was renamed to `evaluation/` (it shadowed the built-in `eval`). `report.md` moved to `docs/audit-report.md`, content unchanged. Dependencies now live in `pyproject.toml`; `requirements.txt` and `requirements-dev.txt` were removed, and `requirements.lock` stays for exact installs. Added a `financeagents` console script, a Makefile with `help`/`format`/`clean`, a reorganised `.gitignore`, and a README rewritten with layout, configuration table and CI badge. Verified in a fresh venv built from the lockfile: 331 tests pass, ruff is clean, eval gates pass, and the demo runs without a key. The Streamlit app boots (`/_stcore/health` returns `ok`). |
| DOC-01 (partial) | The README now describes the actual architecture, design decisions, configuration and limitations. `.env.example` documents the new variables. |

## Validation (latest run, 2026-10-16)
- `make test`: **331 passed**, offline, ~11 s (also verified in a fresh venv from `requirements.lock`). `make eval-offline`: gates pass (resolver 83.8%, constraints 100%).
- `venv/bin/ruff check .`: clean.
- Code searches (excluding `venv/` and `.cortex/`):
  - `chat.completions.create` appears only in `llm.py`.
  - `"LLM error"` and `value=config.OPENAI_API_KEY` return nothing.
- Live calls made this session: free models only for the faithfulness runs; the paid primary model was used only for the 2 approved smoke calls. Live Yahoo calls were used only to record fixtures and check units. Live DDG: one API check.
- **Not run:** a manual browser check of the Streamlit app (AppTests only).
- **Not verified:** the CI workflow has not run on GitHub yet.

- Independent review (verifier-1, read-only): SEC-01, LLM-01 and COR-02 were reviewed. Two defects were found and fixed:
  - The server key could be sent to `api.openai.com`. It is now locked to `LLM_BASE_URL`.
  - Non-numeric `LLM_MAX_ATTEMPTS` / `LLM_TIMEOUT_S` caused a raw crash. They now raise `ConfigError`.
- **Not independently reviewed yet:** COR-03, COR-04, COR-05, COR-06, ARC-01 and SEC-02. These are covered only by the author's tests.

- **Live LLM smoke test (approved by owner, 2026-10-16), OpenRouter, `.env` chain:**
  - Primary `upstage/solar-mini4` answered on the first attempt: "OK" in 2.3 s, 57+2 tokens.
  - Live ticker extraction for "Take Two before GTA 6" passed the JSON validator and gave TTWO.
  - Fallback check: the `.env`'s `:free` models were moved to the front. `gemma-4-31b-it:free` returned 429, then `gemma-4-26b-a4b-it:free` returned 429, then `cohere/north-mini-code:free` answered. Two runs gave identical results.
  - Total cost was a fraction of a cent.

- **Independent review (2026-10-16, 2 read-only reviewers, all post-audit modules):** 15 findings, none high. Every finding was checked against the code and the fixes are regression-tested.
  - **Medium:**
    - the resolver returned an unvalidated Yahoo-search ticker;
    - the fallback chain stopped on a 404 (model retired);
    - server-key sessions could pick any model, spending the operator's credits (now limited to configured models);
    - the portfolio summary stated the configured mix after renormalisation;
    - bond and gold ETFs were hidden inside the "etf" bucket (`asset_class` added; the summary reports the split);
    - the UI over-claimed ("all numbers match");
    - cited web numbers were flagged;
    - a live eval run could shrink the demo file (it now merges).
  - **Low:**
    - demo search was never used (default-argument binding);
    - all-NaN closes raised a bare `IndexError`;
    - `info.get(a, info.get(b))` never fell back when `a` was None;
    - GBp/ZAc minor units were off by 100×;
    - a malformed `RISK_FREE_RATE_*` value was mis-reported;
    - schema validation errors crashed the analysis instead of becoming data gaps;
    - the live eval hard-coded USD;
    - the horizon minimum didn't match its message;
    - stale doc claims (README risk-free rate, "every number").
  - Not changed, documented instead: numbers taken from the user's own question count as supported.

## Remaining (priority, dependencies)
2. **Owner tasks:**
   - review `evaluation/resolver_queries.csv` labels;
   - confirm the copyright line in `NOTICE`;
   - push to GitHub and confirm CI is green;
   - run the app manually.
3. **(P2) TST-01 rest:** a recorded DuckDuckGo fixture.
4. **(P2) PKG-01:** hosting, deferred by the owner. Needs a per-session request cap first.
5. **(P3) DIFF-01/02/03:** backtest, model comparison (paid models need approval), and follow-up memory.
6. **Time-dependent tests:** the `^IRX` fixture is "fresh" only for 10 days after 2026-10-15. After that, fixture runs use the default rate. Tests don't assert the live source on fixtures, but `make fixtures` refreshes it.

## Decisions and assumptions
- US risk-free rate is live (`^IRX`) with the default 0.04 as fallback. India uses 0.065, a labelled assumption; set `RISK_FREE_RATE_IN`.
- Owner decisions (2026-10-16): Apache-2.0 license; free models may be used freely, paid only when necessary; output is educational but should be accurate; hosting deferred.
- Only INR and USD portfolios are supported. Other countries are asked to choose one.
- Pre-existing CRLF working-tree changes noted in REP-02 were not present at resume time; the index is already all LF.

## Blockers
- Team subagents in this environment cannot run shell commands (no approval prompt), so the implementation was done by the lead agent directly. Reviewers can only read code.

## Next task to resume
Owner: review and commit, push, confirm CI is green (the README badge points at `SamratKumar29/FinanceAgents`), run `make run` / `make demo` manually, and review `evaluation/resolver_queries.csv` labels. After that, the next engineering items are PKG-01 (hosting with a request cap) or DIFF-01 (backtest).
