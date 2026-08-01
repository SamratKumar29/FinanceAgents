# FinanceAgents — Detailed Build & Execution Plan

> **Goal:** Build a 3-agent financial analysis team using LangGraph — CIO (supervisor/synthesizer), Investment Analyst, and Risk Management Specialist — with parallel fan-out/fan-in architecture.
>
> **Reference:** `BUILD_GUIDE.md` (9-step high-level guide), `FinalFullAgentsGoal.txt` (full 18-agent vision — Phase 1 covers 3 agents).

---

## Phase 1: Verification & Understanding (Steps 1–2)

### Step 1: Confirm project setup and file inventory ✅ DONE
- [x] Verify all 6 source files exist: `state.py` (27L), `config.py` (10L), `investment_analyst.py` (53L), `risk_specialist.py` (67L), `graph.py` (99L), `main.py` (48L)
- [x] Verify `.env` has valid `LLM_API_KEY` (OpenRouter sk-or-v1-*), `LLM_BASE_URL` (openrouter.ai), `LLM_MODEL` (xiaomi/mimo-v2.5)
- [x] Verify `venv/` is functional — Python 3.12.3
- [x] Verify all required packages installed: langgraph 1.2.10, langchain 1.3.14, openai 2.52.0, yfinance 1.5.2, numpy 2.5.1, python-dotenv 1.2.2
- [x] Created `requirements.txt` via `pip freeze` — 55 packages captured

> **Note:** All checks passed. Project is ready to run. Proceed to Step 2.

### Step 2: Deep-read `state.py` — understand the data model ✅ DONE
- [x] Read and document `TeamState` TypedDict fields: `query` (user question), `ticker` (extracted symbol), `investment_research` (Analyst output), `risk_assessment` (Risk output), `final_brief` (CIO synthesis), `completed_agents` (tracking list), `loop_count` (safety counter)
- [x] Understand `Annotated[list, operator.add]` reducer — allows parallel `Send()` nodes to both append without overwriting; LangGraph merges via `operator.add`
- [x] Understand `MAX_LOOPS = 4` safety cap — prevents confused model from burning tokens in retry loops; fires clear message instead of silent API spend
- [x] Note: Each node writes only to its designated field(s) — `investment_analyst_node` → `investment_research`, `risk_specialist_node` → `risk_assessment`, `cio_synthesize` → `final_brief`

> **Note:** `state.py` is the data contract. All nodes must follow it. Moving to Phase 2 — individual agent testing.

---

## Phase 2: Individual Agent Testing (Steps 3–4)

### Step 3: Test Investment Analyst standalone ✅ DONE
- [x] Ran `investment_analyst.run('RELIANCE.NS')` — success
- [x] yfinance returned real data: price ₹1307.8, market cap ₹17.7T, P/E 23.70, 6mo change -5.5%, 52wk range ₹1249.8–₹1611.8
- [x] LLM produced coherent analysis: valuation context, performance summary, business overview, contextual considerations
- [x] No import/API errors — clean execution
- [x] Response quality: factual, well-structured, no invented numbers

> **Note:** Analyst correctly avoids buy/sell calls — stays in its lane as a researcher. Ready for graph wiring.

### Step 4: Test Risk Specialist standalone ✅ DONE
- [x] Ran `risk_specialist.run('RELIANCE.NS')` — success
- [x] Metrics present and sane: volatility 20.1%, Sharpe -0.49, max drawdown -20.58%, beta "insufficient data"
- [x] Negative Sharpe indicates poor risk-adjusted returns over the past year — consistent with RELIANCE's recent price decline
- [x] LLM interpretation excellent: explains each metric in plain language, notes the missing beta as a "key gap"
- [x] **Benchmark note:** Beta is "insufficient data" because S&P 500 (^GSPC) vs Indian stock has misaligned trading days. For future: swap to `^NSEI` (NIFTY 50) for better Indian-market beta calculation

> **Note:** Both agents tested individually and working correctly. Moving to Phase 3 — graph wiring.

---

## Phase 3: Graph Wiring & Full Team Test (Steps 5–7)

### Step 5: Read and understand `graph.py` — the multi-agent core ✅ DONE
- [x] Flow traced: START → `cio_dispatch` (two `Send()` calls) → parallel `investment_analyst` + `risk_specialist` → fan-in `cio_synthesize` → END
- [x] `cio_dispatch` returns `[Send("investment_analyst", state), Send("risk_specialist", state)]` — parallel fan-out
- [x] `cio_synthesize` only runs after BOTH branches complete — fan-in half of the pattern
- [x] Synthesis prompt enforces: summarize investment case, summarize risk, lay out trade-offs, end with disclaimer
- [x] `build_graph()` verified — compiles to `CompiledStateGraph` successfully

> **Note:** Architecture is clean: CIO dispatches in parallel, waits for both, synthesizes. No custom orchestration needed — pure LangGraph `Send()` pattern.

### Step 6: Run the full team end-to-end ✅ DONE
- [x] Ran RELIANCE.NS — full brief produced with investment case, risk profile, trade-offs, disclaimer
- [x] Output contains all three sections: analyst research, risk assessment, CIO synthesis
- [x] Final brief includes exact disclaimer: "This is general informational analysis, not personalized financial advice"
- [x] TATAMOTORS.NS — ticker returned 404 (possibly delisted on Yahoo Finance); system handled gracefully with error-aware brief
- [x] RVNL.NS — higher-risk ticker produced meaningfully different assessment:
  - RELIANCE: volatility 20.1%, Sharpe -0.49, max drawdown -20.58%
  - RVNL: volatility 34.16%, Sharpe -1.28, max drawdown -42.77%
- [x] Risk assessments clearly differ between tickers — RVNL flagged as "highly volatile" with "severe downside risk"
- [x] `completed_agents` shows `['investment_analyst', 'risk_specialist']` confirming both agents ran

> **Note:** All three tickers tested. Full pipeline works end-to-end. The system handles bad tickers gracefully too.

### Step 7: Verify the safety cap mechanism ✅ DONE
- [x] Set `MAX_LOOPS = 0` in `state.py`
- [x] Ran `python main.py "RELIANCE.NS"` — output showed: "Stopped: hit the safety loop cap before a clean synthesis"
- [x] Safety cap fires correctly — agents still ran but CIO synthesis was skipped (no synthesis API call made)
- [x] Restored `MAX_LOOPS = 4` — verified with import test

> **Note:** Safety cap works as designed. Agents complete their work independently; the cap only prevents the CIO synthesis from looping. This is correct behavior — the cap fires before the LLM synthesis call, protecting token spend.

---

## Phase 4: Documentation & Polish (Steps 8–9)

### Step 8: (Conditional) Minimal Streamlit chat UI ✅ DONE
- [x] Installed streamlit 1.60.0 in venv
- [x] Created `app.py` with chat interface: query input, ticker extraction, spinner during analysis, markdown rendering of brief
- [x] Uses `st.session_state` for graph persistence and chat history across turns
- [x] Shows agents that ran in caption, supports multi-turn conversation
- [x] Syntax verified, imports clean
- [x] Launch with: `streamlit run app.py`

> **Note:** UI is minimal but functional. The graph is compiled once and cached in session state. Each query goes through the full pipeline with spinner feedback.

### Step 9: Write GitHub-level README.md ✅ DONE
- [x] Project title + one-line description
- [x] ASCII architecture diagram showing the full data flow
- [x] "How it works" explanation of the fan-out/fan-in pattern
- [x] Project structure listing all files
- [x] Setup instructions (venv, pip install, .env configuration)
- [x] Usage examples for CLI and Streamlit
- [x] Example output with RELIANCE.NS
- [x] Safety cap explanation
- [x] Tech stack table with versions
- [x] Mandatory disclaimer included
- [x] Full 18-agent roadmap table from FinalFullAgentsGoal.txt
- [x] Created `.env.example` with placeholder values

> **Note:** README is comprehensive and GitHub-ready. Covers architecture, setup, usage, output, safety, disclaimer, and roadmap.

---

## Key Decisions & Notes

| Decision | Choice | Rationale |
|---|---|---|
| LLM provider | OpenRouter (MiMo v2.5) | Cost-effective, OpenAI-compatible API |
| Benchmark | S&P 500 (`^GSPC`) | Global benchmark; can swap to NIFTY 50 for Indian focus |
| Risk-free rate | 6.5% (India 10Y G-Sec) | Aligned with Indian market context |
| Parallel pattern | LangGraph `Send()` | Official 2026 pattern for supervisor fan-out |
| Safety cap | `MAX_LOOPS = 4` | Prevents runaway token spend from retry loops |
| Ticker extraction | Regex-based (simple) | Good enough for v1; LLM-based resolution is a future enhancement |

---

## Execution Order Summary

1. **Steps 1–2** → Verify setup, understand data model
2. **Steps 3–4** → Test each agent individually (catch issues early)
3. **Steps 5–7** → Wire graph, run full team, verify safety cap
4. **Steps 8–9** → Optional UI, then documentation

**Critical path:** Steps 1 → 2 → 3 → 4 → 5 → 6 → 7 → 9 (Step 8 is optional)