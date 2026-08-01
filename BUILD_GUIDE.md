# Financial Agent Team — Build Guide

*3 agents: CIO (supervisor/synthesizer), Investment Analyst, Risk
Management Specialist.*

## Architecture in one paragraph

The CIO doesn't do research itself — it dispatches the same target
(a ticker) to both specialists **in parallel** (`Send()`, since their
analyses don't depend on each other), waits for both to finish, then
synthesizes one brief. This is the current (2026) LangGraph "supervisor
with parallel fan-out" pattern, not something custom-invented for this
project — that matters for explaining it later.


## Directory structure

```
FinanceAgents/
├── venv/
├── state.py              # shared state schema + safety caps
├── config.py              # LLM connection settings
├── investment_analyst.py  # agent 1
├── risk_specialist.py     # agent 2
├── graph.py                # CIO supervisor + graph wiring (the core file)
├── main.py                 # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

```bash
mkdir finance-agents && cd finance-agents
python3 -m venv venv
source venv/bin/activate
pip install langgraph langchain openai yfinance numpy python-dotenv
```

`.env`:
```
LLM_API_KEY=your-openrouter-key
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=xiaomi/mimo-v2.5
```

Copy the 6 reference files provided (`state.py`, `config.py`,
`investment_analyst.py`, `risk_specialist.py`, `graph.py`, `main.py`) into
this directory — they're a complete, syntax-verified starting point, not
pseudocode. Feed them to Cline/Antigravity as the ground truth for what correct looks
like if you want it to extend or debug from here.

## Build sequence

1. **Set up the project and copy in the reference files.**
2. **Read `state.py` first.** This is the whole data model: what each
   agent reads and writes, and the `MAX_LOOPS` safety cap. Since you're
   paying per token for this project, understand this before running
   anything — it's what stops a confused model from quietly spending your
   balance in a retry loop.
3. **Test the Investment Analyst alone**: `python -c "import investment_analyst; print(investment_analyst.run('RELIANCE.NS'))"`
   Confirm real yfinance data comes back and the LLM produces a sensible
   summary before wiring anything else.
4. **Test the Risk Specialist alone**, same pattern. Confirm the computed
   metrics (volatility, Sharpe, max drawdown, beta) look sane for a
   ticker you know something about.
5. **Read `graph.py`** — this is the actual multi-agent part. The
   `cio_dispatch` function returning two `Send()` objects is what causes
   parallel execution; `cio_synthesize` only runs once both branches
   report back.
6. **Run the full team**: `python main.py "Should I look into RELIANCE.NS?"`
   Try 2-3 different tickers, including one you'd expect to be
   higher-risk, to see the risk assessment actually differ meaningfully.
7. **Check the safety cap works**: temporarily set `MAX_LOOPS = 0` in
   `state.py`, confirm you get the "hit the safety loop cap" message
   instead of a real API call, then set it back. Cheap insurance to
   verify before trusting it with your balance.
8. **If time remains**: wrap `main.py`'s logic in a minimal Streamlit
   chat interface. Prioritize this only after steps 1-7 are solid — the
   agent architecture is the actual interesting/demoable part of this
   project, not the UI.
9. Write the GitHub level README: architecture diagram or description, the swap
   decision and why, example output, and the disclaimer.

## Non-negotiable: the disclaimer

This produces investment-adjacent output. `graph.py`'s synthesis prompt
already requires the CIO to end every brief with a clear "not personalized
financial advice" line — don't remove this even under time pressure, and
don't let the CIO's synthesis collapse into a single confident
recommendation. Facts and trade-offs, not a directive.
