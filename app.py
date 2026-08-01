"""
app.py -- Modern Streamlit chat interface for the FinanceAgents team.
Run with: streamlit run app.py

Wraps the existing graph.py pipeline in a polished web UI.
"""
import re
import streamlit as st
from graph import build_graph


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="FinanceAgents",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ---------- Global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #ffffff !important; }

/* Force white background on all Streamlit containers */
.stApp, .main .block-container { background: #ffffff !important; }

/* Narrow the content area for ChatGPT-like centered feel */
.main .block-container {
    max-width: 800px !important;
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    margin: 0 auto !important;
}

/* ---------- Header ---------- */
.hero {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    padding: 1.5rem 2rem;
    border-radius: 14px;
    margin-bottom: 1.5rem;
    color: #111827;
    text-align: center;
}
.hero h1 { margin: 0 0 0.3rem 0; font-size: 1.6rem; font-weight: 700; color: #111827; }
.hero p  { margin: 0; color: #6b7280; font-size: 0.95rem; }

/* ---------- Metric cards ---------- */
.metric-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    text-align: center;
}
.metric-card .label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }
.metric-card .value { font-size: 1.35rem; font-weight: 700; }
.metric-card .value.negative { color: #ef4444; }
.metric-card .value.positive { color: #22c55e; }
.metric-card .value.neutral  { color: #f59e0b; }

/* ---------- Agent badge ---------- */
.badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-size: 0.73rem;
    font-weight: 600;
    margin-right: 0.25rem;
    margin-bottom: 0.25rem;
}
.badge-analyst  { background: #eff6ff; color: #2563eb; }
.badge-risk     { background: #fffbeb; color: #d97706; }
.badge-cio      { background: #f5f3ff; color: #7c3aed; }

/* ---------- Chat bubble tweaks ---------- */
.stChatMessage { border-radius: 10px !important; }

/* ---------- Chat input box — taller like Gemini/ChatGPT ---------- */
.stChatInputTextArea, textarea[data-testid="stChatInputTextArea"] {
    min-height: 100px !important;
    max-height: 200px !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
    border-radius: 14px !important;
}

/* ---------- Sidebar ---------- */
.sidebar-card {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    font-size: 0.85rem;
    line-height: 1.5;
    color: #374151;
}
.sidebar-card h4 { margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #111827; }

/* ---------- Quick-start buttons ---------- */
.stButton > button {
    border-radius: 10px !important;
    border: 1px solid #e5e7eb !important;
    background: #f9fafb !important;
    color: #374151 !important;
    font-weight: 500 !important;
    font-size: 0.82rem !important;
    padding: 0.3rem 0.8rem !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    background: #e5e7eb !important;
    border-color: #d1d5db !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <h4>🏗️ Architecture</h4>
        <code>CIO</code> dispatches two agents <b>in parallel</b> via <code>Send()</code>,<br>
        waits for both, then synthesizes one balanced brief.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>🤖 Agents</h4>
        <span class="badge badge-analyst">Investment Analyst</span>
        <span class="badge badge-risk">Risk Specialist</span>
        <span class="badge badge-cio">CIO</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sidebar-card">
        <h4>⚙️ How it works</h4>
        <ol style="margin:0; padding-left:1.2rem;">
            <li>Yank real market data via <b>yfinance</b></li>
            <li>Compute risk metrics with <b>NumPy</b></li>
            <li>LLM interprets both in parallel</li>
            <li>CIO merges into one brief</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Powered by LangGraph · OpenRouter · yfinance")
    st.caption("⚠️ Not financial advice")


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------------------------
# Helper: extract ticker
# ---------------------------------------------------------------------------
def extract_ticker(query: str) -> str:
    match = re.search(r'\b([A-Z]{2,10}(?:\.NS|\.BO)?)\b', query)
    return match.group(1) if match else query.strip()


# ---------------------------------------------------------------------------
# Helper: render risk metric cards from raw text
# ---------------------------------------------------------------------------
def render_risk_metrics(risk_text: str):
    """Try to extract numeric metrics from the risk assessment text and show cards."""
    import re as _re
    metrics = {}

    vol_match = _re.search(r'(\d+\.?\d*)\s*%', risk_text)
    sharpe_match = _re.search(r'Sharpe.*?(-?\d+\.?\d*)', risk_text, _re.IGNORECASE)
    drawdown_match = _re.search(r'(?:drawdown|decline).*?(-?\d+\.?\d*)\s*%', risk_text, _re.IGNORECASE)

    if vol_match:
        metrics["Volatility"] = (f"{vol_match.group(1)}%", "neutral")
    if sharpe_match:
        val = float(sharpe_match.group(1))
        metrics["Sharpe Ratio"] = (str(val), "positive" if val > 0 else "negative")
    if drawdown_match:
        val = float(drawdown_match.group(1))
        metrics["Max Drawdown"] = (f"{val}%", "negative" if val < -15 else "neutral")

    if metrics:
        cols = st.columns(len(metrics))
        for (label, (value, tone)), col in zip(metrics.items(), cols):
            with col:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">{label}</div>
                    <div class="value {tone}">{value}</div>
                </div>
                """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📈 FinanceAgents</h1>
    <p>AI-powered investment analysis — real market data, quantitative risk metrics, and balanced synthesis.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Quick-start ticker chips
# ---------------------------------------------------------------------------
st.markdown("**Try a ticker:**")
chip_cols = st.columns([1, 1, 1, 1, 1])
quick_tickers = ["RELIANCE.NS", "INFY.NS", "RVNL.NS", "TSLA", "AAPL"]
for i, ticker in enumerate(quick_tickers):
    with chip_cols[i]:
        if st.button(f"🔍 {ticker}", key=f"chip_{ticker}", use_container_width=True):
            st.session_state["pending_query"] = f"Analyze {ticker}"
            st.rerun()


# ---------------------------------------------------------------------------
# Display chat history
# ---------------------------------------------------------------------------
for entry in st.session_state.history:
    with st.chat_message("user", avatar="🧑"):
        st.markdown(f"**{entry['query']}**  →  `{entry['ticker']}`")

    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(entry["brief"])
        render_risk_metrics(entry.get("risk", ""))
        with st.expander("📋 Agents that ran"):
            for agent in entry["agents"]:
                label = agent.replace("_", " ").title()
                if "analyst" in agent:
                    st.markdown(f'<span class="badge badge-analyst">{label}</span>', unsafe_allow_html=True)
                elif "risk" in agent:
                    st.markdown(f'<span class="badge badge-risk">{label}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge badge-cio">{label}</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Handle pending quick-start query
# ---------------------------------------------------------------------------
if "pending_query" in st.session_state:
    query = st.session_state.pop("pending_query")
    # Trigger the same flow as a chat input
    st.session_state["_run_query"] = query


# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
query = st.chat_input("Ask about a stock — e.g. RELIANCE.NS, TSLA, INFY.NS …")

if "_run_query" in st.session_state:
    query = st.session_state.pop("_run_query")

if query:
    ticker = extract_ticker(query)

    with st.chat_message("user", avatar="🧑"):
        st.markdown(f"**{query}**  →  `{ticker}`")

    with st.chat_message("assistant", avatar="🤖"):
        # Progress indicators
        progress_area = st.empty()
        progress_area.markdown("""
        <div style="padding:0.5rem 0;">
            <span class="badge badge-analyst">Investment Analyst</span> ⏳ &nbsp;
            <span class="badge badge-risk">Risk Specialist</span> ⏳ &nbsp;
            <span class="badge badge-cio">CIO</span> ⏳
        </div>
        """, unsafe_allow_html=True)

        with st.spinner(f"🔄  Running Investment Analyst & Risk Specialist in parallel on `{ticker}` …"):
            result = st.session_state.graph.invoke({
                "query": query,
                "ticker": ticker,
                "investment_research": "",
                "risk_assessment": "",
                "final_brief": "",
                "completed_agents": [],
                "loop_count": 0,
            })

        # Update badges to "done"
        progress_area.markdown("""
        <div style="padding:0.5rem 0;">
            <span class="badge badge-analyst">✓ Investment Analyst</span> &nbsp;
            <span class="badge badge-risk">✓ Risk Specialist</span> &nbsp;
            <span class="badge badge-cio">✓ CIO</span>
        </div>
        """, unsafe_allow_html=True)

        # Render the final brief
        st.markdown(result["final_brief"])

        # Render risk metric cards if available
        if result.get("risk_assessment"):
            render_risk_metrics(result["risk_assessment"])

        # Expandable raw sections
        with st.expander("🔬  Raw Investment Research"):
            st.text(result.get("investment_research", "—"))

        with st.expander("📊  Raw Risk Assessment"):
            st.text(result.get("risk_assessment", "—"))

        with st.expander("🤖  Agents that ran"):
            for agent in result["completed_agents"]:
                label = agent.replace("_", " ").title()
                if "analyst" in agent:
                    st.markdown(f'<span class="badge badge-analyst">{label}</span>', unsafe_allow_html=True)
                elif "risk" in agent:
                    st.markdown(f'<span class="badge badge-risk">{label}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<span class="badge badge-cio">{label}</span>', unsafe_allow_html=True)

    # Save to history
    st.session_state.history.append({
        "query": query,
        "ticker": ticker,
        "brief": result["final_brief"],
        "risk": result.get("risk_assessment", ""),
        "agents": result["completed_agents"],
    })