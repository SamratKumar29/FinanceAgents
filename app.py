"""
app.py -- Modern Streamlit chat interface for FinanceAgents v2.
Run with: streamlit run app.py

Two modes:
- Investment Mode: Onboarding → personalized portfolio with specific amounts
- Analysis Mode: Conversational stock analysis with agentic tool use
"""
import re
import json
import streamlit as st
import config
from graph import build_graph, _llm_call, ONBOARDING_PROMPT
from state import initial_state, empty_profile, UserProfile
from portfolio_allocator import suggest_portfolio, modify_portfolio
from session import save_session, load_session, get_profile_summary


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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #ffffff !important; }
.stApp, .main .block-container { background: #ffffff !important; }
.main .block-container { max-width: 800px !important; padding-top: 2rem !important; padding-bottom: 2rem !important; margin: 0 auto !important; }
.hero { background: #ffffff; border: 1px solid #e5e7eb; padding: 1.5rem 2rem; border-radius: 14px; margin-bottom: 1.5rem; color: #111827; text-align: center; }
.hero h1 { margin: 0 0 0.3rem 0; font-size: 1.6rem; font-weight: 700; color: #111827; }
.hero p  { margin: 0; color: #6b7280; font-size: 0.95rem; }
.metric-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 0.8rem 1rem; text-align: center; }
.metric-card .label { font-size: 0.75rem; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.2rem; }
.metric-card .value { font-size: 1.35rem; font-weight: 700; }
.metric-card .value.negative { color: #ef4444; }
.metric-card .value.positive { color: #22c55e; }
.metric-card .value.neutral  { color: #f59e0b; }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.73rem; font-weight: 600; margin-right: 0.25rem; margin-bottom: 0.25rem; }
.badge-analyst { background: #eff6ff; color: #2563eb; }
.badge-risk    { background: #fffbeb; color: #d97706; }
.badge-cio     { background: #f5f3ff; color: #7c3aed; }
.badge-mode    { background: #f0fdf4; color: #16a34a; }
.stChatMessage { border-radius: 10px !important; }
.stChatInputTextArea, textarea[data-testid="stChatInputTextArea"] { min-height: 100px !important; max-height: 200px !important; font-size: 0.95rem !important; padding: 0.75rem 1rem !important; border-radius: 14px !important; }
.sidebar-card { background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; font-size: 0.85rem; line-height: 1.5; color: #374151; }
.sidebar-card h4 { margin: 0 0 0.4rem 0; font-size: 0.9rem; color: #111827; }
.stButton > button { border-radius: 10px !important; border: 1px solid #e5e7eb !important; background: #f9fafb !important; color: #374151 !important; font-weight: 500 !important; font-size: 0.82rem !important; padding: 0.3rem 0.8rem !important; }
.stButton > button:hover { background: #e5e7eb !important; border-color: #d1d5db !important; }
.portfolio-table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
.portfolio-table th { text-align: left; padding: 0.5rem; border-bottom: 2px solid #e5e7eb; color: #6b7280; font-weight: 600; }
.portfolio-table td { padding: 0.5rem; border-bottom: 1px solid #f3f4f6; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------
if "graph" not in st.session_state:
    st.session_state.graph = build_graph()
if "history" not in st.session_state:
    st.session_state.history = []
if "mode" not in st.session_state:
    st.session_state.mode = "idle"  # "idle", "investment", "analysis"
if "user_profile" not in st.session_state:
    st.session_state.user_profile = empty_profile()
if "portfolio" not in st.session_state:
    st.session_state.portfolio = None
if "onboarding_step" not in st.session_state:
    st.session_state.onboarding_step = 0
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []


# ---------------------------------------------------------------------------
# Sidebar: Settings + Profile
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    api_key = st.text_input("API Key", value=config.OPENAI_API_KEY, type="password", help="Your OpenRouter or OpenAI API key")
    model_options = [m["name"] for m in config.SUGGESTED_MODELS]
    selected_model = st.selectbox("Model", model_options, index=0, help="Select a model. Tool-calling models enable the full agent loop.")
    base_url = st.text_input("Base URL", value=config.OPENAI_BASE_URL, help="API base URL")

    # Show tool-calling status
    if config.supports_tool_calling(selected_model):
        st.success("✅ Tool calling supported — full agent mode")
    else:
        st.info("ℹ️ Limited mode — pre-fetches all data, no interactive tool use")

    if st.button("Apply Settings", use_container_width=True):
        config.update_config(api_key=api_key, base_url=base_url, model=selected_model)
        st.session_state.graph = build_graph()
        st.success(f"Settings applied: {selected_model}")

    st.markdown("---")

    # Profile management
    st.markdown("## 👤 Profile")
    if st.session_state.user_profile.get("completed_onboarding"):
        st.markdown(get_profile_summary(st.session_state.user_profile))
    else:
        st.caption("No profile yet. Start Investment Mode to create one.")

    # Download profile
    if st.session_state.user_profile.get("completed_onboarding"):
        session_json = save_session(st.session_state.user_profile)
        st.download_button(
            "📥 Download Profile",
            data=session_json,
            file_name="financeagents_profile.json",
            mime="application/json",
            use_container_width=True,
        )

    # Upload profile
    uploaded_file = st.file_uploader("📤 Upload Profile", type=["json"], help="Resume a previous session")
    if uploaded_file:
        try:
            data = load_session(uploaded_file.read().decode())
            st.session_state.user_profile = data["user_profile"]
            st.session_state.mode = "investment"
            st.success("Profile loaded! Resuming Investment Mode.")
            st.rerun()
        except Exception as e:
            st.error(f"Failed to load profile: {e}")

    st.markdown("---")
    st.caption("Powered by LangGraph · OpenRouter · yfinance · DuckDuckGo")
    st.caption("⚠️ Not financial advice")


# ---------------------------------------------------------------------------
# Hero header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <h1>📈 FinanceAgents</h1>
    <p>AI-powered investment analysis — agentic reasoning, real market data, personalized portfolios.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Mode selector
# ---------------------------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📊 Analysis Mode", use_container_width=True, type="primary" if st.session_state.mode == "analysis" else "secondary"):
        st.session_state.mode = "analysis"
        st.rerun()
with col2:
    if st.button("💼 Investment Mode", use_container_width=True, type="primary" if st.session_state.mode == "investment" else "secondary"):
        st.session_state.mode = "investment"
        st.rerun()
with col3:
    if st.button("🔄 Reset", use_container_width=True):
        for key in ["mode", "user_profile", "portfolio", "onboarding_step", "chat_messages", "history"]:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Show current mode
if st.session_state.mode == "analysis":
    st.markdown('<span class="badge badge-mode">📊 Analysis Mode</span>', unsafe_allow_html=True)
elif st.session_state.mode == "investment":
    st.markdown('<span class="badge badge-mode">💼 Investment Mode</span>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Helper: extract ticker
# ---------------------------------------------------------------------------
def extract_ticker(query: str) -> str:
    match = re.search(r'\b([A-Z]{2,10}(?:\.NS|\.BO)?)\b', query)
    return match.group(1) if match else query.strip()


# ---------------------------------------------------------------------------
# Helper: render portfolio table
# ---------------------------------------------------------------------------
def render_portfolio(portfolio: dict):
    """Render portfolio allocation as a table."""
    positions = portfolio.get("positions", [])
    if not positions:
        st.warning("No positions in portfolio.")
        return

    currency = portfolio.get("currency", "USD")
    total = portfolio.get("total_amount", 0)

    st.markdown(f"### Portfolio Allocation ({currency})")

    for p in positions:
        cols = st.columns([3, 1, 1, 1])
        with cols[0]:
            st.markdown(f"**{p.get('ticker', 'N/A')}** — {p.get('company', 'N/A')}")
            st.caption(p.get("rationale", ""))
        with cols[1]:
            st.markdown(f"**{p.get('allocation_pct', 0):.1f}%**")
        with cols[2]:
            st.markdown(f"**{currency} {p.get('amount', 0):,.0f}**")
        with cols[3]:
            risk = p.get("risk_level", "medium")
            color = {"low": "green", "medium": "orange", "high": "red"}.get(risk, "gray")
            st.markdown(f":{color}[{risk}]")

    st.markdown(f"\n**Total: {currency} {total:,.0f}**")
    st.caption(portfolio.get("summary", ""))
    st.caption(portfolio.get("disclaimer", ""))


# ---------------------------------------------------------------------------
# Investment Mode: Onboarding Flow
# ---------------------------------------------------------------------------
def run_onboarding():
    """Run the LLM-guided onboarding conversation."""
    profile = st.session_state.user_profile
    step = st.session_state.onboarding_step

    # Define onboarding steps
    questions = [
        ("location", "Where are you based? (e.g., India, US, UK)"),
        ("base_currency", "What currency do you prefer? (INR or USD)"),
        ("budget", "What's your investable budget?"),
        ("budget_type", "Is this a one-time lumpsum or monthly investment?"),
        ("goals", "What are your investment goals? (e.g., retirement, wealth growth, passive income)"),
        ("risk_tolerance", "How would you describe your risk tolerance? (conservative, moderate, aggressive)"),
        ("investment_horizon", "What's your investment time horizon? (e.g., 1 year, 5 years, 20 years)"),
        ("num_positions", "How many stocks/ETFs would you like in your portfolio?"),
        ("preferred_markets", "Which markets? (US, India, or both)"),
        ("existing_investments", "Any existing investments? (ticker + value, or 'none')"),
    ]

    if step == 0:
        # Show welcome message
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("Welcome to **Investment Mode**! I'll help you build a personalized investment portfolio. Let me ask you a few questions.")
        st.session_state.onboarding_step = 1
        st.rerun()

    if step > len(questions):
        # Onboarding complete — generate portfolio
        profile["completed_onboarding"] = True
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("✅ Your profile is complete!")
            st.markdown(get_profile_summary(profile))

        with st.spinner("Generating your personalized portfolio..."):
            st.session_state.portfolio = suggest_portfolio(profile)

        with st.chat_message("assistant", avatar="🤖"):
            render_portfolio(st.session_state.portfolio)
        return

    # Ask current question
    field, question = questions[step - 1]

    # Check if we already have this field
    if profile.get(field) and field != "existing_investments":
        st.session_state.onboarding_step = step + 1
        st.rerun()
        return

    with st.chat_message("assistant", avatar="🤖"):
        if step == 3:  # budget
            st.markdown(f"{question}\n\n*Example: 500000 (INR) or 10000 (USD)*")
        elif step == 4:  # budget type
            st.markdown(f"{question}\n\n*Options: lumpsum or monthly*")
        elif step == 5:  # goals
            st.markdown(f"{question}\n\n*Options: retirement, wealth growth, passive income, capital preservation — type multiple separated by commas*")
        elif step == 6:
            st.markdown(f"{question}\n\n*Options: conservative, moderate, aggressive*")
        elif step == 8:  # num positions
            st.markdown(f"{question}\n\n*Suggestion based on your budget: **{st.session_state.portfolio.get('suggested_positions', 5)}** positions*")
        else:
            st.markdown(question)

    # Get user input
    user_input = st.chat_input(f"Answer question {step}/{len(questions)}...")
    if user_input:
        # Parse and store the answer
        answer = user_input.strip()

        if field == "budget":
            # Extract number from answer
            nums = re.findall(r'[\d,]+\.?\d*', answer.replace(",", ""))
            if nums:
                profile["budget"] = float(nums[0])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown("Please enter a valid number (e.g., 500000 or 10,000)")
                return

        elif field == "num_positions":
            nums = re.findall(r'\d+', answer)
            if nums:
                profile["num_positions"] = int(nums[0])
            else:
                with st.chat_message("assistant", avatar="🤖"):
                    st.markdown("Please enter a number (e.g., 5)")
                return

        elif field in ("goals", "preferred_markets", "preferred_sectors"):
            profile[field] = [x.strip().lower() for x in answer.split(",")]

        elif field == "existing_investments":
            if answer.lower() in ("none", "no", "nil", "n/a", "no existing investments"):
                profile[field] = []
            else:
                profile[field] = [{"description": answer}]

        elif field == "location":
            # Detect currency from location
            loc = answer.lower()
            profile["location"] = answer.title()
            if any(x in loc for x in ["india", "in"]):
                if not profile.get("base_currency"):
                    profile["base_currency"] = "INR"
            elif any(x in loc for x in ["us", "usa", "united states", "america"]):
                if not profile.get("base_currency"):
                    profile["base_currency"] = "USD"

        elif field == "base_currency":
            cur = answer.upper().strip()
            if "INR" in cur or "RUPEE" in cur:
                profile["base_currency"] = "INR"
            elif "USD" in cur or "DOLLAR" in cur:
                profile["base_currency"] = "USD"
            else:
                profile["base_currency"] = cur[:3]

        else:
            profile[field] = answer

        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"**{field}**: {answer}")

        st.session_state.onboarding_step = step + 1
        st.rerun()


# ---------------------------------------------------------------------------
# Analysis Mode: Chat Interface
# ---------------------------------------------------------------------------
def run_analysis():
    """Run the analysis mode chat interface."""
    # Display chat history
    for entry in st.session_state.chat_messages:
        with st.chat_message(entry["role"], avatar="🧑" if entry["role"] == "user" else "🤖"):
            st.markdown(entry["content"])

    # Chat input
    query = st.chat_input("Ask about a stock — e.g. Take Two before GTA 6, RELIANCE.NS, AAPL...")
    if query:
        with st.chat_message("user", avatar="🧑"):
            st.markdown(f"**{query}**")

        st.session_state.chat_messages.append({"role": "user", "content": query})

        # Run the graph
        with st.spinner("🔍 Researching..."):
            result = st.session_state.graph.invoke({
                **initial_state(query=query),
                "messages": [],
            })

        brief = result.get("final_brief", "No analysis available.")

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(brief)

            # Show ticker resolution info
            if result.get("ticker"):
                st.markdown(f"<span class='badge badge-analyst'>Resolved: {result['ticker']}</span>", unsafe_allow_html=True)

        st.session_state.chat_messages.append({"role": "assistant", "content": brief})

        # Save to history
        st.session_state.history.append({
            "query": query,
            "ticker": result.get("ticker", ""),
            "brief": brief,
            "mode": "analysis",
        })


# ---------------------------------------------------------------------------
# Main routing
# ---------------------------------------------------------------------------
if st.session_state.mode == "investment":
    run_onboarding()
elif st.session_state.mode == "analysis":
    run_analysis()
else:
    # Idle state — prompt user to choose
    st.markdown("""
    ### Choose a mode to get started:

    **📊 Analysis Mode** — Ask about any stock and get agentic research with real data.

    **💼 Investment Mode** — Complete a short profile and get a personalized portfolio with specific amounts.

    Click one of the buttons above to begin.
    """)