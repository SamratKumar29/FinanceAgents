"""
app.py -- Streamlit chat interface for FinanceAgents.
Run with: streamlit run app.py

- Analysis Mode: natural-language stock question -> data-grounded brief
- Investment Mode: onboarding questionnaire -> educational model allocation
"""
import html
import logging

import streamlit as st

import config
from errors import ConfigError, FinanceAgentsError
from graph import build_graph
from llm import resolve_settings
from onboarding import apply_answer, is_complete, next_field, question_text
from portfolio_allocator import suggest_portfolio
from session import get_profile_summary, load_session_from_str, save_session
from state import empty_profile, initial_state

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title="FinanceAgents", page_icon="📈", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #ffffff !important; }
.stApp, .main .block-container { background: #ffffff !important; }
.main .block-container { max-width: 800px !important; padding-top: 2rem !important; padding-bottom: 2rem !important; margin: 0 auto !important; }
.hero { background: #ffffff; border: 1px solid #e5e7eb; padding: 1.5rem 2rem; border-radius: 14px; margin-bottom: 1.5rem; color: #111827; text-align: center; }
.hero h1 { margin: 0 0 0.3rem 0; font-size: 1.6rem; font-weight: 700; color: #111827; }
.hero p  { margin: 0; color: #6b7280; font-size: 0.95rem; }
.badge { display: inline-block; padding: 0.2rem 0.6rem; border-radius: 999px; font-size: 0.73rem; font-weight: 600; margin-right: 0.25rem; margin-bottom: 0.25rem; }
.badge-analyst { background: #eff6ff; color: #2563eb; }
.badge-mode    { background: #f0fdf4; color: #16a34a; }
.stChatMessage { border-radius: 10px !important; }
.stChatInputTextArea, textarea[data-testid="stChatInputTextArea"] { min-height: 100px !important; max-height: 200px !important; font-size: 0.95rem !important; padding: 0.75rem 1rem !important; border-radius: 14px !important; }
.stButton > button { border-radius: 10px !important; border: 1px solid #e5e7eb !important; background: #f9fafb !important; color: #374151 !important; font-weight: 500 !important; font-size: 0.82rem !important; padding: 0.3rem 0.8rem !important; }
.stButton > button:hover { background: #e5e7eb !important; border-color: #d1d5db !important; }
</style>
""", unsafe_allow_html=True)


SERVER = config.server_settings()


def _reset_flow_state() -> None:
    st.session_state.mode = "idle"
    st.session_state.user_profile = empty_profile()
    st.session_state.answered = set()
    st.session_state.onboarding_log = []
    st.session_state.portfolio = None
    st.session_state.portfolio_error = None
    st.session_state.chat_messages = []


if "mode" not in st.session_state:
    _reset_flow_state()
if "settings" not in st.session_state:
    try:
        st.session_state.settings = resolve_settings(SERVER)
    except ConfigError:
        st.session_state.settings = None
if "processed_upload" not in st.session_state:
    st.session_state.processed_upload = None


def show_error(e: FinanceAgentsError) -> None:
    logging.getLogger("app").warning("request failed: %r", e)
    st.error(e.user_message)


# ---------------------------------------------------------------------------
# Sidebar: settings + profile
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    # Never pre-fill: widget values are sent to the browser.
    api_key = st.text_input("API Key", value="", type="password", key="api_key_input",
                            help="Optional. Leave empty to use the server's key (if configured).")
    model_options = list(dict.fromkeys([SERVER.model, *config.SUGGESTED_MODELS]))
    selected_model = st.selectbox("Model", model_options, index=0)
    base_url = st.text_input("Base URL", value=SERVER.base_url,
                             help="A custom URL requires your own API key.")
    if st.button("Apply Settings", use_container_width=True):
        try:
            st.session_state.settings = resolve_settings(SERVER, api_key, base_url, selected_model)
            st.success(f"Settings applied: {selected_model}")
        except ConfigError as e:
            show_error(e)

    current = st.session_state.settings
    if current is None:
        st.warning("No API key configured. Enter your own key and click Apply.")
    elif current.uses_server_key:
        st.caption(f"🔑 Using the server key · model `{current.model}`")
    else:
        st.caption(f"🔑 Using your key · model `{current.model}`")

    st.markdown("---")
    st.markdown("## 👤 Profile")
    profile = st.session_state.user_profile
    if profile.get("completed_onboarding"):
        st.markdown(get_profile_summary(profile))
        st.download_button("📥 Download Profile", data=save_session(profile),
                           file_name="financeagents_profile.json", mime="application/json",
                           use_container_width=True)
    else:
        st.caption("No profile yet. Start Investment Mode to create one.")

    uploaded = st.file_uploader("📤 Upload Profile", type=["json"], help="Resume a previous session")
    if uploaded is not None and st.session_state.processed_upload != uploaded.file_id:
        st.session_state.processed_upload = uploaded.file_id  # handle each upload exactly once
        try:
            loaded = load_session_from_str(uploaded.getvalue().decode("utf-8", errors="replace"))
        except ValueError as e:
            st.error(f"Failed to load profile: {e}")
        else:
            p = loaded["user_profile"]
            st.session_state.user_profile = p
            st.session_state.answered = set()
            st.session_state.onboarding_log = []
            st.session_state.portfolio = None
            st.session_state.portfolio_error = None
            if p.get("completed_onboarding") and not is_complete(p):
                p["completed_onboarding"] = False
            st.session_state.mode = "investment"
            st.success("Profile loaded.")
            st.rerun()

    st.markdown("---")
    st.caption("Powered by LangGraph · OpenRouter · yfinance · DuckDuckGo")
    st.caption("⚠️ Educational use only — not financial advice")


st.markdown("""
<div class="hero">
    <h1>📈 FinanceAgents</h1>
    <p>Data-grounded stock briefs and educational model allocations.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📊 Analysis Mode", use_container_width=True,
                 type="primary" if st.session_state.mode == "analysis" else "secondary"):
        st.session_state.mode = "analysis"
        st.rerun()
with col2:
    if st.button("💼 Investment Mode", use_container_width=True,
                 type="primary" if st.session_state.mode == "investment" else "secondary"):
        st.session_state.mode = "investment"
        st.rerun()
with col3:
    if st.button("🔄 Reset", use_container_width=True):
        _reset_flow_state()
        st.rerun()

if st.session_state.mode in ("analysis", "investment"):
    label = "📊 Analysis Mode" if st.session_state.mode == "analysis" else "💼 Investment Mode"
    st.markdown(f'<span class="badge badge-mode">{label}</span>', unsafe_allow_html=True)


def render_portfolio(portfolio: dict) -> None:
    currency = portfolio["currency"]
    st.markdown(f"### Educational model allocation ({currency})")
    for p in portfolio["positions"]:
        cols = st.columns([3, 1, 1, 1])
        with cols[0]:
            st.markdown(f"**{p['ticker']}** — {p['company']}")
            st.caption(p["rationale"])
        cols[1].markdown(f"**{p['allocation_pct']:.1f}%**")
        cols[2].markdown(f"**{currency} {p['amount']:,.0f}**")
        color = {"low": "green", "medium": "orange", "high": "red"}[p["risk_level"]]
        cols[3].markdown(f":{color}[{p['risk_level']}]")
    st.markdown(f"\n**Total: {currency} {portfolio['total_amount']:,.0f}**")
    st.caption(portfolio["summary"])
    st.caption(f"Weights: fixed rules + inverse volatility. Rationales: {portfolio.get('rationale_source', 'n/a')}.")
    st.caption(portfolio["disclaimer"])


def run_onboarding() -> None:
    profile = st.session_state.user_profile
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown("Welcome to **Investment Mode**! A few questions to build an educational model allocation.")
    for role, text in st.session_state.onboarding_log:
        with st.chat_message(role, avatar="🧑" if role == "user" else "🤖"):
            st.markdown(text)

    if profile.get("completed_onboarding") or (not st.session_state.answered and is_complete(profile)):
        profile["completed_onboarding"] = True
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("✅ Your profile is complete!")
            st.markdown(get_profile_summary(profile))
        if st.session_state.portfolio is None and st.session_state.portfolio_error is None:
            settings = st.session_state.settings
            if settings is None:
                st.warning("Configure an API key in the sidebar to generate the allocation.")
                return
            with st.spinner("Generating your model allocation..."):
                try:
                    st.session_state.portfolio = suggest_portfolio(profile, settings)
                except FinanceAgentsError as e:
                    st.session_state.portfolio_error = e
        if st.session_state.portfolio_error is not None:
            show_error(st.session_state.portfolio_error)
            if st.button("Retry"):
                st.session_state.portfolio_error = None
                st.rerun()
        elif st.session_state.portfolio is not None:
            with st.chat_message("assistant", avatar="🤖"):
                render_portfolio(st.session_state.portfolio)
        return

    field = next_field(profile, st.session_state.answered)
    if field is None:  # all answered but invalid combination (should not happen)
        st.error("Profile incomplete; please press Reset.")
        return
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(question_text(field))
    answer = st.chat_input("Your answer...")
    if answer:
        new_profile, error = apply_answer(profile, field, answer)
        st.session_state.onboarding_log.append(("assistant", question_text(field)))
        st.session_state.onboarding_log.append(("user", answer))
        if error:
            st.session_state.onboarding_log.append(("assistant", f"⚠️ {error}"))
        else:
            st.session_state.user_profile = new_profile
            st.session_state.answered.add(field)
            if next_field(new_profile, st.session_state.answered) is None and is_complete(new_profile):
                new_profile["completed_onboarding"] = True
        st.rerun()


def run_analysis() -> None:
    for entry in st.session_state.chat_messages:
        with st.chat_message(entry["role"], avatar="🧑" if entry["role"] == "user" else "🤖"):
            st.markdown(entry["content"])

    query = st.chat_input("Ask about a stock — e.g. Take Two before GTA 6, RELIANCE.NS, AAPL...")
    if not query:
        return
    with st.chat_message("user", avatar="🧑"):
        st.markdown(query)
    st.session_state.chat_messages.append({"role": "user", "content": query})

    settings = st.session_state.settings
    if settings is None:
        st.error("No API key configured. Enter your own key in the sidebar and click Apply.")
        return
    with st.spinner("🔍 Researching..."):
        try:
            result = build_graph(settings).invoke(
                initial_state(query=query, user_profile=st.session_state.user_profile)
            )
        except FinanceAgentsError as e:
            show_error(e)
            return

    brief = result["final_brief"]
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(brief)
        tr = result.get("ticker_result") or {}
        if tr:
            st.markdown(
                f"<span class='badge badge-analyst'>Resolved: {html.escape(tr['ticker'])} · "
                f"{html.escape(str(tr.get('company', '')))} · confidence {html.escape(tr['confidence'])}</span>",
                unsafe_allow_html=True,
            )
        with st.expander("Sources & trace"):
            st.write({"data_gaps": result.get("data_gaps", []), "sources": result.get("sources", []),
                      "llm": result.get("llm_trace", [])})
            st.json(result.get("data", {}), expanded=False)
    st.session_state.chat_messages.append({"role": "assistant", "content": brief})


if st.session_state.mode == "investment":
    run_onboarding()
elif st.session_state.mode == "analysis":
    run_analysis()
else:
    st.markdown("""
    ### Choose a mode to get started:

    **📊 Analysis Mode** — Ask about a stock and get a brief grounded in market data.

    **💼 Investment Mode** — Answer a short questionnaire and get an educational model allocation.
    """)
