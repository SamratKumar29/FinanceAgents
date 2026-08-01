"""
state.py -- The shared state schema for the FinanceAgents v2 agentic system.

Includes UserProfile for Investment Mode, conversation history,
tool tracking, and safety caps. Every node reads from and writes to this state.
"""
import operator
from typing import Annotated, TypedDict, Optional


# ---------------------------------------------------------------------------
# User Profile (for Investment Mode onboarding)
# ---------------------------------------------------------------------------
class UserProfile(TypedDict, total=False):
    location: str               # country code: "IN", "US", "GB", etc.
    base_currency: str          # "INR" or "USD"
    budget: float               # investable capital
    budget_type: str            # "lumpsum" or "monthly"
    goals: list[str]            # ["retirement", "wealth_growth", "passive_income", etc.]
    risk_tolerance: str         # "conservative", "moderate", "aggressive"
    investment_horizon: str     # "1 year", "5 years", "20 years"
    preferred_markets: list[str]  # ["US", "IN", "both"]
    preferred_sectors: list[str]
    num_positions: int          # how many stocks to suggest
    existing_investments: list  # [{"ticker": "AAPL", "shares": 10, "value": 1500}]
    completed_onboarding: bool


# Default empty profile
def empty_profile() -> UserProfile:
    return UserProfile(
        location="",
        base_currency="",
        budget=0.0,
        budget_type="",
        goals=[],
        risk_tolerance="",
        investment_horizon="",
        preferred_markets=[],
        preferred_sectors=[],
        num_positions=5,
        existing_investments=[],
        completed_onboarding=False,
    )


# ---------------------------------------------------------------------------
# Message (for conversation history)
# ---------------------------------------------------------------------------
class Message(TypedDict):
    role: str       # "user", "assistant", "system"
    content: str


# ---------------------------------------------------------------------------
# Main State (shared across all nodes)
# ---------------------------------------------------------------------------
class TeamState(TypedDict):
    # --- Core fields ---
    query: str                          # the user's question
    ticker: str                         # resolved ticker symbol
    ticker_context: str                 # context from resolver (e.g. "GTA 6 catalyst")
    investment_research: str            # filled by Investment Analyst
    risk_assessment: str                # filled by Risk Specialist
    final_brief: str                    # filled by CIO synthesis
    completed_agents: Annotated[list, operator.add]

    # --- User profile (Investment Mode) ---
    user_profile: UserProfile

    # --- Conversation ---
    messages: Annotated[list, operator.add]  # conversation history

    # --- Mode & routing ---
    mode: str                           # "idle", "investment", "analysis"
    router_decision: str                # raw router LLM output

    # --- Safety caps ---
    loop_count: int                     # synthesis loop counter
    tool_call_count: int                # tool call counter (prevents infinite loops)

    # --- Tool calling ---
    tools_enabled: bool                 # whether tool calling is available


# --- Safety caps ---
MAX_LOOPS = 4              # max CIO synthesis loops
MAX_TOOL_CALLS = 10        # max tool calls per agent turn


# ---------------------------------------------------------------------------
# State initializer
# ---------------------------------------------------------------------------
def initial_state(query: str = "", ticker: str = "") -> dict:
    """Create a fresh initial state dict."""
    return {
        "query": query,
        "ticker": ticker,
        "ticker_context": "",
        "investment_research": "",
        "risk_assessment": "",
        "final_brief": "",
        "completed_agents": [],
        "user_profile": empty_profile(),
        "messages": [],
        "mode": "idle",
        "router_decision": "",
        "loop_count": 0,
        "tool_call_count": 0,
        "tools_enabled": True,
    }