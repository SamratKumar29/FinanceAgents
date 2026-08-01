"""
graph.py -- The agentic CIO supervisor and graph wiring (v2).

Supports two modes based on the LLM model:
1. Tool-calling mode (GPT-4o, Claude, etc.): Full ReAct loop with LangGraph ToolNode
2. Fallback mode (MiMo, etc.): Pre-fetch all tools, then LLM synthesizes

The CIO agent uses tools to research, reason, and produce a balanced brief.
"""
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from openai import OpenAI

import config
from state import TeamState, MAX_LOOPS, MAX_TOOL_CALLS
from tools import ALL_TOOLS
from ticker_resolver import resolve_ticker


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
ROUTER_PROMPT = """You are a financial assistant router. Classify the user's message into ONE of these categories:

1. "analysis_request" — The user wants to analyze a specific stock or company
2. "investment_request" — The user wants investment advice, portfolio suggestions, or help building a portfolio
3. "general_question" — General finance question not about a specific stock
4. "onboarding_continue" — Continuing an onboarding conversation (answering profile questions)

Respond with ONLY the category name, nothing else."""


CIO_SYSTEM_PROMPT = """You are the Chief Investment Officer of a financial analysis team.
You have access to research tools that can fetch real market data, compute risk metrics,
search the web for news and catalysts, and analyze financial statements.

Your job is to:
1. Research the stock/company thoroughly using the available tools
2. Consider the user's context and any catalysts or events they mentioned
3. Synthesize all findings into a balanced, factual brief
4. Always end with the disclaimer: "This is general informational analysis, not personalized financial advice -- consult a licensed financial advisor before making investment decisions."

Be thorough but concise. Use specific numbers from the data, not vague statements.
Present facts and trade-offs -- do NOT issue a single confident buy/sell/hold directive."""

ONBOARDING_PROMPT = """You are a friendly financial advisor conducting an onboarding interview.
Your goal is to collect the user's investment profile. Ask ONE question at a time.

Current profile: {profile}

Ask the next question based on what's missing. Available questions:
1. Location/country ("Where are you based?")
2. Budget ("What's your investable budget? Is this a one-time lumpsum or monthly?")
3. Currency preference ("Do you prefer INR or USD?")
4. Financial goals ("What are your investment goals? e.g., retirement, wealth growth, passive income")
5. Risk tolerance ("How would you describe your risk tolerance? Conservative, moderate, or aggressive?")
6. Investment horizon ("What's your investment time horizon?")
7. Number of positions ("How many stocks would you like in your portfolio?")
8. Existing investments ("Do you have any existing investments?")

After all questions are answered, confirm the profile summary and say "PROFILE_COMPLETE".

Keep responses short and conversational."""

PORTFOLIO_PROMPT = """You are the Chief Investment Officer creating a personalized investment portfolio.

User Profile:
{profile}

Suggest exactly {num_positions} stocks/ETFs with specific allocation amounts in {currency}.
For each suggestion, provide:
- Ticker and company name
- Allocation percentage and amount in {currency}
- Brief rationale (1-2 sentences)
- Risk level for this pick

Also include:
- Total allocation summary
- Diversification notes
- A clear disclaimer: "This is general informational analysis, not personalized financial advice."

Be specific with numbers. Do NOT suggest amounts the user can't afford."""

SYNTHESIS_PROMPT = """You are the Chief Investment Officer synthesizing research from your team.
Combine the investment analysis and risk assessment into one coherent brief:

1. Summarize the investment case
2. Summarize the risk profile
3. Lay out key trade-offs
4. Consider any context the user provided (catalysts, events, comparisons)
5. End with: "This is general informational analysis, not personalized financial advice -- consult a licensed financial advisor before making investment decisions."

Be factual and specific. Do NOT issue a single confident buy/sell call."""


# ---------------------------------------------------------------------------
# Helper: LLM call (works for any model)
# ---------------------------------------------------------------------------
def _llm_call(messages: list[dict], system: str = None, **kwargs) -> str:
    """Make an LLM call with error handling."""
    client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(messages)
    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=msgs,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 2000),
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        return f"LLM error: {str(e)}"


# ---------------------------------------------------------------------------
# Node: Router
# ---------------------------------------------------------------------------
def router_node(state: TeamState) -> dict:
    """Classify user intent and set the mode."""
    query = state.get("query", "")
    if not query:
        return {"mode": "idle", "router_decision": "idle"}

    # Check if continuing onboarding
    profile = state.get("user_profile", {})
    if profile.get("completed_onboarding"):
        return {"mode": "analysis", "router_decision": "analysis_request"}

    # Check if profile has partial data (onboarding in progress)
    if profile.get("location") and not profile.get("completed_onboarding"):
        # Check if all required fields are filled
        required = ["location", "base_currency", "budget", "risk_tolerance", "investment_horizon"]
        if all(profile.get(f) for f in required):
            return {"mode": "investment", "router_decision": "onboarding_complete"}
        return {"mode": "investment", "router_decision": "onboarding_continue"}

    decision = _llm_call([{"role": "user", "content": query}], system=ROUTER_PROMPT)
    decision = decision.strip().lower().replace('"', '').replace("'", "")

    # Map to mode
    if "onboarding" in decision:
        return {"mode": "investment", "router_decision": decision}
    elif "investment" in decision:
        return {"mode": "investment", "router_decision": decision}
    elif "analysis" in decision or "general" in decision:
        return {"mode": "analysis", "router_decision": decision}
    else:
        # Default to analysis
        return {"mode": "analysis", "router_decision": "analysis_request"}


# ---------------------------------------------------------------------------
# Node: Ticker Resolution
# ---------------------------------------------------------------------------
def resolve_ticker_node(state: TeamState) -> dict:
    """Resolve the user's query to a stock ticker."""
    query = state.get("query", "")
    result = resolve_ticker(query)
    if result:
        return {
            "ticker": result.ticker,
            "ticker_context": result.context,
        }
    return {"ticker": "", "ticker_context": ""}


# ---------------------------------------------------------------------------
# Node: CIO Agent (ReAct loop for tool-calling models)
# ---------------------------------------------------------------------------
def cio_agent_tool_calling(state: TeamState) -> dict:
    """CIO agent that uses tool calling for models that support it."""
    from tools import TOOL_NAMES

    client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)

    # Build context
    profile = state.get("user_profile", {})
    context_parts = []
    if state.get("ticker"):
        context_parts.append(f"Ticker: {state['ticker']}")
    if state.get("ticker_context"):
        context_parts.append(f"Context: {state['ticker_context']}")
    if state.get("query"):
        context_parts.append(f"User query: {state['query']}")
    if profile.get("completed_onboarding"):
        context_parts.append(f"User profile: risk={profile.get('risk_tolerance')}, horizon={profile.get('investment_horizon')}, goals={profile.get('goals')}")

    context = "\n".join(context_parts) if context_parts else state.get("query", "")

    # Simple ReAct loop (limited iterations)
    tool_call_count = state.get("tool_call_count", 0)
    accumulated_research = []
    accumulated_risk = []

    for iteration in range(3):  # Max 3 tool-calling rounds
        if tool_call_count >= MAX_TOOL_CALLS:
            break

        messages = [{"role": "user", "content": f"Research this: {context}\n\nAccumulated findings:\n" + "\n".join(accumulated_research + accumulated_risk) if accumulated_research else f"Research this: {context}"}]

        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
                messages=[
                    {"role": "system", "content": CIO_SYSTEM_PROMPT},
                    *messages,
                ],
                tools=[{"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.args}} for t in ALL_TOOLS],
                temperature=0.3,
                max_tokens=2000,
            )
        except Exception:
            # Fallback: no tool calling support
            break

        msg = response.choices[0].message

        if not msg.tool_calls:
            # Final answer
            return {"final_brief": msg.content or "", "tool_call_count": tool_call_count}

        # Execute tool calls
        for tc in msg.tool_calls:
            tool_call_count += 1
            func_name = tc.function.name
            try:
                args = __import__("json").loads(tc.function.arguments)
            except Exception:
                args = {}

            # Find and execute the tool
            result = ""
            for t in ALL_TOOLS:
                if t.name == func_name:
                    try:
                        result = t.invoke(args)
                    except Exception as e:
                        result = f"Tool error: {e}"
                    break

            if func_name in ("get_market_data", "get_financials", "get_analyst_estimates"):
                accumulated_research.append(f"[{func_name}]: {result[:1000]}")
            elif func_name == "compute_risk":
                accumulated_risk.append(f"[Risk]: {result[:1000]}")
            elif func_name == "web_search":
                accumulated_research.append(f"[Web Search]: {result[:1000]}")

            messages.append({"role": "assistant", "content": None, "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": func_name, "arguments": tc.function.arguments}}]})
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result[:3000]})

    # Synthesize from accumulated findings
    if accumulated_research or accumulated_risk:
        all_findings = "\n\n".join(accumulated_research + accumulated_risk)
        synthesis = _llm_call(
            [{"role": "user", "content": f"Synthesize these findings into a brief:\n\n{all_findings}\n\nUser context: {context}"}],
            system=SYNTHESIS_PROMPT,
        )
        return {"final_brief": synthesis, "tool_call_count": tool_call_count}

    return {"final_brief": "Unable to gather sufficient data.", "tool_call_count": tool_call_count}


# ---------------------------------------------------------------------------
# Node: CIO Agent (Fallback for non-tool-calling models like MiMo)
# ---------------------------------------------------------------------------
def cio_agent_fallback(state: TeamState) -> dict:
    """CIO agent that pre-fetches all data, then has LLM synthesize."""
    import tools as tool_module

    ticker = state.get("ticker", "")
    context = state.get("ticker_context", "")
    query = state.get("query", "")

    if not ticker:
        return {"final_brief": "Could not identify a stock ticker from your query. Please try using the explicit ticker symbol (e.g., TTWO, RELIANCE.NS)."}

    # Pre-fetch all data
    findings = []

    # Market data
    try:
        market_data = tool_module.get_market_data.invoke({"ticker": ticker})
        findings.append(f"## Market Data\n{market_data}")
    except Exception as e:
        findings.append(f"## Market Data\nError: {e}")

    # Risk metrics
    try:
        risk_data = tool_module.compute_risk.invoke({"ticker": ticker})
        findings.append(f"## Risk Metrics\n{risk_data}")
    except Exception as e:
        findings.append(f"## Risk Metrics\nError: {e}")

    # Financials
    try:
        financials = tool_module.get_financials.invoke({"ticker": ticker})
        findings.append(f"## Financial Statements\n{financials}")
    except Exception as e:
        findings.append(f"## Financial Statements\nError: {e}")

    # Analyst estimates
    try:
        analysts = tool_module.get_analyst_estimates.invoke({"ticker": ticker})
        findings.append(f"## Analyst Estimates\n{analysts}")
    except Exception as e:
        findings.append(f"## Analyst Estimates\nError: {e}")

    # Web search for context
    if context:
        try:
            web_results = tool_module.web_search.invoke({"query": f"{ticker} {context}"})
            findings.append(f"## Recent News & Context\n{web_results}")
        except Exception as e:
            findings.append(f"## Recent News & Context\nError: {e}")

    # Synthesize
    all_data = "\n\n".join(findings)
    profile = state.get("user_profile", {})
    profile_context = ""
    if profile.get("completed_onboarding"):
        profile_context = f"\nUser profile: risk={profile.get('risk_tolerance')}, horizon={profile.get('investment_horizon')}, goals={profile.get('goals')}, budget={profile.get('budget')} {profile.get('base_currency')}"

    synthesis = _llm_call(
        [{"role": "user", "content": f"Stock: {ticker} | Context: {context} | Query: {query}{profile_context}\n\nData collected:\n{all_data}"}],
        system=SYNTHESIS_PROMPT,
    )
    return {"final_brief": synthesis}


# ---------------------------------------------------------------------------
# Node: Investment Analyst (parallel research)
# ---------------------------------------------------------------------------
def investment_analyst_node(state: TeamState) -> dict:
    """Fetch and summarize investment research."""
    ticker = state.get("ticker", "")
    if not ticker:
        return {"investment_research": "No ticker identified.", "completed_agents": ["investment_analyst"]}

    import tools as tool_module
    try:
        data = tool_module.get_market_data.invoke({"ticker": ticker})
    except Exception:
        data = f"Error fetching data for {ticker}"

    return {"investment_research": data, "completed_agents": ["investment_analyst"]}


# ---------------------------------------------------------------------------
# Node: Risk Specialist (parallel research)
# ---------------------------------------------------------------------------
def risk_specialist_node(state: TeamState) -> dict:
    """Compute and interpret risk metrics."""
    ticker = state.get("ticker", "")
    if not ticker:
        return {"risk_assessment": "No ticker identified.", "completed_agents": ["risk_specialist"]}

    import tools as tool_module
    try:
        data = tool_module.compute_risk.invoke({"ticker": ticker})
    except Exception:
        data = f"Error computing risk for {ticker}"

    return {"risk_assessment": data, "completed_agents": ["risk_specialist"]}


# ---------------------------------------------------------------------------
# Node: CIO Synthesis (fan-in from parallel agents)
# ---------------------------------------------------------------------------
def cio_synthesize_node(state: TeamState) -> dict:
    """Synthesize findings from both agents into one brief."""
    if state.get("loop_count", 0) >= MAX_LOOPS:
        return {"final_brief": "Stopped: hit the synthesis loop cap."}

    research = state.get("investment_research", "")
    risk = state.get("risk_assessment", "")
    context = state.get("ticker_context", "")
    query = state.get("query", "")

    synthesis = _llm_call(
        [{"role": "user", "content": f"User asked: {query}\nContext: {context}\n\nInvestment Research:\n{research}\n\nRisk Assessment:\n{risk}"}],
        system=SYNTHESIS_PROMPT,
    )
    return {"final_brief": synthesis, "loop_count": state.get("loop_count", 0) + 1}


# ---------------------------------------------------------------------------
# Conditional edge: choose agent path based on model capability
# ---------------------------------------------------------------------------
def choose_cio_agent(state: TeamState) -> str:
    """Route to tool-calling or fallback agent based on model capabilities."""
    if state.get("tools_enabled") and config.supports_tool_calling(config.LLM_MODEL):
        return "cio_agent_tool_calling"
    return "cio_agent_fallback"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------
def build_graph():
    """Build the main agent graph.

    For tool-calling models: START → Router → Ticker Resolution → CIO Agent (ReAct) → END
    For non-tool-calling models: START → Router → Ticker Resolution → CIO Fallback → END

    Both paths support Investment Mode (onboarding) and Analysis Mode.
    """
    builder = StateGraph(TeamState)

    # Core nodes
    builder.add_node("router", router_node)
    builder.add_node("resolve_ticker", resolve_ticker_node)
    builder.add_node("cio_agent_tool_calling", cio_agent_tool_calling)
    builder.add_node("cio_agent_fallback", cio_agent_fallback)

    # Parallel research nodes (for fan-out synthesis)
    builder.add_node("investment_analyst", investment_analyst_node)
    builder.add_node("risk_specialist", risk_specialist_node)
    builder.add_node("cio_synthesize", cio_synthesize_node)

    # Start → Router
    builder.add_edge(START, "router")

    # Router → Ticker Resolution (always)
    builder.add_edge("router", "resolve_ticker")

    # Ticker Resolution → Choose agent based on model
    builder.add_conditional_edges(
        "resolve_ticker",
        choose_cio_agent,
        {
            "cio_agent_tool_calling": "cio_agent_tool_calling",
            "cio_agent_fallback": "cio_agent_fallback",
        },
    )

    # Both agent paths → END
    builder.add_edge("cio_agent_tool_calling", END)
    builder.add_edge("cio_agent_fallback", END)

    return builder.compile()