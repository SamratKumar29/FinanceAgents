"""
graph.py -- the CIO supervisor and the graph wiring. This is what makes
it a "team" rather than three separate scripts: the CIO fans out to both
specialists in PARALLEL (their analyses don't depend on each other), waits
for both, then synthesizes one brief. This is the file to walk through
when explaining "how do the agents work together" in an interview.

Verify this against your installed LangGraph version's docs
(pip show langgraph, then check the matching docs) -- the Send()/fan-out
API is confirmed current as of writing, but LangGraph ships new minor
versions often enough that exact syntax can shift.
"""
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from openai import OpenAI

import config
from state import TeamState, MAX_LOOPS
import investment_analyst
import risk_specialist


SYNTHESIS_PROMPT = """You are the Chief Investment Officer. You have
received research from your Investment Analyst and your Risk Management
Specialist on the same opportunity. Synthesize both into one coherent
brief for the user:

1. Summarize the investment case (from the analyst's research)
2. Summarize the risk profile (from the risk specialist's assessment)
3. Lay out the key trade-offs -- do NOT issue a single confident
   buy/sell/hold call as if it fits everyone. That depends on the user's
   own risk tolerance and goals, which you don't know.
4. End with exactly this line: "This is general informational analysis,
   not personalized financial advice -- consult a licensed financial
   advisor before making investment decisions."
"""


def investment_analyst_node(state: TeamState) -> dict:
    result = investment_analyst.run(state["ticker"])
    return {"investment_research": result, "completed_agents": ["investment_analyst"]}


def risk_specialist_node(state: TeamState) -> dict:
    result = risk_specialist.run(state["ticker"])
    return {"risk_assessment": result, "completed_agents": ["risk_specialist"]}


def cio_dispatch(state: TeamState) -> list:
    """The CIO's first action: fan out to both specialists in parallel via
    Send(). No data dependency between them, so no reason to run them
    sequentially and double the latency."""
    return [
        Send("investment_analyst", state),
        Send("risk_specialist", state),
    ]


def cio_synthesize(state: TeamState) -> dict:
    """The CIO's second action: LangGraph only reaches this node once
    BOTH fanned-out branches have completed -- that's the fan-in half of
    fan-out/fan-in. Combine both specialists' findings into one brief."""
    if state["loop_count"] >= MAX_LOOPS:
        return {"final_brief": "Stopped: hit the safety loop cap before "
                                "a clean synthesis. Check investment_research "
                                "and risk_assessment in the raw state directly."}

    client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)
    user_content = (
        f"INVESTMENT ANALYST'S RESEARCH:\n{state['investment_research']}\n\n"
        f"RISK SPECIALIST'S ASSESSMENT:\n{state['risk_assessment']}"
    )
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    return {"final_brief": response.choices[0].message.content,
            "loop_count": state["loop_count"] + 1}


def build_graph():
    builder = StateGraph(TeamState)

    builder.add_node("investment_analyst", investment_analyst_node)
    builder.add_node("risk_specialist", risk_specialist_node)
    builder.add_node("cio_synthesize", cio_synthesize)

    # START fans out to both specialists via Send() -- the confirmed
    # current LangGraph pattern for parallel dispatch from a supervisor.
    builder.add_conditional_edges(START, cio_dispatch)

    builder.add_edge("investment_analyst", "cio_synthesize")
    builder.add_edge("risk_specialist", "cio_synthesize")
    builder.add_edge("cio_synthesize", END)

    return builder.compile()
