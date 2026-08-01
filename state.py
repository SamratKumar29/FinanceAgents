"""
state.py -- the shared state every node reads from and writes to, plus the
safety caps. Read this file first when explaining the project: it's the
whole data model in one place.
"""
import operator
from typing import Annotated, TypedDict


class TeamState(TypedDict):
    query: str                      # the user's question, e.g. "Should I look at RELIANCE.NS?"
    ticker: str                     # extracted/confirmed ticker symbol
    investment_research: str        # filled in by the Investment Analyst node
    risk_assessment: str            # filled in by the Risk Specialist node
    final_brief: str                # filled in by the CIO's synthesis step
    # operator.add makes this a reducer: when Send() fans out to two nodes
    # in parallel, both can append to this list without overwriting each
    # other -- LangGraph merges them automatically.
    completed_agents: Annotated[list, operator.add]
    loop_count: int                 # safety cap -- see MAX_LOOPS below


# Hard cap on how many times the supervisor can loop before forcing a stop.
# You are paying per token for this project (MiMo v2.5) -- this exists
# specifically so a confused model can't quietly spend your balance in a
# retry loop. If this fires, you'll see a clear message, not a silent bill.
MAX_LOOPS = 4
