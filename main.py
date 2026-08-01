"""
main.py -- CLI to test the agent team before wrapping any UI around it.
Get this working and correct first; UI is optional polish given the time
budget.

    python main.py "Should I look into RELIANCE.NS?"
"""
import sys
import re

from graph import build_graph


def extract_ticker(query: str) -> str:
    """Very simple ticker extraction -- looks for an all-caps token,
    optionally with .NS/.BO suffix for Indian exchanges. Good enough for
    a 1-day build; make this smarter later if you have time (e.g. an LLM
    call to resolve a company name to a ticker)."""
    match = re.search(r'\b([A-Z]{2,10}(?:\.NS|\.BO)?)\b', query)
    return match.group(1) if match else query.strip()


def main():
    query = " ".join(sys.argv[1:]) or "Analyze RELIANCE.NS"
    ticker = extract_ticker(query)
    print(f"Analyzing: {ticker}\n")

    graph = build_graph()
    result = graph.invoke({
        "query": query,
        "ticker": ticker,
        "investment_research": "",
        "risk_assessment": "",
        "final_brief": "",
        "completed_agents": [],
        "loop_count": 0,
    })

    print("=" * 60)
    print("FINAL BRIEF")
    print("=" * 60)
    print(result["final_brief"])
    print()
    print(f"(Agents that ran: {result['completed_agents']})")


if __name__ == "__main__":
    main()
