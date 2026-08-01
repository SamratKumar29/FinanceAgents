"""
main.py -- CLI entry point for FinanceAgents v2.

Usage:
    python main.py "Should I invest in Apple?"
    python main.py --mode investment
    python main.py --ticker RELIANCE.NS
"""
import argparse
import sys
import json
from graph import build_graph
from state import initial_state, empty_profile
from portfolio_allocator import suggest_portfolio
from session import save_session, get_profile_summary


def run_analysis(query: str):
    """Run analysis mode from CLI."""
    graph = build_graph()
    print(f"\n🔍 Researching: {query}\n")
    result = graph.invoke(initial_state(query=query))
    print(f"📊 Ticker: {result.get('ticker', 'N/A')}")
    print(f"\n{'='*60}")
    print(result.get("final_brief", "No analysis available."))
    print(f"{'='*60}\n")


def run_investment():
    """Run investment mode onboarding from CLI."""
    profile = empty_profile()
    questions = [
        ("location", "Where are you based? (e.g., India, US, UK): "),
        ("base_currency", "Currency preference (INR or USD): "),
        ("budget", "Investable budget (number only): "),
        ("budget_type", "Lumpsum or monthly? "),
        ("goals", "Investment goals (comma-separated, e.g., retirement, wealth growth): "),
        ("risk_tolerance", "Risk tolerance (conservative/moderate/aggressive): "),
        ("investment_horizon", "Investment horizon (e.g., 5 years): "),
        ("num_positions", "How many positions? (number): "),
        ("preferred_markets", "Markets (US/India/both): "),
        ("existing_investments", "Existing investments (or 'none'): "),
    ]

    print("\n💼 Investment Mode — Let's build your portfolio.\n")
    for field, question in questions:
        answer = input(question).strip()

        if field == "budget":
            import re
            nums = re.findall(r'[\d,]+\.?\d*', answer.replace(",", ""))
            profile["budget"] = float(nums[0]) if nums else 0
        elif field == "num_positions":
            import re
            nums = re.findall(r'\d+', answer)
            profile["num_positions"] = int(nums[0]) if nums else 5
        elif field in ("goals", "preferred_markets", "preferred_sectors"):
            profile[field] = [x.strip().lower() for x in answer.split(",")]
        elif field == "existing_investments":
            profile[field] = [] if answer.lower() in ("none", "no") else [{"description": answer}]
        elif field == "location":
            profile["location"] = answer.title()
            loc = answer.lower()
            if "india" in loc:
                profile["base_currency"] = "INR"
            elif "us" in loc or "usa" in loc or "united states" in loc:
                profile["base_currency"] = "USD"
        elif field == "base_currency":
            cur = answer.upper().strip()
            profile["base_currency"] = "INR" if "INR" in cur else "USD" if "USD" in cur else cur[:3]
        else:
            profile[field] = answer

    profile["completed_onboarding"] = True
    print(f"\n{get_profile_summary(profile)}\n")
    print("Generating your portfolio...\n")

    portfolio = suggest_portfolio(profile)
    currency = portfolio.get("currency", "USD")
    for p in portfolio.get("positions", []):
        print(f"  {p.get('ticker', 'N/A'):8s} | {p.get('allocation_pct', 0):5.1f}% | {currency} {p.get('amount', 0):>12,.0f} | {p.get('company', '')}")
    print(f"\n  Total: {currency} {portfolio.get('total_amount', 0):,.0f}")
    print(f"\n  {portfolio.get('summary', '')}")
    print(f"\n⚠️  {portfolio.get('disclaimer', '')}\n")

    # Save profile
    save_session(profile, filepath="profile.json")
    print("💾 Profile saved to profile.json\n")


def main():
    parser = argparse.ArgumentParser(description="FinanceAgents v2 — AI Investment Analysis Team")
    parser.add_argument("query", nargs="?", help="Stock analysis query (Analysis Mode)")
    parser.add_argument("--mode", choices=["analysis", "investment"], help="Mode to run")
    parser.add_argument("--ticker", help="Direct ticker to analyze (bypasses natural language)")
    args = parser.parse_args()

    if args.ticker:
        run_analysis(f"Analyze {args.ticker}")
    elif args.mode == "investment":
        run_investment()
    elif args.query:
        run_analysis(args.query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()