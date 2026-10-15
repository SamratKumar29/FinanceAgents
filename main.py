"""
main.py -- CLI entry point for FinanceAgents.

Usage:
    python main.py "Should I invest in Apple?"
    python main.py --ticker RELIANCE.NS
    python main.py --mode investment [--save profile.json]
"""
import argparse
import logging
import sys

import config
from errors import FinanceAgentsError
from graph import build_graph
from onboarding import apply_answer, is_complete, next_field, question_text
from portfolio_allocator import suggest_portfolio
from session import get_profile_summary, save_session
from state import empty_profile, initial_state


def run_analysis(query: str, settings) -> None:
    print(f"\n🔍 Researching: {query}\n")
    result = build_graph(settings).invoke(initial_state(query=query))
    print(f"📊 Ticker: {result.get('ticker') or 'N/A'}")
    if result.get("data_gaps"):
        print(f"⚠️  Unavailable data: {', '.join(result['data_gaps'])}")
    print(f"\n{'=' * 60}\n{result['final_brief']}\n{'=' * 60}")
    for t in result.get("llm_trace", []):
        print(f"(answered by {t['served_model']} after {len(t['attempts'])} attempt(s))")


def run_investment(settings, save_path: str | None) -> None:
    profile, answered = empty_profile(), set()
    print("\n💼 Investment Mode — let's build an educational model allocation.\n")
    while (field := next_field(profile, answered)) is not None:
        profile, error = apply_answer(profile, field, input(question_text(field) + " "))
        if error:
            print(f"  ↳ {error}")
            continue
        answered.add(field)
    if not is_complete(profile):
        raise SystemExit("Profile incomplete.")
    profile["completed_onboarding"] = True
    print(f"\n{get_profile_summary(profile)}\n\nGenerating the model allocation...\n")

    portfolio = suggest_portfolio(profile, settings)
    cur = portfolio["currency"]
    for p in portfolio["positions"]:
        print(f"  {p['ticker']:12s} | {p['allocation_pct']:6.2f}% | {cur} {p['amount']:>14,.2f} | {p['company']}")
    print(f"\n  Total: {cur} {portfolio['total_amount']:,.2f}\n\n  {portfolio['summary']}\n\n⚠️  {portfolio['disclaimer']}\n")
    if save_path:
        save_session(profile, filepath=save_path)
        print(f"💾 Profile saved to {save_path} (contains personal financial data)\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="FinanceAgents — LLM-assisted stock briefs")
    parser.add_argument("query", nargs="?", help="Stock analysis query (Analysis Mode)")
    parser.add_argument("--mode", choices=["analysis", "investment"])
    parser.add_argument("--ticker", help="Analyze an explicit ticker")
    parser.add_argument("--save", metavar="PATH", help="Investment Mode: save the profile JSON to PATH")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    settings = config.server_settings()
    try:
        if args.ticker:
            run_analysis(f"Analyze {args.ticker}", settings)
        elif args.mode == "investment":
            run_investment(settings, args.save)
        elif args.query:
            run_analysis(args.query, settings)
        else:
            parser.print_help()
            return 2
    except FinanceAgentsError as e:
        print(f"❌ {e.user_message}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
