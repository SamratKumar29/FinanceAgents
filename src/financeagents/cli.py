"""
financeagents.cli -- the `financeagents` command.

Usage:
    financeagents "Should I invest in Apple?"
    financeagents --ticker RELIANCE.NS
    financeagents --mode investment [--save profile.json]
    financeagents --demo "How is Infosys doing?"
"""
import argparse
import sys

from financeagents import config, demo, tracing
from financeagents.errors import FinanceAgentsError
from financeagents.onboarding import apply_answer, is_complete, next_field, question_text
from financeagents.pipeline import build_graph
from financeagents.pipeline import run_analysis as run_graph
from financeagents.portfolio.service import suggest_portfolio
from financeagents.session import get_profile_summary, save_session
from financeagents.state import empty_profile, initial_state


def run_analysis(query: str, settings) -> None:
    print(f"\n🔍 Researching: {query}\n")
    result = run_graph(build_graph(settings), initial_state(query=query))
    print(f"📊 Ticker: {result.get('ticker') or 'N/A'}")
    if result.get("data_gaps"):
        print(f"⚠️  Unavailable data: {', '.join(result['data_gaps'])}")
    print(f"\n{'=' * 60}\n{result['final_brief']}\n{'=' * 60}")
    tr = result["trace"]
    print(f"(run {tr['run_id']}: {tr['total_latency_ms']:.0f} ms, {tr['total_tokens']} tokens, "
          f"models {', '.join(tr['models_used']) or '-'})")
    f = result.get("faithfulness") or {}
    if f.get("unsupported"):
        print(f"⚠️  {len(f['unsupported'])} number(s) not found in the source data: "
              + ", ".join(u["raw"] for u in f["unsupported"][:5]))


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
    parser.add_argument("--demo", action="store_true", help="Offline demo: recorded data and responses, no API key")
    args = parser.parse_args(argv)
    tracing.configure_logging()

    if args.demo or demo.requested():
        demo.enable()
        settings = demo.settings()
        print(f"[demo mode: recorded data as of {demo.data_date()}, no network]")
    else:
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
