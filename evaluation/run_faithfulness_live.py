"""
evaluation/run_faithfulness_live.py -- Live brief-faithfulness run on FREE models only, over recorded fixture data.

    python -m evaluation.run_faithfulness_live

Uses only the ``:free`` models from FALLBACK_MODELS/LLM_MODEL (paid models are never called), the
deterministic tools on the recorded fixtures (src/financeagents/data/fixtures) (no Yahoo network, no web search), and records every brief with
its unsupported numbers in evaluation/results/<date>_faithfulness_live.{json,md}. The briefs are also saved to
src/financeagents/demo_data/llm_responses.json for offline demo mode.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from financeagents import config, demo
from financeagents import pipeline as graph
from financeagents.data import market as market_data
from financeagents.data import tools
from financeagents.data.fixtures import FixtureTicker
from financeagents.resolver import TickerResult
from financeagents.state import initial_state

TICKERS = ["AAPL", "NVDA", "TTWO", "RELIANCE.NS", "INFY.NS"]


def free_settings():
    s = config.server_settings()
    free = [m for m in dict.fromkeys([s.model, *s.fallback_models]) if m.endswith(":free")]
    if not free:
        raise SystemExit("No ':free' models configured in LLM_MODEL/FALLBACK_MODELS.")
    return dataclasses.replace(s, model=free[0], fallback_models=tuple(free[1:]), max_attempts=len(free))


def main() -> int:
    market_data.yf.Ticker = FixtureTicker
    settings = free_settings()
    rows = []
    for t in TICKERS:
        info = tools.fetch_info(t)

        def resolver(q, s, t=t, info=info):
            return TickerResult(t, info.get("longName") or t, info.get("exchange") or "fixture", "high", "",
                                info.get("currency") or "USD", "fixed")

        try:
            out = graph.build_graph(settings, resolver=resolver).invoke(initial_state(f"Analyze {t}"))
            f = out["faithfulness"]
            rows.append({"ticker": t, "served_model": out["llm_trace"][0]["served_model"],
                         "attempts": out["llm_trace"][0]["attempts"], "numbers_checked": f["numbers_checked"],
                         "unsupported": f["unsupported"], "unsupported_rate": f["unsupported_rate"],
                         "brief": out["final_brief"]})
        except Exception as e:  # record failures instead of hiding them
            rows.append({"ticker": t, "error": f"{type(e).__name__}: {e}"[:300]})
        print(t, {k: v for k, v in rows[-1].items() if k not in ("brief", "attempts", "unsupported")})

    ok = [r for r in rows if "error" not in r]
    checked = sum(r["numbers_checked"] for r in ok)
    unsupported = sum(len(r["unsupported"]) for r in ok)
    summary = {"date": date.today().isoformat(), "models_allowed": settings.model_chain, "runs": len(rows),
               "successful_runs": len(ok), "numbers_checked": checked, "unsupported_numbers": unsupported,
               "unsupported_rate": round(unsupported / checked, 3) if checked else None}
    out_dir = ROOT / "evaluation" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{summary['date']}_faithfulness_live"
    stem.with_suffix(".json").write_text(json.dumps({"summary": summary, "runs": rows}, indent=1) + "\n")
    md = [f"## Live faithfulness ({summary['date']}, free models only, fixture data)", "",
          "| Ticker | Model | Numbers | Unsupported | Examples |", "|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            md.append(f"| {r['ticker']} | error | - | - | {r['error'][:60]} |")
        else:
            ex = ", ".join(u["raw"] for u in r["unsupported"][:4])
            md.append(f"| {r['ticker']} | {r['served_model']} | {r['numbers_checked']} | {len(r['unsupported'])} | {ex} |")
    md += ["", f"Overall: {unsupported}/{checked} numbers unsupported "
               f"({summary['unsupported_rate']}) across {len(ok)}/{len(rows)} successful runs.", ""]
    stem.with_suffix(".md").write_text("\n".join(md))
    demo_file = demo.RESPONSES
    demo_file.parent.mkdir(exist_ok=True)
    existing = json.loads(demo_file.read_text(encoding="utf-8")) if demo_file.exists() else {}
    recorded = existing.get("synthesis", {})
    recorded.update({r["ticker"]: {"brief": r["brief"], "model": r["served_model"], "recorded_on": summary["date"]}
                     for r in ok})  # merge: a rate-limited run never removes earlier recordings
    demo_file.write_text(json.dumps({"synthesis": recorded}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
