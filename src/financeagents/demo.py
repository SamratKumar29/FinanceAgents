"""
demo.py -- Offline demo mode (``FINANCEAGENTS_DEMO=1`` / ``make demo``): no API key, no network.

* Yahoo data comes from the recorded fixtures in data/fixtures/ (recorded on the date shown in the UI banner).
* Yahoo search is replaced by a lookup over fixture company names; web search is disabled.
* LLM calls go to ``DemoClient``: briefs recorded from a free model on the same fixtures
  (demo_data/llm_responses.json, refreshed by `make eval-live`), or a deterministic template otherwise.
"""
from __future__ import annotations

import json
import os
import re
import types
from pathlib import Path
from types import SimpleNamespace

DEMO_BASE_URL = "demo://recorded"
RESPONSES = Path(__file__).resolve().parent / "demo_data" / "llm_responses.json"
_saved: dict = {}


def requested() -> bool:
    return os.getenv("FINANCEAGENTS_DEMO", "").strip() in ("1", "true", "yes")


def settings():
    from financeagents.llm import Settings

    return Settings(api_key="demo", base_url=DEMO_BASE_URL, model="demo-recorded", uses_server_key=True)


def data_date() -> str:
    from financeagents.data.fixtures import load

    return load("AAPL").get("recorded_at", "unknown")[:10]


def _fixture_search(company_name: str, max_results: int = 10) -> list[dict]:
    from financeagents.data.fixtures import FIXTURES

    name = company_name.lower().strip()
    out = []
    for path in sorted(FIXTURES.glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        info = d.get("info") or {}
        long_name = info.get("longName") or ""
        if name and d.get("close") and name in long_name.lower():
            out.append({"symbol": d["symbol"], "shortname": info.get("shortName") or "", "longname": long_name,
                        "exchange": info.get("exchange") or ""})
    return out[:max_results]


def enable() -> None:
    """Patch data sources for offline use (idempotent)."""
    from financeagents import resolver as ticker_resolver
    from financeagents.data import market as market_data
    from financeagents.data import tools
    from financeagents.data.fixtures import FixtureTicker
    from financeagents.errors import DataUnavailableError

    if _saved:
        return
    _saved.update(yf=market_data.yf, search=ticker_resolver.search_yahoo, web=tools.web_search)
    market_data.yf = SimpleNamespace(Ticker=FixtureTicker)
    ticker_resolver.search_yahoo = _fixture_search

    def no_web(query, max_results=5):
        raise DataUnavailableError("web search is disabled in demo mode")

    tools.web_search = no_web
    market_data.clear_cache()


def disable() -> None:
    from financeagents import resolver as ticker_resolver
    from financeagents.data import market as market_data
    from financeagents.data import tools

    if not _saved:
        return
    market_data.yf, ticker_resolver.search_yahoo, tools.web_search = _saved["yf"], _saved["search"], _saved["web"]
    _saved.clear()
    market_data.clear_cache()


def _response(text: str, model: str):
    msg = SimpleNamespace(content=text)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg, finish_reason="stop")], model=model, usage=None)


def _template_brief(payload: dict) -> str:
    from financeagents.pipeline import DISCLAIMER

    d = payload.get("tool_data", {})
    md, risk, av = d.get("market_data", {}), d.get("risk_metrics", {}), d.get("analyst_view", {})
    cur = md.get("currency") or ""
    lines = [f"**{md.get('company') or payload.get('ticker')} ({payload.get('ticker')})** — demo template brief "
             "(no recorded model response for this ticker)."]
    if md.get("current_price") is not None:
        lines.append(f"- Price {md['current_price']} {cur}; trailing P/E {md.get('pe_ratio_trailing')}; "
                     f"6-month change {md.get('price_change_6mo_pct')}%.")
    if risk:
        lines.append(f"- 1-year volatility {risk.get('annualized_volatility_pct')}%, max drawdown "
                     f"{risk.get('max_drawdown_pct')}%, beta {risk.get('beta')} vs {risk.get('benchmark_name')}.")
    if av.get("analyst_target_mean") is not None:
        lines.append(f"- Mean analyst target {av['analyst_target_mean']} {cur} "
                     f"({av.get('analyst_count')} analysts, consensus {av.get('analyst_recommendation')}).")
    lines.append(f"\n{DISCLAIMER}")
    return "\n".join(lines)


class DemoClient:
    """Replays recorded responses; mimics ``client.chat.completions.create``."""

    def __init__(self):
        self.recorded = json.loads(RESPONSES.read_text(encoding="utf-8")) if RESPONSES.exists() else {}
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, **kwargs):
        system = messages[0]["content"] if messages else ""
        user = messages[-1]["content"] if messages else ""
        if system.startswith("You are a stock ticker resolver"):
            # Defer to the deterministic resolver stages (static map, fixture search, regex).
            return _response('{"company_name": "", "ticker_if_known": "", "context": ""}', "demo-recorded")
        if "educational model" in system:
            positions = json.loads(user).get("positions", [])
            text = {p["ticker"]: f"{p['company']}: {p['category'].replace('_', '-')} holding with 1-year volatility "
                                 f"{p['volatility_pct']}%." for p in positions}
            return _response(json.dumps(text), "demo-template")
        payload = json.loads(re.sub(r"^TOOL DATA \(JSON\):\n", "", user.split("\n\n<untrusted_web_results>")[0]))
        rec = self.recorded.get("synthesis", {}).get(payload.get("ticker", ""))
        if rec:
            return _response(rec["brief"], f"{rec['model']} (recorded {rec['recorded_on']})")
        return _response(_template_brief(payload), "demo-template")
