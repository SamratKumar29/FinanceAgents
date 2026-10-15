"""
portfolio_allocator.py -- Educational model allocation from a user profile.

Weights come from ``portfolio.allocator`` (rule-based buckets + inverse volatility over a curated,
Yahoo-verified universe) and are validated in code. The LLM only writes one-sentence rationales
from the computed metrics; if it fails, template rationales are used and the allocation is unchanged.
"""
from __future__ import annotations

import json
import logging
from typing import Callable, Optional

import yfinance as yf

from analytics.risk import annualized_volatility, daily_returns
from errors import DataUnavailableError, LLMError
from llm import Settings, complete
from portfolio.allocator import BUCKET_WEIGHTS, build_allocation, validate_allocation
from state import UserProfile

log = logging.getLogger(__name__)

DISCLAIMER = (
    "Educational model allocation produced by fixed rules (risk buckets + inverse volatility) from stated "
    "preferences -- not personalized financial advice. Past volatility does not predict future returns."
)

RATIONALE_PROMPT = """You write one-sentence, factual descriptions for positions in an educational model
allocation. The weights are already fixed by rules; do not suggest changes or give buy/sell advice.
Use only the metrics provided. Respond with ONLY a JSON object mapping each ticker to its sentence."""


def extract_json_object(text: str) -> dict:
    """First JSON object in ``text`` (tolerates code fences and prose around it)."""
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("top-level JSON is not an object")
    return obj


def normalize_allocations(positions: list[dict], budget: float) -> list[dict]:
    """Scale weights to sum to 100 and set amounts so they sum exactly to ``budget`` (2 dp)."""
    total = sum(p["allocation_pct"] for p in positions)
    result = []
    for p in positions:
        q = dict(p)
        q["allocation_pct"] = round(p["allocation_pct"] * 100 / total, 2)
        q["amount"] = round(budget * p["allocation_pct"] / total, 2)
        result.append(q)
    # put rounding residue on the largest position
    big = max(result, key=lambda x: x["amount"])
    big["amount"] = round(big["amount"] + budget - sum(x["amount"] for x in result), 2)
    big["allocation_pct"] = round(big["allocation_pct"] + 100 - sum(x["allocation_pct"] for x in result), 2)
    return result


def yahoo_volatility(ticker: str) -> Optional[float]:
    """Annualised 1-year volatility from Yahoo closes, or None when unavailable."""
    try:
        prices = yf.Ticker(ticker).history(period="1y")["Close"]
        return annualized_volatility(daily_returns(prices))
    except Exception as e:  # network/parse errors make the instrument ineligible, not fatal
        log.info("volatility unavailable for %s: %s", ticker, e)
        return None


def template_rationale(p: dict) -> str:
    return (f"{p['company']}: {p['category'].replace('_', '-')} holding; 1-year volatility "
            f"{p['volatility_pct']:.1f}% (lower volatility receives more weight within its bucket).")


def _llm_rationales(positions: list[dict], profile: UserProfile, settings: Settings) -> dict[str, str]:
    tickers = {p["ticker"] for p in positions}

    def validator(text: str) -> None:
        obj = extract_json_object(text)
        if not tickers <= set(obj) or not all(isinstance(obj[t], str) and obj[t].strip() for t in tickers):
            raise ValueError("missing rationale for some tickers")

    facts = [{k: p[k] for k in ("ticker", "company", "category", "allocation_pct", "volatility_pct")}
             for p in positions]
    result = complete(
        settings,
        [{"role": "system", "content": RATIONALE_PROMPT},
         {"role": "user", "content": json.dumps({"risk_tolerance": profile.get("risk_tolerance"),
                                                 "horizon": profile.get("investment_horizon"),
                                                 "positions": facts})}],
        validator=validator,
        max_tokens=1500,
    )
    obj = extract_json_object(result.text)
    return {t: str(obj[t]).strip()[:300] for t in tickers}


def suggest_portfolio(
    profile: UserProfile,
    settings: Optional[Settings],
    volatility: Callable[[str], Optional[float]] = yahoo_volatility,
) -> dict:
    """Return ``{positions, summary, disclaimer, total_amount, currency, rationale_source}``."""
    budget = float(profile.get("budget") or 0)
    if budget <= 0:
        raise ValueError("Profile budget must be positive.")
    risk = profile.get("risk_tolerance") or "moderate"
    n = int(profile.get("num_positions") or 5)
    markets = profile.get("preferred_markets") or []
    currency = profile.get("base_currency") or "USD"

    try:
        raw = build_allocation(risk, n, markets, volatility)
    except ValueError as e:
        raise DataUnavailableError(f"Cannot build allocation: {e}") from e
    positions = normalize_allocations(raw, budget)
    validate_allocation(positions, budget, markets)

    source = "template"
    rationales: dict[str, str] = {}
    if settings is not None:
        try:
            rationales = _llm_rationales(positions, profile, settings)
            source = "llm"
        except LLMError as e:
            log.warning("rationale generation failed; using templates: %s", e)
    for p in positions:
        p["rationale"] = rationales.get(p["ticker"]) or template_rationale(p)

    weights = ", ".join(f"{b.replace('_', '-')} {w:.0f}%" for b, w in BUCKET_WEIGHTS[risk].items())
    per = " per month" if profile.get("budget_type") == "monthly" else ""
    summary = (f"Rule-based {risk} mix ({weights}) across {len(positions)} instruments; within each bucket, "
               f"weights are inverse to 1-year volatility. Amounts are in {currency}{per}.")
    if len(positions) < n:
        summary += f" Only {len(positions)} eligible instruments were available for the selected markets."
    return {
        "positions": positions,
        "summary": summary,
        "disclaimer": DISCLAIMER,
        "total_amount": round(sum(p["amount"] for p in positions), 2),
        "currency": currency,
        "rationale_source": source,
    }
