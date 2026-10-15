"""
ticker_resolver.py -- Resolve natural-language company references to stock tickers.

Order: LLM extraction (company + explicit ticker) -> validate explicit ticker -> Yahoo search on
the company name -> word-boundary static map -> all-caps regex (with stopwords). Every candidate is
validated against Yahoo before it is returned; weak matches abstain instead of guessing.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import yfinance as yf

from errors import LLMAuthError, LLMBudgetError, LLMError
from llm import Settings, complete

log = logging.getLogger(__name__)


@dataclass
class TickerResult:
    ticker: str
    company: str
    exchange: str
    confidence: str  # "high", "medium", "low"
    context: str     # why the user asked (e.g. "GTA 6 launch catalyst")
    currency: str
    source: str = ""  # which resolver stage produced it
    candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def __str__(self) -> str:
        return f"{self.ticker} ({self.company}) on {self.exchange}"


SYSTEM_PROMPT = """You are a stock ticker resolver. Given a user's natural language message,
extract the company or stock they are asking about, and any relevant context.

Respond with ONLY a JSON object (no markdown, no explanation):
{
    "company_name": "the company's common name (max 5 words)",
    "ticker_if_known": "just the ticker symbol if mentioned (e.g. TSLA, AAPL) or empty string",
    "context": "brief reason for the query (max 10 words)"
}

Rules:
- If the user mentions a clear ticker symbol (all-caps like TSLA, AAPL, INFY), put it in ticker_if_known
- If they mention a company name (like "Take Two", "Nvidia", "Reliance"), put it in company_name
- The "context" field captures WHY they're asking, as a short phrase
- Do NOT fabricate information; if unsure, leave fields empty"""

EMPTY_EXTRACTION = {"company_name": "", "ticker_if_known": "", "context": ""}

# Known company name -> ticker. Matched on word boundaries only.
KNOWN_COMPANIES = {
    "take two": "TTWO",
    "take-two": "TTWO",
    "take two interactive": "TTWO",
    "rockstar games": "TTWO",
    "nvidia": "NVDA",
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta platforms": "META",
    "facebook": "META",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata steel": "TATASTEEL.NS",
    "infosys": "INFY.NS",
    "hdfc bank": "HDFCBANK.NS",
    "icici bank": "ICICIBANK.NS",
    "wipro": "WIPRO.NS",
    "byd": "1211.HK",
    "samsung electronics": "005930.KS",
    "sony": "SONY",
    "nintendo": "NTDOY",
    "electronic arts": "EA",
    "ubisoft": "UBI.PA",
}

# All-caps words that are not tickers in ordinary finance questions.
STOPWORDS = frozenset({
    "CEO", "CFO", "CTO", "COO", "ETF", "ETFS", "PE", "EPS", "GTA", "IPO", "AI", "USA", "US", "UK", "EU",
    "USD", "INR", "EUR", "GBP", "GDP", "SIP", "NAV", "ROI", "ROE", "YOY", "QOQ", "FY", "Q1", "Q2", "Q3",
    "Q4", "NYSE", "NSE", "BSE", "SEC", "SEBI", "API", "LLM", "OK", "IT", "IS", "I", "A", "THE", "AND",
    "OR", "FOR", "TO", "BUY", "SELL", "HOLD", "DCF", "EBITDA", "PEG", "ESG", "FD", "PPF", "NRI", "YTD",
})

# Yahoo exchange suffixes; any other single-letter suffix is a share class (BRK.B -> BRK-B).
EXCHANGE_SUFFIXES = frozenset({
    "NS", "BO", "L", "HK", "KS", "KQ", "PA", "DE", "F", "AS", "MI", "MC", "SW", "TO", "V", "AX", "T",
    "SS", "SZ", "SI", "TW", "SA", "MX", "ST", "OL", "CO", "HE", "IR", "BR", "VI", "LS", "NZ", "JK",
})

INDIA_HINTS = ("india", "reliance", "tata", "infosys", "hdfc", "icici", "wipro", "nifty", "sensex")

_TICKER_RE = re.compile(r"^([A-Z0-9]{1,10})(?:[.\-]([A-Z]{1,3}))?")


def normalize_ticker(raw: str) -> str:
    """Upper-case, strip decorations, keep exchange suffixes and share classes."""
    m = _TICKER_RE.match((raw or "").strip().upper())
    if not m:
        return ""
    base, suffix = m.group(1), m.group(2)
    if not suffix:
        return base
    return f"{base}.{suffix}" if suffix in EXCHANGE_SUFFIXES else f"{base}-{suffix}"


def _parse_extraction(text: str) -> dict:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if not m:
        raise ValueError("no JSON object in output")
    parsed = json.loads(m.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("JSON is not an object")
    return {k: str(parsed.get(k) or "").strip() for k in EMPTY_EXTRACTION}


def llm_extract(query: str, settings: Optional[Settings]) -> dict:
    """LLM extraction. Auth/budget errors propagate; transient failures degrade to fallbacks."""
    if settings is None or not settings.api_key:
        return dict(EMPTY_EXTRACTION)
    try:
        result = complete(
            settings,
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": query}],
            validator=_parse_extraction,
            temperature=0,
            max_tokens=1000,  # reasoning models need headroom before the JSON appears
        )
    except (LLMAuthError, LLMBudgetError):
        raise
    except LLMError as e:
        log.warning("ticker extraction LLM failed; using fallbacks: %s", e)
        return dict(EMPTY_EXTRACTION)
    return _parse_extraction(result.text)


def known_company_matches(query: str) -> list[str]:
    q = query.lower()
    hits = []
    for name in sorted(KNOWN_COMPANIES, key=len, reverse=True):
        if re.search(rf"(?<![a-z0-9]){re.escape(name)}(?![a-z0-9])", q):
            ticker = KNOWN_COMPANIES[name]
            if ticker not in hits:
                hits.append(ticker)
            q = q.replace(name, " ")  # prevent shorter overlapping names matching too
    return hits


def regex_candidates(query: str) -> list[str]:
    out = []
    for m in re.finditer(r"(?<![A-Za-z0-9])([A-Z]{2,5}(?:\.[A-Z]{1,2})?)(?![A-Za-z0-9])", query):
        token = m.group(1)
        if token.split(".")[0] in STOPWORDS:
            continue
        t = normalize_ticker(token)
        if t and t not in out:
            out.append(t)
    return out


def lookup_info(ticker: str) -> Optional[dict]:
    """Return Yahoo info for a real listing, else None."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        log.info("ticker lookup failed for %s: %s", ticker, e)
        return None
    return info if info.get("shortName") or info.get("longName") else None


def search_yahoo(company_name: str, max_results: int = 10) -> list[dict]:
    if not company_name:
        return []
    try:
        quotes = yf.Search(company_name, max_results=max_results).quotes
    except Exception as e:
        log.info("Yahoo search failed for %r: %s", company_name, e)
        return []
    return [
        {
            "symbol": q.get("symbol", ""),
            "shortname": q.get("shortname", "") or "",
            "longname": q.get("longname", "") or "",
            "exchange": q.get("exchange", "") or "",
        }
        for q in quotes
        if q.get("quoteType") == "EQUITY"
    ]


MIN_MATCH_SCORE = 8  # a name match is required; exchange preference alone never wins


def score_match(r: dict, company_hint: str) -> int:
    hint = (company_hint or "").lower().strip()
    if not hint:
        return 0
    score = 0
    if hint in r.get("longname", "").lower():
        score += 10
    elif hint in r.get("shortname", "").lower():
        score += 8
    if score and r.get("exchange", "").upper() in ("NMS", "NYQ", "NASDAQ", "NYSE", "NSI", "NSE", "BSE"):
        score += 5
    return score


def pick_best_match(results: list[dict], company_hint: str) -> tuple[Optional[dict], list[str]]:
    """Best search result above the threshold (or None) plus up to 3 candidate symbols."""
    scored = sorted(((score_match(r, company_hint), r) for r in results), key=lambda x: x[0], reverse=True)
    good = [(s, r) for s, r in scored if s >= MIN_MATCH_SCORE]
    candidates = [r["symbol"] for _, r in good[:3]]
    return (good[0][1] if good else None), candidates


def _currency(ticker: str, info: dict) -> str:
    cur = info.get("currency")
    if isinstance(cur, str) and cur:
        return cur.upper()
    return "INR" if ticker.endswith((".NS", ".BO")) else "USD"


def _result(ticker: str, info: dict, confidence: str, context: str, source: str, candidates=None) -> TickerResult:
    return TickerResult(
        ticker=ticker,
        company=info.get("longName") or info.get("shortName") or ticker,
        exchange=info.get("exchange") or "Unknown",
        confidence=confidence,
        context=context,
        currency=_currency(ticker, info),
        source=source,
        candidates=list(candidates or [ticker]),
    )


def resolve_ticker(
    query: str,
    settings: Optional[Settings] = None,
    *,
    lookup: Callable[[str], Optional[dict]] = lookup_info,
    search: Callable[[str], list[dict]] = search_yahoo,
) -> Optional[TickerResult]:
    """Resolve a query to a validated ticker, or None when nothing is confidently identified."""
    extracted = llm_extract(query, settings)
    company_name = extracted["company_name"]
    context = extracted["context"]
    explicit = normalize_ticker(extracted["ticker_if_known"])

    # 1. Explicit ticker from the LLM, validated; try the NSE listing for Indian companies.
    if explicit:
        tries = [explicit]
        if "." not in explicit and any(h in company_name.lower() for h in INDIA_HINTS):
            tries.append(f"{explicit}.NS")
        for t in tries:
            info = lookup(t)
            if info:
                return _result(t, info, "high", context, "llm_ticker")

    # 2. Yahoo search on the extracted company name.
    if company_name:
        best, candidates = pick_best_match(search(company_name), company_name)
        if best:
            info = lookup(best["symbol"]) or {
                "longName": best["longname"] or best["shortname"], "exchange": best["exchange"]
            }
            confidence = "high" if company_name.lower() in best["longname"].lower() else "medium"
            return _result(best["symbol"], info, confidence, context, "yahoo_search", candidates)

    # 3. Static map on word boundaries.
    for t in known_company_matches(query):
        info = lookup(t)
        if info:
            return _result(t, info, "medium", context, "static_map")

    # 4. All-caps tokens that look like tickers (stopwords removed).
    for t in regex_candidates(query):
        info = lookup(t)
        if info:
            return _result(t, info, "low", context, "regex")

    return None
