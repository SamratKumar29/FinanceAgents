"""
ticker_resolver.py -- Resolves natural language company references to stock tickers.

Uses an LLM to extract the intended company name from user queries, then
yfinance's Search API to find the correct ticker symbol. This fixes the
"Take Two before GTA 6" → TTWO problem that regex extraction cannot handle.

Usage:
    result = resolve_ticker("I was thinking of investing in Take Two before GTA 6")
    # → TickerResult(ticker="TTWO", company="Take-Two Interactive Software, Inc.", exchange="NMS")
"""
import re
import json
from dataclasses import dataclass, asdict
from typing import Optional

import yfinance as yf
from openai import OpenAI
import config


@dataclass
class TickerResult:
    ticker: str
    company: str
    exchange: str
    confidence: str  # "high", "medium", "low"
    context: str     # any additional context from the query (e.g., "GTA 6 launch catalyst")
    currency: str    # "USD", "INR", etc.

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
- If they mention both, include both
- The "context" field captures WHY they're asking — keep it to a short phrase (e.g. "GTA 6 launch catalyst")
- Do NOT write long explanations or sentences in any field
- Do NOT fabricate information — if unsure, leave fields empty"""


def _llm_extract(query: str) -> dict:
    """Use LLM to extract company name and context from natural language."""
    client = OpenAI(base_url=config.OPENAI_BASE_URL, api_key=config.OPENAI_API_KEY)

    # Check if API key is valid
    if not config.OPENAI_API_KEY:
        return {"company_name": "", "ticker_if_known": "", "context": ""}

    try:
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=400,
        )
        raw_content = response.choices[0].message.content
        if not raw_content:
            return {"company_name": "", "ticker_if_known": "", "context": ""}
        text = raw_content.strip()
        # Strip markdown code fences if present
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        # Try to find the JSON object even if truncated
        json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if json_match:
            text = json_match.group(0)
        parsed = json.loads(text)
        # Ensure no None values
        return {k: (v or "") for k, v in parsed.items()}
    except (json.JSONDecodeError, Exception) as e:
        return {"company_name": "", "ticker_if_known": "", "context": ""}


# Known company name → ticker mappings for common cases (fallback when LLM fails)
KNOWN_COMPANIES = {
    "take two": "TTWO",
    "take-two": "TTWO",
    "take two interactive": "TTWO",
    "rockstar": "TTWO",
    "nvidia": "NVDA",
    "apple": "AAPL",
    "tesla": "TSLA",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "reliance": "RELIANCE.NS",
    "reliance industries": "RELIANCE.NS",
    "tata motors": "TATAMOTORS.NS",
    "tata": "TATAMOTORS.NS",
    "infosys": "INFY.NS",
    "hdfc": "HDFCBANK.NS",
    "icici": "ICICIBANK.NS",
    "wipro": "WIPRO.NS",
    "byd": "1211.HK",
    "samsung": "005930.KS",
    "sony": "SONY",
    "nintendo": "NTDOY",
    "electronic arts": "EA",
    "ubisoft": "UBI.PA",
    "activision": "ATVI",
}


def _known_company_fallback(query: str) -> Optional[str]:
    """Check if the query mentions a known company name."""
    query_lower = query.lower()
    # Check from longest to shortest keys for best match
    for name in sorted(KNOWN_COMPANIES.keys(), key=len, reverse=True):
        if name in query_lower:
            return KNOWN_COMPANIES[name]
    return None


def _regex_fallback(query: str) -> Optional[str]:
    """Fallback: look for an all-caps ticker pattern in the query."""
    match = re.search(r'\b([A-Z]{2,10}(?:\.NS|\.BO)?)\b', query)
    return match.group(1) if match else None


def _search_yfinance(company_name: str, max_results: int = 10) -> list[dict]:
    """Search yfinance for matching equities."""
    if not company_name:
        return []
    try:
        search = yf.Search(company_name, max_results=max_results)
        results = []
        for quote in search.quotes:
            if quote.get("quoteType") == "EQUITY":
                results.append({
                    "symbol": quote.get("symbol", ""),
                    "shortname": quote.get("shortname", ""),
                    "longname": quote.get("longname", ""),
                    "exchange": quote.get("exchange", ""),
                    "quoteType": quote.get("quoteType", ""),
                })
        return results
    except Exception:
        return []


def _detect_currency(ticker: str, exchange: str) -> str:
    """Detect the currency based on exchange or ticker suffix."""
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "INR"
    exchange = exchange.upper() if exchange else ""
    if exchange in ("NSI", "BSE", "NSE"):
        return "INR"
    return "USD"


def _pick_best_match(results: list[dict], company_hint: str) -> Optional[dict]:
    """Pick the best match from yfinance search results."""
    if not results:
        return None

    company_lower = company_hint.lower() if company_hint else ""

    # Score each result
    scored = []
    for r in results:
        score = 0
        longname = (r.get("longname") or "").lower()
        shortname = (r.get("shortname") or "").lower()

        # Exact or partial name match
        if company_lower and company_lower in longname:
            score += 10
        if company_lower and company_lower in shortname:
            score += 8

        # Prefer major exchanges
        exchange = r.get("exchange", "").upper()
        if exchange in ("NMS", "NYQ", "NASDAQ", "NYSE"):
            score += 5
        elif exchange in ("NSI", "NSE", "BSE"):
            score += 5
        elif exchange in ("LON", "FRA", "TSE"):
            score += 2

        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1] if scored else None


def resolve_ticker(query: str) -> Optional[TickerResult]:
    """
    Main entry point: resolve a natural language query to a stock ticker.

    Steps:
    1. LLM extracts company name and context from the query
    2. If a clear ticker was mentioned, validate it exists via yfinance
    3. Otherwise, search yfinance with the company name
    4. Return the best match

    Returns None if no resolution is possible.
    """
    # Step 1: LLM extraction
    extracted = _llm_extract(query)
    company_name = extracted.get("company_name", "").strip()
    # Clean ticker: extract just the symbol part (e.g. "TTWO (NASDAQ)" → "TTWO")
    raw_ticker = (extracted.get("ticker_if_known") or "").strip()
    ticker_match = re.match(r'^([A-Z]{1,10}(?:\.NS|\.BO)?)', raw_ticker.upper())
    known_ticker = ticker_match.group(1) if ticker_match else raw_ticker.upper().split()[0] if raw_ticker else ""
    context = (extracted.get("context") or "").strip()

    # Step 2: If a ticker was explicitly mentioned, validate it
    if known_ticker:
        # Add exchange suffix for Indian stocks if needed
        if not known_ticker.endswith((".NS", ".BO")):
            # Check if company name suggests Indian stock
            if any(ind in (company_name + " " + query).lower()
                   for ind in ["india", "reliance", "tata", "infosys", "hdfc", "icici", "wipro"]):
                known_ticker += ".NS"

        try:
            stock = yf.Ticker(known_ticker)
            info = stock.info
            if info and info.get("shortName"):
                return TickerResult(
                    ticker=known_ticker,
                    company=info.get("longName", info.get("shortName", known_ticker)),
                    exchange=info.get("exchange", "Unknown"),
                    confidence="high",
                    context=context,
                    currency=_detect_currency(known_ticker, info.get("exchange", "")),
                )
        except Exception:
            pass

    # Step 3: Search yfinance with company name
    if company_name:
        search_results = _search_yfinance(company_name)
        best = _pick_best_match(search_results, company_name)
        if best:
            ticker = best["symbol"]
            company = best.get("longname") or best.get("shortname") or ticker
            exchange = best.get("exchange", "Unknown")

            # Determine confidence
            confidence = "medium"
            company_lower = company_name.lower()
            if company_lower in (best.get("longname", "").lower()):
                confidence = "high"

            return TickerResult(
                ticker=ticker,
                company=company,
                exchange=exchange,
                confidence=confidence,
                context=context,
                currency=_detect_currency(ticker, exchange),
            )

    # Step 4: Known company name fallback
    known_ticker = _known_company_fallback(query)
    if known_ticker:
        try:
            stock = yf.Ticker(known_ticker)
            info = stock.info
            if info and info.get("shortName"):
                return TickerResult(
                    ticker=known_ticker,
                    company=info.get("longName", info.get("shortName", known_ticker)),
                    exchange=info.get("exchange", "Unknown"),
                    confidence="medium",
                    context=context,
                    currency=_detect_currency(known_ticker, info.get("exchange", "")),
                )
        except Exception:
            pass

    # Step 5: Regex fallback
    fallback_ticker = _regex_fallback(query)
    if fallback_ticker:
        try:
            stock = yf.Ticker(fallback_ticker)
            info = stock.info
            if info and info.get("shortName"):
                return TickerResult(
                    ticker=fallback_ticker,
                    company=info.get("longName", info.get("shortName", fallback_ticker)),
                    exchange=info.get("exchange", "Unknown"),
                    confidence="low",
                    context=context,
                    currency=_detect_currency(fallback_ticker, info.get("exchange", "")),
                )
        except Exception:
            pass

    return None


def resolve_ticker_or_raise(query: str) -> TickerResult:
    """Resolve ticker, raising ValueError if resolution fails."""
    result = resolve_ticker(query)
    if result is None:
        raise ValueError(
            f"Could not resolve a stock ticker from: '{query}'. "
            "Please try using the explicit ticker symbol (e.g., TTWO, RELIANCE.NS)."
        )
    return result


# Quick test
if __name__ == "__main__":
    test_queries = [
        "Take Two before GTA 6",
        "Nvidia",
        "Reliance Industries",
        "TSLA",
        "I want to invest in Apple",
        "Tata Motors",
    ]
    for q in test_queries:
        result = resolve_ticker(q)
        if result:
            print(f"✅ '{q}' → {result} [confidence: {result.confidence}, context: {result.context}]")
        else:
            print(f"❌ '{q}' → No resolution found")