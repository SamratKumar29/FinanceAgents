"""
scripts/record_fixtures.py -- Record Yahoo Finance payloads used by the offline tests.

    python scripts/record_fixtures.py            # writes src/financeagents/data/fixtures/*.json
    python scripts/record_fixtures.py CEO GTA    # record specific symbols only

Only the fields the app reads are stored. Re-run to refresh; commit the result together with the
date shown in each file's "recorded_at".
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yfinance as yf

BASE_TICKERS = ["AAPL", "TTWO", "NVDA", "RELIANCE.NS", "INFY.NS", "^GSPC", "^NSEI", "ZZZZINVALID"]
UNIVERSE = json.loads((Path(__file__).resolve().parents[1] / "src" / "financeagents" / "portfolio" / "universe.json").read_text())
TICKERS = list(dict.fromkeys(BASE_TICKERS + [i["ticker"] for i in UNIVERSE["instruments"]]))
INFO_KEYS = [
    "shortName", "longName", "exchange", "currency", "financialCurrency", "sector", "industry",
    "currentPrice", "regularMarketPrice", "marketCap", "trailingPE", "forwardPE", "fiftyTwoWeekLow",
    "fiftyTwoWeekHigh", "targetMeanPrice", "targetHighPrice", "targetLowPrice", "recommendationKey",
    "numberOfAnalystOpinions", "earningsQuarterlyGrowth", "revenueGrowth", "pegRatio", "trailingPegRatio",
    "priceToBook", "priceToSalesTrailing12Months", "dividendYield", "longBusinessSummary",
]
STATEMENT_ROWS = {
    "income_stmt": ["Total Revenue", "Net Income", "Operating Income", "Gross Profit", "Diluted EPS"],
    "balance_sheet": ["Total Assets", "Total Debt", "Stockholders Equity", "Cash And Cash Equivalents"],
    "cashflow": ["Operating Cash Flow", "Free Cash Flow", "Capital Expenditure"],
}
OUT = Path(__file__).resolve().parents[1] / "src" / "financeagents" / "data" / "fixtures"


def _clean(v):
    try:
        if v != v:  # NaN
            return None
    except Exception:
        pass
    return v.item() if hasattr(v, "item") else v


def record(symbol: str) -> dict:
    t = yf.Ticker(symbol)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    hist = t.history(period="1y")
    data = {
        "symbol": symbol,
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "info": {k: _clean(info[k]) for k in INFO_KEYS if k in info},
        "tz": str(hist.index.tz) if len(hist) else None,
        "close": {ts.strftime("%Y-%m-%dT%H:%M:%S"): _clean(v) for ts, v in hist["Close"].items()} if len(hist) else {},
        "statements": {},
    }
    if not symbol.startswith("^") and len(hist):
        for attr, rows in STATEMENT_ROWS.items():
            frame = getattr(t, attr)
            if frame is not None and not frame.empty:
                col = frame.columns[0]
                data["statements"][attr] = {
                    "period_end": col.date().isoformat(),
                    "rows": {r: _clean(frame[col].get(r)) for r in rows if r in frame.index},
                }
    return data


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for s in sys.argv[1:] or TICKERS:
        d = record(s)
        path = OUT / f"{s.replace('^', '_')}.json"
        path.write_text(json.dumps(d, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{s:14s} closes={len(d['close']):4d} info_keys={len(d['info']):3d} -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
