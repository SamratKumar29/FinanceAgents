"""
financeagents.ui.presentation -- Turn an analysis result into display facts (resolution, as-of, sources, checks).

Kept free of Streamlit so it can be unit-tested; ui/app.py renders the returned strings.
"""
from __future__ import annotations


def describe_result(result: dict) -> dict:
    tr = result.get("ticker_result") or {}
    data = result.get("data") or {}
    md = data.get("market_data") or {}
    risk = data.get("risk_metrics") or {}
    faith = result.get("faithfulness") or {}
    trace = result.get("trace") or {}

    resolution = None
    alternatives = None
    if tr:
        resolution = (f"Resolved **{tr['ticker']}**: {tr.get('company') or tr['ticker']} ({tr.get('exchange', '?')}) "
                      f"· confidence **{tr['confidence']}** · via {tr.get('source') or 'resolver'}")
        others = [c for c in tr.get("candidates") or [] if c != tr["ticker"]]
        if tr["confidence"] != "high":
            alternatives = ("Not the company you meant? Ask again with the exact ticker"
                            + (f" (other candidates: {', '.join(others)})." if others else "."))

    as_of_parts = []
    if md.get("last_close_date"):
        as_of_parts.append(f"Prices as of last close {md['last_close_date']}")
    if md.get("currency"):
        as_of_parts.append(f"in {md['currency']}")
    if risk.get("benchmark"):
        as_of_parts.append(f"risk {risk.get('start_date')} to {risk.get('end_date')} vs {risk.get('benchmark_name')} "
                           f"({risk['benchmark']})")
    if risk.get("risk_free_rate_source"):
        as_of_parts.append(f"risk-free rate {risk['risk_free_rate'] * 100:.2f}% ({risk['risk_free_rate_source']})")
    as_of = " · ".join(as_of_parts) or None

    sources = ["Yahoo Finance: quotes, price history, financial statements, analyst estimates"]
    sources += [f"Web: {u}" for u in result.get("sources") or []]

    gaps = result.get("data_gaps") or []
    gaps_note = f"Unavailable data: {', '.join(gaps)}" if gaps else None

    n = faith.get("numbers_checked", 0)
    bad = faith.get("unsupported") or []
    if not n:
        check = ("info", "No numeric claims to check.")
    elif bad:
        examples = ", ".join(u["raw"] for u in bad[:5])
        check = ("warning", f"{len(bad)} of {n} numbers in this brief were not found in the source data "
                            f"({examples}). Treat them with caution.")
    else:
        check = ("success", f"No unsupported figures: all {n} numbers checked were found in the source data. "
                            "(The check catches invented precise figures; round numbers can match by chance.)")

    trace_line = None
    if trace:
        trace_line = (f"Run `{trace['run_id']}` · {trace['total_latency_ms']:.0f} ms · {trace['total_tokens']} tokens · "
                      f"models: {', '.join(trace.get('models_used') or []) or 'n/a'}")
    return {"resolution": resolution, "alternatives": alternatives, "as_of": as_of, "sources": sources,
            "gaps": gaps_note, "check": check, "trace": trace_line}
