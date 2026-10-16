"""
evaluation/run_offline.py -- Offline evaluation (no network, no LLM). Run: ``make eval-offline``.

Suites
1. Resolver: evaluation/resolver_queries.csv through the *deterministic* resolver stages (static map + regex),
   with Yahoo lookups served from the recorded fixtures (src/financeagents/data/fixtures) and Yahoo search disabled. The audited resolver from
   commit 32abe04 is run the same way as the baseline. This does NOT measure the LLM or Yahoo-search stages.
2. Portfolio constraints: synthetic profiles through the deterministic allocator on fixture volatilities.

Writes evaluation/results/<date>_offline.json and .md. Exit code 1 if a gate fails:
constraint pass rate < 100%, or resolver accuracy < committed evaluation/baseline.json accuracy - 2 points.
"""
from __future__ import annotations

import csv
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import tempfile
import types
from collections import defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FINANCEAGENTS_SKIP_DOTENV", "1")

from financeagents import resolver as ticker_resolver
from financeagents.data import market as market_data
from financeagents.data.fixtures import FixtureTicker, load
from financeagents.portfolio import service as pa
from financeagents.portfolio.allocator import BUCKET_WEIGHTS, validate_allocation

BASELINE_COMMIT = "32abe04"
QUERIES = ROOT / "evaluation" / "resolver_queries.csv"
RESULTS = ROOT / "evaluation" / "results"
BASELINE_FILE = ROOT / "evaluation" / "baseline.json"
TOLERANCE_PTS = 2.0


class _NoSearch:
    def __init__(self, *a, **k):
        self.quotes = []


def _block_network():
    import socket

    def guard(*a, **k):
        raise RuntimeError("network disabled during offline eval")

    socket.socket.connect = guard
    socket.create_connection = guard


def load_baseline_resolver():
    """Import ticker_resolver.py as it was at BASELINE_COMMIT, with no LLM key and fixture-backed yfinance."""
    src = subprocess.run(["git", "show", f"{BASELINE_COMMIT}:ticker_resolver.py"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout
    stub_config = types.SimpleNamespace(OPENAI_API_KEY="", OPENAI_BASE_URL="", LLM_MODEL="")
    saved = sys.modules.get("config")
    sys.modules["config"] = stub_config
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
        spec = importlib.util.spec_from_file_location("baseline_resolver", f.name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        if saved is not None:
            sys.modules["config"] = saved
        else:
            sys.modules.pop("config", None)
        os.unlink(f.name)
    mod.OpenAI = lambda **k: None  # never called: the old code returns early without a key
    mod.yf = types.SimpleNamespace(Ticker=FixtureTicker, Search=_NoSearch)
    return lambda q: mod.resolve_ticker(q)


def current_resolver():
    market_data.yf = types.SimpleNamespace(Ticker=FixtureTicker)
    ticker_resolver.yf = types.SimpleNamespace(Ticker=FixtureTicker, Search=_NoSearch)
    return lambda q: ticker_resolver.resolve_ticker(q, None)


def eval_resolver(resolve) -> dict:
    rows = list(csv.DictReader(QUERIES.open(encoding="utf-8")))
    per_cat = defaultdict(lambda: [0, 0])
    errors, correct, false_pos, traps, abstain = [], 0, 0, 0, 0
    for r in rows:
        exp = r["expected"] or None
        try:
            res = resolve(r["query"])
            got = res.ticker if res else None
        except Exception as e:  # a crash counts as an error, not an abstention
            got = f"<error {type(e).__name__}>"
        ok = got == exp
        correct += ok
        per_cat[r["category"]][0] += ok
        per_cat[r["category"]][1] += 1
        if exp is None:
            traps += 1
            false_pos += got is not None
        abstain += got is None
        if not ok:
            errors.append({"query": r["query"], "expected": exp, "got": got, "category": r["category"]})
    n = len(rows)
    return {
        "n": n,
        "accuracy_pct": round(100 * correct / n, 1),
        "per_category_accuracy_pct": {c: round(100 * a / t, 1) for c, (a, t) in sorted(per_cat.items())},
        "no_answer_false_positive_pct": round(100 * false_pos / traps, 1) if traps else None,
        "abstain_pct": round(100 * abstain / n, 1),
        "errors": errors,
    }


def eval_constraints() -> dict:
    market_data.yf = types.SimpleNamespace(Ticker=FixtureTicker)
    cases = list(itertools.product(BUCKET_WEIGHTS, [["us"], ["india"], ["us", "india"], []],
                                   [3, 4, 5, 8, 12, 15], [999.99, 100_000.0]))
    failures = []
    for risk, markets, n, budget in cases:
        prof = {"budget": budget, "base_currency": "USD", "risk_tolerance": risk, "num_positions": n,
                "preferred_markets": markets}
        try:
            out = pa.suggest_portfolio(prof, None)
            validate_allocation(out["positions"], budget, markets)
            buckets = defaultdict(float)
            for p in out["positions"]:
                buckets[p["category"]] += p["allocation_pct"]
            for b, w in BUCKET_WEIGHTS[risk].items():
                if abs(buckets[b] - w) > 0.02:
                    raise ValueError(f"bucket {b} {buckets[b]:.2f} != {w}")
        except Exception as e:
            failures.append({"profile": prof, "error": str(e)})
    return {"n": len(cases), "pass_rate_pct": round(100 * (len(cases) - len(failures)) / len(cases), 2),
            "failures": failures}


def to_markdown(res: dict) -> str:
    cur, base, con = res["resolver"]["current"], res["resolver"]["baseline"], res["constraints"]
    cats = sorted(set(cur["per_category_accuracy_pct"]) | set(base["per_category_accuracy_pct"]))
    lines = [
        f"## Offline evaluation ({res['date']}, fixtures recorded {res['fixtures_recorded_at']})",
        "",
        f"Resolver: {cur['n']} labelled queries, deterministic stages only (no LLM, no Yahoo search).",
        "",
        f"| Metric | Baseline `{BASELINE_COMMIT}` | Current |",
        "|---|---|---|",
        f"| Accuracy | {base['accuracy_pct']}% | {cur['accuracy_pct']}% |",
        f"| False positives on no-answer queries | {base['no_answer_false_positive_pct']}% | "
        f"{cur['no_answer_false_positive_pct']}% |",
        f"| Abstain rate | {base['abstain_pct']}% | {cur['abstain_pct']}% |",
    ]
    lines += [f"| Accuracy: {c} | {base['per_category_accuracy_pct'].get(c, '-')}% | "
              f"{cur['per_category_accuracy_pct'].get(c, '-')}% |" for c in cats]
    lines += ["", f"Portfolio constraints: {con['n']} synthetic profiles, pass rate {con['pass_rate_pct']}%.", ""]
    return "\n".join(lines)


def main() -> int:
    _block_network()
    res = {
        "date": date.today().isoformat(),
        "fixtures_recorded_at": load("AAPL").get("recorded_at", "?")[:10],
        "resolver": {"current": eval_resolver(current_resolver()),
                     "baseline": eval_resolver(load_baseline_resolver())},
        "constraints": eval_constraints(),
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = RESULTS / f"{res['date']}_offline"
    stem.with_suffix(".json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    md = to_markdown(res)
    stem.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)

    gates = []
    if res["constraints"]["pass_rate_pct"] < 100:
        gates.append("constraint pass rate < 100%")
    if BASELINE_FILE.exists():
        committed = json.loads(BASELINE_FILE.read_text())["resolver_accuracy_pct"]
        if res["resolver"]["current"]["accuracy_pct"] < committed - TOLERANCE_PTS:
            gates.append(f"resolver accuracy below committed baseline {committed}% - {TOLERANCE_PTS}")
    for g in gates:
        print(f"GATE FAILED: {g}", file=sys.stderr)
    return 1 if gates else 0


if __name__ == "__main__":
    sys.exit(main())
