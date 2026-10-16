"""
analytics/faithfulness.py -- Check that the numbers in a brief are supported by the tool data it was given.

``check_brief(brief, payload)`` pulls numeric claims out of the brief and tries to match each one to a
number in the payload JSON (or the user query). Matching allows for rounding to the decimals shown (plus 0.1% slack), fractions
written as percentages, unit words (trillion/billion/million/K, crore/lakh), sign changes ("fell 13.8%")
and a few derived values (percentage gap between two numbers, e.g. upside to the analyst target).

Measured leniency (AAPL / RELIANCE.NS fixture payloads): random values in 1-100 are accepted by
coincidence 5-9% of the time with 2 decimals, 12-15% with 1 decimal and 50-61% as integers. It reliably flags invented precise figures, but
cannot prove that every number, especially round ones, is correct. Currency symbols are not checked
(a market cap in INR written with '$' passes if the number matches).
"""
from __future__ import annotations

import bisect
import itertools
import json
import re
from typing import Any, Iterable

_SCALE = {
    "trillion": 1e12, "trn": 1e12, "tn": 1e12, "t": 1e12,
    "billion": 1e9, "bn": 1e9, "b": 1e9,
    "million": 1e6, "mn": 1e6, "m": 1e6,
    "thousand": 1e3, "k": 1e3,
    "crore": 1e7, "cr": 1e7, "lakh": 1e5,
}
_NUM = re.compile(
    r"(?<![\w.])(?P<cur>[$₹€£]|rs\.?\s*|inr\s*|usd\s*)?"
    r"(?P<num>-?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
    r"\s*(?P<pct>%|percent\b)?"
    r"(?:\s*(?P<scale>trillion|billion|million|thousand|crore|lakh|trn|tn|bn|mn|cr|[tbmk])\b)?",
    re.IGNORECASE,
)
# Time spans and labels are not claims about the data ("6-month", "52-week", "Q3", "1 year").
_DASHES = "\u2010\u2011\u2012\u2013\u2014-"  # hyphen last: literal in a character class
_TIME_AFTER = re.compile(rf"^[\s\u202f{_DASHES}]*(?:day|week|month|year|yr|quarter)s?\b", re.IGNORECASE)
_BOUND_BEFORE = re.compile(r"(?:[<>≤≥]|\b(?:over|above|below|under|exceeds?|at least|at most))[\s\u202f]*$", re.I)
_APPROX_BEFORE = re.compile(r"(?:[≈~]|\b(?:about|around|roughly|approximately|nearly|almost))[\s\u202f+\-]*$", re.I)
APPROX_SLACK = 0.05
_GREATER = (">", "≥", "over", "above", "exceed", "at least")


def _bound_direction(text: str) -> str:
    t = text.strip().lower()
    return "gt" if t.startswith(_GREATER) else "lt"
RELATIVE_SLACK = 0.001


_INDEX_NAMES = re.compile(r"\b(?:S&P|NIFTY|Nifty|FTSE|CAC|DAX|Russell|Sensex|KOSPI)[\s\u202f]*\d+", re.IGNORECASE)


def _strip_disclaimer(text: str) -> str:
    text = re.split(r"this is general informational analysis", text, flags=re.IGNORECASE)[0]
    return _INDEX_NAMES.sub("INDEX", text)


def extract_claims(text: str) -> list[dict]:
    claims = []
    text = _strip_disclaimer(text)
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line)  # list markers
        for m in _NUM.finditer(line):
            raw = m.group(0).strip()
            num = float(m.group("num").replace(",", ""))
            after = line[m.end():]
            before = line[max(0, m.start() - 2):m.start()]
            if _TIME_AFTER.match(after) and not m.group("pct") and not m.group("cur"):
                continue
            if before.endswith(("Q", "q")) or (re.fullmatch(r"(19|20)\d\d", m.group("num")) and not m.group("cur")):
                continue  # quarters and years
            scale = (m.group("scale") or "").lower()
            if scale and not m.group("cur") and len(scale) == 1 and scale not in ("b", "m", "k", "t"):
                scale = ""
            prefix = line[max(0, m.start() - 16):m.start()]
            bound = _BOUND_BEFORE.search(prefix)
            digits = m.group("num").split(".")
            decimals = len(digits[1]) if len(digits) > 1 else 0
            claims.append({"raw": raw, "number": abs(num), "scale": _SCALE.get(scale, 1.0), "decimals": decimals,
                           "value": num * _SCALE.get(scale, 1.0), "is_pct": bool(m.group("pct")),
                           "bound": _bound_direction(bound.group(0)) if bound else None,
                           "approx": bool(_APPROX_BEFORE.search(prefix)),
                           "context": line.strip()[:160]})
    return claims


def _numbers(obj: Any) -> Iterable[float]:
    if isinstance(obj, bool) or obj is None:
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _numbers(v)
    elif isinstance(obj, str):
        for m in re.finditer(r"-?\d+(?:\.\d+)?", obj.replace(",", "")):
            yield float(m.group(0))


_PRICE_KEY = re.compile(r"price|target|week52|close", re.IGNORECASE)


def _price_fields(obj: Any) -> Iterable[float]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and _PRICE_KEY.search(str(k)):
                yield float(v)
            else:
                yield from _price_fields(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _price_fields(v)


def _sections(obj: Any) -> Iterable[list[float]]:
    """Numeric values of each financial-statement section (dicts carrying a ``period_end``)."""
    if isinstance(obj, dict):
        if "period_end" in obj:
            yield [float(v) for v in obj.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
        for v in obj.values():
            yield from _sections(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _sections(v)


def reference_values(payload: dict, extra_text: str = "") -> list[float]:
    base = {v for v in _numbers(payload)} | set(_numbers(extra_text))
    refs = set()
    for v in base:
        if v == v:
            refs.add(abs(v))
            if abs(v) < 1.5:  # fractions such as 0.12 growth may be written as 12%
                refs.add(abs(v) * 100)
    # Derived: ratios within one financial statement (e.g. debt / assets, cash / debt).
    for section in _sections(payload):
        vals = [abs(v) for v in section if v]
        for a, b in itertools.permutations(vals, 2):
            refs.update({a / b, a / b * 100})
    # Derived: percentage gaps between price-like fields only (upside to target, distance from 52w high).
    prices = sorted({v for v in _price_fields(payload) if v > 0})
    for a, b in itertools.permutations(prices, 2):
        refs.add(abs((a / b - 1) * 100))
    return sorted(refs)


def _supported(claim: dict, refs: list[float]) -> bool:
    """Supported if some reference, shown in the claim's units, rounds to the claimed number.

    Allowed error is half a unit in the last shown decimal place, plus 0.1% slack ("about").
    """
    n, scale = claim["number"], claim["scale"]
    if claim.get("bound"):  # "> 17x" is supported by a reference on the stated side of the bound
        lo = bisect.bisect_left(refs, n * scale)
        if claim["bound"] == "gt":
            return lo < len(refs) and refs[lo] <= 3 * n * scale
        return lo > 0 and refs[lo - 1] >= n * scale / 3
    slack = APPROX_SLACK if claim.get("approx") else RELATIVE_SLACK
    tol = 0.5 * 10 ** -claim["decimals"] + slack * n
    lo = bisect.bisect_left(refs, (n - tol) * scale)
    return lo < len(refs) and refs[lo] <= (n + tol) * scale


def check_brief(brief: str, payload: dict, query: str = "") -> dict:
    claims = extract_claims(brief)
    refs = reference_values(payload, query)
    unsupported = [c for c in claims if not _supported(c, refs)]
    n = len(claims)
    return {
        "numbers_checked": n,
        "unsupported": [{"raw": c["raw"], "context": c["context"]} for c in unsupported],
        "unsupported_rate": round(len(unsupported) / n, 3) if n else 0.0,
    }


if __name__ == "__main__":  # python -m analytics.faithfulness brief.txt payload.json
    import sys

    print(json.dumps(check_brief(open(sys.argv[1]).read(), json.load(open(sys.argv[2]))), indent=1))
