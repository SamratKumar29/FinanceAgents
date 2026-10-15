"""
portfolio/allocator.py -- Deterministic, rule-based model allocation (no LLM).

1. Filter the curated universe (portfolio/universe.json) by the user's markets.
2. Split ``num_positions`` across risk buckets in proportion to BUCKET_WEIGHTS[risk].
3. Pick instruments in curated order (alternating US/India when both are allowed),
   skipping any whose price history is unavailable.
4. Within a bucket, weight by inverse 1-year volatility; bucket totals equal the configured weights.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

UNIVERSE_PATH = Path(__file__).with_name("universe.json")

BUCKET_WEIGHTS: dict[str, dict[str, float]] = {
    "conservative": {"etf": 60.0, "blue_chip": 40.0},
    "moderate": {"etf": 40.0, "growth": 40.0, "speculative": 20.0},
    "aggressive": {"etf": 20.0, "growth": 50.0, "speculative": 30.0},
}
BUCKET_RISK = {"etf": "low", "blue_chip": "low", "growth": "medium", "speculative": "high"}
ALL_MARKETS = ("us", "india")


@dataclass(frozen=True)
class Instrument:
    ticker: str
    name: str
    market: str
    bucket: str


def load_universe(path: Path = UNIVERSE_PATH) -> list[Instrument]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [Instrument(i["ticker"], i["name"], i["market"], i["bucket"]) for i in data["instruments"]]


def _ordered_candidates(universe: list[Instrument], bucket: str, markets: tuple[str, ...]) -> list[Instrument]:
    per_market = [[i for i in universe if i.bucket == bucket and i.market == m] for m in markets]
    out = []
    for k in range(max((len(x) for x in per_market), default=0)):
        out.extend(x[k] for x in per_market if k < len(x))
    return out


def allocate_slots(n: int, weights: dict[str, float], capacity: dict[str, int]) -> dict[str, int]:
    """Largest-remainder split of n positions, at least 1 per bucket, never above capacity."""
    buckets = [b for b in weights if capacity.get(b, 0) > 0]
    if not buckets:
        return {}
    total = sum(weights[b] for b in buckets)
    raw = {b: n * weights[b] / total for b in buckets}
    slots = {b: max(1, int(raw[b])) for b in buckets}
    order = sorted(buckets, key=lambda b: (raw[b] - int(raw[b]), weights[b]), reverse=True)
    while sum(slots.values()) < n and any(slots[b] < capacity[b] for b in buckets):
        for b in order:
            if sum(slots.values()) >= n:
                break
            if slots[b] < capacity[b]:
                slots[b] += 1
    while sum(slots.values()) > n:  # only possible when n < number of buckets
        b = min((b for b in buckets if slots[b] > 0), key=lambda b: weights[b])
        slots[b] -= 1
    return {b: min(s, capacity[b]) for b, s in slots.items() if s > 0}


def build_allocation(
    risk_tolerance: str,
    num_positions: int,
    markets: Optional[list[str]],
    volatility: Callable[[str], Optional[float]],
    universe: Optional[list[Instrument]] = None,
) -> list[dict]:
    """Return positions ``{ticker, company, market, category, risk_level, allocation_pct, volatility_pct}``."""
    if risk_tolerance not in BUCKET_WEIGHTS:
        raise ValueError(f"unknown risk tolerance {risk_tolerance!r}")
    if num_positions < 1:
        raise ValueError("num_positions must be >= 1")
    universe = load_universe() if universe is None else universe
    mk = tuple(m for m in ALL_MARKETS if m in (markets or ALL_MARKETS)) or ALL_MARKETS
    weights = BUCKET_WEIGHTS[risk_tolerance]

    # Only instruments with usable volatility are eligible.
    vols: dict[str, float] = {}
    eligible: dict[str, list[Instrument]] = {}
    for b in weights:
        eligible[b] = []
        for inst in _ordered_candidates(universe, b, mk):
            v = volatility(inst.ticker)
            if v is not None and v > 0:
                vols[inst.ticker] = v
                eligible[b].append(inst)

    slots = allocate_slots(num_positions, weights, {b: len(v) for b, v in eligible.items()})
    if not slots:
        raise ValueError("no eligible instruments for the selected markets")
    total_w = sum(weights[b] for b in slots)  # renormalise if a bucket had no eligible instruments

    positions = []
    for b, k in slots.items():
        chosen = eligible[b][:k]
        inv = {i.ticker: 1 / vols[i.ticker] for i in chosen}
        bucket_pct = weights[b] * 100 / total_w
        for inst in chosen:
            positions.append({
                "ticker": inst.ticker,
                "company": inst.name,
                "market": inst.market,
                "category": b,
                "risk_level": BUCKET_RISK[b],
                "allocation_pct": bucket_pct * inv[inst.ticker] / sum(inv.values()),
                "volatility_pct": round(vols[inst.ticker] * 100, 2),
            })
    return positions


def validate_allocation(positions: list[dict], budget: float, markets: Optional[list[str]],
                        universe: Optional[list[Instrument]] = None) -> None:
    """Raise ValueError if any portfolio constraint is violated."""
    universe = load_universe() if universe is None else universe
    known = {i.ticker: i for i in universe}
    allowed = set(markets or ALL_MARKETS)
    if not positions:
        raise ValueError("empty allocation")
    if len({p["ticker"] for p in positions}) != len(positions):
        raise ValueError("duplicate tickers")
    for p in positions:
        if p["ticker"] not in known:
            raise ValueError(f"{p['ticker']} not in universe")
        if known[p["ticker"]].market not in allowed:
            raise ValueError(f"{p['ticker']} violates market preference")
        if p["allocation_pct"] < 0 or p.get("amount", 0) < 0:
            raise ValueError("negative weight")
    if abs(sum(p["allocation_pct"] for p in positions) - 100) > 0.01:
        raise ValueError("weights do not sum to 100")
    if "amount" in positions[0] and abs(sum(p["amount"] for p in positions) - budget) > 0.01:
        raise ValueError("amounts do not sum to budget")
