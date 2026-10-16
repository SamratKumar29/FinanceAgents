"""
financeagents.data.market -- Cached, retrying access to Yahoo Finance (via yfinance).

* ``history``, ``info`` and ``statements`` results are cached in-process for ``TTL_S`` seconds, keyed by
  (kind, symbol, period). The S&P 500 / NIFTY benchmark is therefore fetched once per TTL, not per query.
* Each fetch is retried up to ``ATTEMPTS`` times with exponential backoff and jitter. Only exceptions are
  retried; an empty result (e.g. a delisted symbol) is a valid answer and is cached.
"""
from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Callable

import yfinance as yf

log = logging.getLogger(__name__)

TTL_S = float(os.getenv("MARKET_DATA_TTL_S", "900"))
ATTEMPTS = 3
BACKOFF_S = 0.5
sleep: Callable[[float], None] = time.sleep  # replaced in tests

_cache: dict[tuple, tuple[float, Any]] = {}
_lock = threading.Lock()
stats = {"hits": 0, "misses": 0, "retries": 0}


def clear_cache() -> None:
    with _lock:
        _cache.clear()
        stats.update(hits=0, misses=0, retries=0)


def with_retry(fn: Callable[[], Any], what: str, attempts: int = ATTEMPTS) -> Any:
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # yfinance raises many HTTP/parse error types
            if i == attempts - 1:
                raise
            stats["retries"] += 1
            delay = BACKOFF_S * (2 ** i) * (1 + random.random() * 0.5)
            log.info("retrying %s after %s (%.2fs)", what, type(e).__name__, delay)
            sleep(delay)
    raise AssertionError("unreachable")


def _cached(key: tuple, fetch: Callable[[], Any]) -> Any:
    now = time.monotonic()
    with _lock:
        hit = _cache.get(key)
        if hit and now - hit[0] < TTL_S:
            stats["hits"] += 1
            return hit[1]
    stats["misses"] += 1
    value = with_retry(fetch, f"{key[0]} {key[1]}")
    with _lock:
        _cache[key] = (time.monotonic(), value)
    return value


def history(symbol: str, period: str = "1y"):
    return _cached(("history", symbol, period), lambda: yf.Ticker(symbol).history(period=period))


def info(symbol: str) -> dict:
    return _cached(("info", symbol, ""), lambda: yf.Ticker(symbol).info or {})


def statements(symbol: str) -> tuple:
    def fetch():
        t = yf.Ticker(symbol)
        return t.income_stmt, t.balance_sheet, t.cashflow

    return _cached(("statements", symbol, ""), fetch)
