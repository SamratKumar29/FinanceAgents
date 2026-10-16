"""
evaluation/bench_latency.py -- Synthetic latency benchmark for the analysis data path (REL-01).

Every Yahoo call is served from fixtures after a simulated delay (default 200 ms); the LLM is a fake that
answers instantly. So the numbers show the effect of concurrency and caching on data fetching only. They
are not real-world latencies.

    python -m evaluation.bench_latency [delay_ms]
"""
from __future__ import annotations

import os
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("FINANCEAGENTS_SKIP_DOTENV", "1")

from financeagents import pipeline as graph
from financeagents.data import market as market_data
from financeagents.data.fixtures import FixtureTicker
from financeagents.llm import Settings
from financeagents.resolver import TickerResult
from financeagents.state import initial_state

DELAY_S = (float(sys.argv[1]) if len(sys.argv) > 1 else 200) / 1000


class InstantLLM:
    """Answers every chat completion immediately with a fixed brief."""

    def __init__(self):
        reply = types.SimpleNamespace(content="BRIEF")
        resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=reply, finish_reason="stop")],
                                     model="fake", usage=None)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=lambda **kw: resp))

    def factory(self, _settings):
        return self


class SlowTicker(FixtureTicker):
    calls = 0

    def history(self, period="1y"):
        SlowTicker.calls += 1
        time.sleep(DELAY_S)
        return super().history(period)

    @property
    def info(self):
        SlowTicker.calls += 1
        time.sleep(DELAY_S)
        return dict(self._d["info"])

    def _statement(self, name):
        time.sleep(DELAY_S / 3)
        return super()._statement(name)


def run_once(workers: int) -> tuple[float, int]:
    client = InstantLLM()
    original = graph.complete
    graph.complete = lambda s, m, **kw: original(s, m, client_factory=client.factory, **kw)
    real_pool = graph.ThreadPoolExecutor
    graph.ThreadPoolExecutor = lambda max_workers: real_pool(max_workers=workers)
    try:
        g = graph.build_graph(Settings(api_key="k", model="m"),
                              resolver=lambda q, s: TickerResult("AAPL", "Apple", "NMS", "high", "", "USD"))
        SlowTicker.calls = 0
        t0 = time.perf_counter()
        g.invoke(initial_state("Analyze AAPL"))
        return (time.perf_counter() - t0) * 1000, SlowTicker.calls
    finally:
        graph.complete = original
        graph.ThreadPoolExecutor = real_pool


def main() -> int:
    market_data.yf.Ticker = SlowTicker
    rows = []
    market_data.clear_cache()
    rows.append(("sequential, cold cache", *run_once(workers=1)))
    market_data.clear_cache()
    rows.append(("concurrent, cold cache", *run_once(workers=5)))
    rows.append(("concurrent, warm cache (repeat query)", *run_once(workers=5)))
    print(f"Synthetic benchmark: {DELAY_S * 1000:.0f} ms per Yahoo call, fake LLM.\n")
    print("| Configuration | Wall time (ms) | Yahoo calls |\n|---|---|---|")
    for name, ms, calls in rows:
        print(f"| {name} | {ms:.0f} | {calls} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
