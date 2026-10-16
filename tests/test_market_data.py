import pytest
from fixture_yf import FixtureTicker

from financeagents.data import market as market_data
from financeagents.data import tools
from financeagents.errors import DataUnavailableError


class Counting(FixtureTicker):
    history_calls: list = []

    def history(self, period="1y"):
        Counting.history_calls.append((self.symbol, period))
        return super().history(period)


@pytest.fixture
def counting(monkeypatch):
    Counting.history_calls = []
    monkeypatch.setattr(market_data.yf, "Ticker", Counting)


def test_second_query_within_ttl_downloads_benchmark_zero_times(counting):
    tools.compute_risk("AAPL")
    tools.compute_risk("NVDA")
    assert Counting.history_calls.count(("^GSPC", "1y")) == 1
    assert market_data.stats["hits"] >= 1


def test_ttl_expiry_refetches(counting, monkeypatch):
    tools.compute_risk("AAPL")
    monkeypatch.setattr(market_data, "TTL_S", 0)
    tools.compute_risk("AAPL")
    assert Counting.history_calls.count(("AAPL", "1y")) == 2


def test_flaky_source_recovers_with_retries(monkeypatch):
    failures = {"n": 2}

    class Flaky(FixtureTicker):
        def history(self, period="1y"):
            if failures["n"]:
                failures["n"] -= 1
                raise ConnectionError("reset by peer")
            return super().history(period)

    monkeypatch.setattr(market_data.yf, "Ticker", Flaky)
    assert tools.compute_risk("AAPL")["observations"] > 200
    assert market_data.stats["retries"] == 2


def test_persistent_failure_is_typed_and_bounded(monkeypatch):
    calls = []

    class Down(FixtureTicker):
        @property
        def info(self):
            calls.append(1)
            raise TimeoutError("yahoo down")

    monkeypatch.setattr(market_data.yf, "Ticker", Down)
    with pytest.raises(DataUnavailableError):
        tools.fetch_info("AAPL")
    assert len(calls) == market_data.ATTEMPTS


def test_backoff_grows(monkeypatch):
    delays = []
    monkeypatch.setattr(market_data, "sleep", delays.append)
    with pytest.raises(ValueError):
        market_data.with_retry(lambda: (_ for _ in ()).throw(ValueError("x")), "boom", attempts=3)
    assert len(delays) == 2 and delays[1] > delays[0]
