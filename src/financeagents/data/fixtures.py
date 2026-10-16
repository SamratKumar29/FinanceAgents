"""data/fixtures.py -- Offline stand-in for ``yfinance.Ticker`` backed by recorded JSON in data/fixtures/.

Used by the test suite, the offline evaluation and demo mode. Refresh the data with scripts/record_fixtures.py.
"""
import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

FIXTURES = Path(__file__).resolve().parent / "fixtures"
_PERIOD_DAYS = {"5d": 7, "1mo": 31, "6mo": 183, "1y": 366}  # calendar days before recorded_at


@lru_cache(maxsize=None)
def load(symbol: str) -> dict:
    path = FIXTURES / f"{symbol.replace('^', '_')}.json"
    if not path.exists():
        return {"symbol": symbol, "info": {}, "close": {}, "tz": None, "statements": {}}
    return json.loads(path.read_text(encoding="utf-8"))


class FixtureTicker:
    instances: list = []  # every constructed ticker, for call-count assertions

    def __init__(self, symbol: str):
        self.symbol = symbol
        self._d = load(symbol)
        self.info_reads = 0
        FixtureTicker.instances.append(self)

    @property
    def info(self) -> dict:
        self.info_reads += 1
        return dict(self._d["info"])

    def history(self, period: str = "1y") -> pd.DataFrame:
        close = self._d["close"]
        if not close:
            return pd.DataFrame({"Close": pd.Series(dtype=float)})
        idx = pd.DatetimeIndex(pd.to_datetime(list(close)))
        if self._d.get("tz"):
            idx = idx.tz_localize(self._d["tz"])
        s = pd.Series(list(close.values()), index=idx, dtype=float)
        recorded = pd.Timestamp(self._d.get("recorded_at", s.index[-1])).tz_localize(None)
        cutoff = recorded - pd.Timedelta(days=_PERIOD_DAYS.get(period, 366))
        return pd.DataFrame({"Close": s[s.index.tz_localize(None) >= cutoff]})

    def _statement(self, name: str):
        st = self._d["statements"].get(name)
        if not st:
            return pd.DataFrame()
        return pd.DataFrame({pd.Timestamp(st["period_end"]): pd.Series(st["rows"], dtype=float)})

    income_stmt = property(lambda self: self._statement("income_stmt"))
    balance_sheet = property(lambda self: self._statement("balance_sheet"))
    cashflow = property(lambda self: self._statement("cashflow"))
