import json

import pytest

from financeagents import resolver as tr
from financeagents.errors import LLMAuthError
from financeagents.llm import Settings

LISTED = {
    "AAPL": {"shortName": "Apple Inc.", "longName": "Apple Inc.", "exchange": "NMS", "currency": "USD"},
    "TTWO": {"shortName": "Take-Two", "longName": "Take-Two Interactive Software, Inc.", "exchange": "NMS"},
    "META": {"shortName": "Meta", "longName": "Meta Platforms, Inc.", "exchange": "NMS"},
    "TATASTEEL.NS": {"shortName": "TATA STEEL", "longName": "Tata Steel Limited", "exchange": "NSI",
                     "currency": "INR"},
    "TATAMOTORS.NS": {"shortName": "TATA MOTORS", "longName": "Tata Motors Limited", "exchange": "NSI"},
    "BRK-B": {"shortName": "Berkshire Hathaway B", "longName": "Berkshire Hathaway Inc.", "exchange": "NYQ"},
    "1211.HK": {"shortName": "BYD", "longName": "BYD Company Limited", "exchange": "HKG", "currency": "HKD"},
    # Real-looking symbols that are NOT what the user meant:
    "CEO": {"shortName": "Some CEO ETF"}, "GTA": {"shortName": "GTA Corp"}, "PE": {"shortName": "PE Inc"},
    "ETF": {"shortName": "ETF Co"}, "ZZZ": {"shortName": "Zzz"},
}


def lookup(t):
    return LISTED.get(t)


def no_search(_name):
    return []


def resolve(q, **kw):
    return tr.resolve_ticker(q, None, lookup=lookup, search=kw.get("search", no_search))


@pytest.mark.parametrize("query,expected", [
    ("pineapple futures", None),
    ("Are metals a good hedge?", None),
    ("I think ETF is safer", None),
    ("Is the CEO of Apple good?", "AAPL"),
    ("Take Two before GTA 6", "TTWO"),
    ("what is the PE of BYD", "1211.HK"),
    ("Tata Steel outlook", "TATASTEEL.NS"),
])
def test_known_false_positives_fixed(query, expected):
    r = resolve(query)
    assert (r.ticker if r else None) == expected


def test_regex_candidate_is_low_confidence():
    r = resolve("thoughts on META here")
    assert r.ticker == "META" and r.confidence == "low" and r.source == "regex"


@pytest.mark.parametrize("raw,norm", [("BRK.B", "BRK-B"), ("brk-b", "BRK-B"), ("RELIANCE.NS", "RELIANCE.NS"),
                                      ("TTWO (NASDAQ)", "TTWO"), ("0700.HK", "0700.HK"), ("", "")])
def test_normalize_ticker(raw, norm):
    assert tr.normalize_ticker(raw) == norm


def test_pick_best_match_abstains_without_name_match():
    results = [{"symbol": "ZZZ", "shortname": "Zzz", "longname": "Zzz Corp", "exchange": "NMS"}]
    assert tr.pick_best_match(results, "Nvidia") == (None, [])


def test_pick_best_match_prefers_name_and_returns_candidates():
    results = [
        {"symbol": "NVD.F", "shortname": "NVIDIA", "longname": "NVIDIA Corporation", "exchange": "FRA"},
        {"symbol": "NVDA", "shortname": "NVIDIA", "longname": "NVIDIA Corporation", "exchange": "NMS"},
    ]
    best, cands = tr.pick_best_match(results, "nvidia")
    assert best["symbol"] == "NVDA" and cands == ["NVDA", "NVD.F"]


def _with_llm(patch_llm, fake_llm, content):
    patch_llm(fake_llm({"m": [content]}))
    return Settings(api_key="k", model="m")


def test_llm_explicit_share_class_ticker(patch_llm, fake_llm):
    s = _with_llm(patch_llm, fake_llm, json.dumps({"company_name": "Berkshire", "ticker_if_known": "BRK.B",
                                                   "context": "value"}))
    r = tr.resolve_ticker("Berkshire B shares?", s, lookup=lookup, search=no_search)
    assert r.ticker == "BRK-B" and r.confidence == "high"


def test_india_suffix_only_from_company_name(patch_llm, fake_llm):
    s = _with_llm(patch_llm, fake_llm, json.dumps({"company_name": "Tata Motors", "ticker_if_known": "TATAMOTORS",
                                                   "context": ""}))
    assert tr.resolve_ticker("tata motors?", s, lookup=lookup, search=no_search).ticker == "TATAMOTORS.NS"


def test_llm_garbage_falls_back_to_static_map(patch_llm, fake_llm):
    s = _with_llm(patch_llm, fake_llm, "I cannot help")
    assert tr.resolve_ticker("apple stock", s, lookup=lookup, search=no_search).ticker == "AAPL"


def test_llm_auth_error_propagates(patch_llm, fake_llm):
    from test_llm import status_error

    patch_llm(fake_llm({"m": [status_error(401)]}))
    with pytest.raises(LLMAuthError):
        tr.resolve_ticker("apple", Settings(api_key="bad", model="m"), lookup=lookup, search=no_search)


@pytest.mark.parametrize("query,expected", [
    ("HDFCBANK.NS loan growth", ["HDFCBANK.NS"]), ("NIFTYBEES.NS for passive", ["NIFTYBEES.NS"]),
    ("BRK-B valuation", ["BRK-B"]), ("BRK.B outlook", ["BRK-B"]), ("Is the CEO of Apple good?", []),
    ("LONGWORDHERE is not a ticker", []),
])
def test_regex_candidates(query, expected):
    assert tr.regex_candidates(query) == expected


def test_search_skips_unvalidated_match_and_tries_next(patch_llm, fake_llm):
    s = _with_llm(patch_llm, fake_llm, json.dumps({"company_name": "Take-Two", "ticker_if_known": "", "context": ""}))
    results = [
        {"symbol": "TTWO.DELISTED", "shortname": "Take-Two", "longname": "Take-Two Interactive Software, Inc.",
         "exchange": "NMS"},
        {"symbol": "TTWO", "shortname": "Take-Two", "longname": "Take-Two Interactive Software, Inc.",
         "exchange": "NMS"},
    ]
    r = tr.resolve_ticker("take-two?", s, lookup=lookup, search=lambda name: results)
    assert r.ticker == "TTWO" and r.source == "yahoo_search"


def test_search_with_no_valid_match_falls_through_to_static_map(patch_llm, fake_llm):
    s = _with_llm(patch_llm, fake_llm, json.dumps({"company_name": "Apple", "ticker_if_known": "", "context": ""}))
    bad = [{"symbol": "APPL.GONE", "shortname": "Apple", "longname": "Apple Gone Inc", "exchange": "NMS"}]
    r = tr.resolve_ticker("apple", s, lookup=lookup, search=lambda name: bad)
    assert r.ticker == "AAPL" and r.source == "static_map"


def test_module_level_search_can_be_swapped(monkeypatch, patch_llm, fake_llm):
    s = _with_llm(patch_llm, fake_llm, json.dumps({"company_name": "Meta", "ticker_if_known": "", "context": ""}))
    monkeypatch.setattr(tr, "lookup_info", lookup)
    monkeypatch.setattr(tr, "search_yahoo", lambda name: [
        {"symbol": "META", "shortname": "Meta", "longname": "Meta Platforms, Inc.", "exchange": "NMS"}])
    assert tr.resolve_ticker("meta?", s).ticker == "META"
