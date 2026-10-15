import builtins
import json

import pytest

from onboarding import (
    FIELDS,
    apply_answer,
    is_complete,
    next_field,
    parse_amount,
    parse_horizon_years,
    parse_location,
)
from session import get_profile_summary, load_session_from_str, save_session
from state import empty_profile


@pytest.mark.parametrize("text,expected", [
    ("India", "INR"), ("united states", "USD"), ("USA", "USD"),
    ("United Kingdom", None), ("Singapore", None), ("Finland", None), ("Argentina", None),
])
def test_location_currency_uses_whole_names(text, expected):
    assert parse_location(text)[1] == expected


@pytest.mark.parametrize("text,value", [
    ("10k", 10_000), ("5 lakh", 500_000), ("1.5 million", 1_500_000), ("2,50,000", 250_000),
    ("1 cr", 10_000_000), ("₹ 75000", 75_000), ("USD 1200", 1200), ("3mn", 3_000_000),
])
def test_parse_amount_units(text, value):
    assert parse_amount(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["-5000", "0", "lots", "5 zillion", ""])
def test_parse_amount_rejects(text):
    with pytest.raises(ValueError):
        parse_amount(text)


def test_horizon_parsing():
    assert parse_horizon_years("18 months") == 1.5
    assert parse_horizon_years("5 years") == 5
    with pytest.raises(ValueError):
        parse_horizon_years("idk")


@pytest.mark.parametrize("field,text", [
    ("risk_tolerance", "agressive"), ("budget_type", "sometimes"), ("num_positions", "100"),
    ("num_positions", "five"), ("base_currency", "GBP"), ("preferred_markets", "mars"), ("budget", "-5000"),
])
def test_invalid_answers_keep_profile_and_return_error(field, text):
    p = empty_profile()
    new, err = apply_answer(p, field, text)
    assert err and new is p


def test_full_flow_asks_num_positions_and_skips_currency_for_india():
    p, answered, asked = empty_profile(), set(), []
    answers = {"location": "India", "budget": "5 lakh", "budget_type": "lumpsum", "goals": "retirement, growth",
               "risk_tolerance": "moderate", "investment_horizon": "10 years", "num_positions": "6",
               "preferred_markets": "both", "existing_investments": "none"}
    while (f := next_field(p, answered)) is not None:
        asked.append(f)
        p, err = apply_answer(p, f, answers[f])
        assert err is None, (f, err)
        answered.add(f)
    assert "num_positions" in asked and "base_currency" not in asked
    assert p["base_currency"] == "INR" and p["budget"] == 500_000 and p["num_positions"] == 6
    assert p["preferred_markets"] == ["us", "india"] and is_complete(p)


def test_uk_user_is_asked_for_supported_currency():
    p, _ = apply_answer(empty_profile(), "location", "United Kingdom")
    assert next_field(p, {"location"}) == "base_currency"
    assert set(FIELDS) >= {"base_currency"}


# --- session loading (F-11, SEC-02) ---------------------------------------------------------

def _complete_profile():
    p = empty_profile()
    p.update(location="India", base_currency="INR", budget=1000.0, budget_type="lumpsum",
             risk_tolerance="moderate", investment_horizon="5 years", horizon_years=5.0, num_positions=5,
             completed_onboarding=True)
    return p


def test_roundtrip():
    p = _complete_profile()
    loaded = load_session_from_str(save_session(p))["user_profile"]
    assert loaded == p and "Budget: 1,000.00 INR" in get_profile_summary(loaded)


@pytest.mark.parametrize("payload", [
    "[1,2]", "not json", '{"user_profile": [1]}', '{"user_profile": {"budget": "lots"}}',
    '{"user_profile": {"risk_tolerance": "yolo"}}', "{}",
])
def test_bad_uploads_raise_value_error_without_touching_filesystem(payload, monkeypatch):
    def no_open(*a, **k):
        raise AssertionError("open() must not be called for uploaded content")

    monkeypatch.setattr(builtins, "open", no_open)
    with pytest.raises(ValueError):
        load_session_from_str(payload)


def test_oversized_upload_rejected():
    with pytest.raises(ValueError, match="too large"):
        load_session_from_str(json.dumps({"user_profile": {}, "pad": "x" * 70_000}))
