"""
onboarding.py -- Investment Mode questionnaire shared by the CLI and the Streamlit UI.

``apply_answer(profile, field, text)`` validates one answer and returns ``(new_profile, error)``;
``next_field(profile, answered)`` returns the next question to ask, or None when complete.
"""
from __future__ import annotations

import re
from typing import Optional

from financeagents.state import UserProfile

SUPPORTED_CURRENCIES = ("INR", "USD")

QUESTIONS: list[tuple[str, str]] = [
    ("location", "Where are you based? (e.g. India, US, UK)"),
    ("base_currency", "Which currency should the portfolio use? (INR or USD)"),
    ("budget", "What's your investable budget? (e.g. 500000, 5 lakh, 10k, 1.5 million)"),
    ("budget_type", "Is this a one-time lumpsum or a monthly investment? (lumpsum / monthly)"),
    ("goals", "What are your goals? (e.g. retirement, wealth growth, passive income; comma-separated)"),
    ("risk_tolerance", "How would you describe your risk tolerance? (conservative / moderate / aggressive)"),
    ("investment_horizon", "What's your investment horizon? (e.g. 1 year, 5 years, 18 months)"),
    ("num_positions", "How many stocks/ETFs would you like? (3-15)"),
    ("preferred_markets", "Which markets? (US, India, or both)"),
    ("existing_investments", "Any existing investments? (ticker + value, or 'none')"),
]
FIELDS = [f for f, _ in QUESTIONS]

# Country -> currency for supported currencies; other known countries are told to choose.
COUNTRY_CURRENCY = {
    "india": "INR", "bharat": "INR", "in": "INR", "ind": "INR",
    "united states": "USD", "united states of america": "USD", "usa": "USD", "us": "USD",
    "u.s.": "USD", "u.s.a.": "USD", "america": "USD",
}

AMOUNT_MULTIPLIERS = {
    "k": 1e3, "thousand": 1e3,
    "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "l": 1e5,
    "crore": 1e7, "crores": 1e7, "cr": 1e7,
    "m": 1e6, "mn": 1e6, "million": 1e6, "millions": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
}

RISK_LEVELS = ("conservative", "moderate", "aggressive")
RISK_SYNONYMS = {"low": "conservative", "medium": "moderate", "balanced": "moderate", "high": "aggressive"}
BUDGET_TYPES = {"lumpsum": "lumpsum", "lump sum": "lumpsum", "one-time": "lumpsum", "one time": "lumpsum",
                "once": "lumpsum", "monthly": "monthly", "sip": "monthly", "per month": "monthly"}
MARKETS = {"us": ["us"], "usa": ["us"], "united states": ["us"], "america": ["us"],
           "india": ["india"], "in": ["india"], "both": ["us", "india"], "all": ["us", "india"]}
MIN_POSITIONS, MAX_POSITIONS = 3, 15


def parse_location(text: str) -> tuple[str, Optional[str]]:
    """Return (display name, currency or None). Matches whole names, never substrings."""
    name = " ".join(text.strip().split())
    return name.title(), COUNTRY_CURRENCY.get(name.lower().strip(" ."))


def parse_currency(text: str) -> Optional[str]:
    t = text.strip().upper()
    if t in ("INR", "RUPEE", "RUPEES", "₹"):
        return "INR"
    if t in ("USD", "DOLLAR", "DOLLARS", "$", "US DOLLAR"):
        return "USD"
    return None


def parse_amount(text: str) -> float:
    """Parse '10k', '5 lakh', '1.5 million', '2,50,000', '1 cr'. Raises ValueError if invalid or <= 0."""
    t = text.strip().lower().replace(",", "").replace("₹", "").replace("$", "")
    t = re.sub(r"^(inr|usd|rs\.?)\s*|\s*(inr|usd|rupees|dollars)$", "", t).strip()
    m = re.fullmatch(r"(-?\d+(?:\.\d+)?)\s*([a-z]+)?", t)
    if not m:
        raise ValueError("Please enter an amount like 500000, 5 lakh, 10k or 1.5 million.")
    value = float(m.group(1))
    unit = m.group(2)
    if unit:
        if unit not in AMOUNT_MULTIPLIERS:
            raise ValueError(f"Unknown unit '{unit}'. Use k, lakh, crore, million or billion.")
        value *= AMOUNT_MULTIPLIERS[unit]
    if value <= 0:
        raise ValueError("The budget must be greater than zero.")
    return value


def parse_horizon_years(text: str) -> float:
    t = text.strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(years?|yrs?|y|months?|mos?|m)?", t)
    if not m:
        raise ValueError("Please enter a horizon like '5 years' or '18 months'.")
    n = float(m.group(1))
    unit = m.group(2) or "years"
    years = n / 12 if unit.startswith("m") else n
    if years < 1 / 12 or years > 60:
        raise ValueError("The horizon must be between 1 month and 60 years.")
    return years


def parse_choice(text: str, options: dict[str, str] | tuple[str, ...], label: str) -> str:
    t = " ".join(text.strip().lower().split())
    if isinstance(options, tuple):
        if t in options:
            return t
        raise ValueError(f"Please choose one of: {', '.join(options)}.")
    if t in options:
        return options[t]
    raise ValueError(f"Please choose a valid {label}.")


def apply_answer(profile: UserProfile, field: str, text: str) -> tuple[UserProfile, Optional[str]]:
    """Validate ``text`` for ``field``. Returns a new profile and None, or the old profile and an error."""
    new = UserProfile(**profile)
    answer = (text or "").strip()
    if not answer:
        return profile, "Please enter an answer."
    try:
        if field == "location":
            new["location"], currency = parse_location(answer)
            if currency:
                new["base_currency"] = currency
        elif field == "base_currency":
            cur = parse_currency(answer)
            if cur is None:
                raise ValueError("Only INR and USD portfolios are supported; please enter INR or USD.")
            new["base_currency"] = cur
        elif field == "budget":
            new["budget"] = parse_amount(answer)
        elif field == "budget_type":
            new["budget_type"] = parse_choice(answer, BUDGET_TYPES, "budget type (lumpsum or monthly)")
        elif field == "goals":
            goals = [g.strip().lower() for g in answer.split(",") if g.strip()]
            new["goals"] = goals
        elif field == "risk_tolerance":
            t = answer.lower()
            new["risk_tolerance"] = parse_choice(RISK_SYNONYMS.get(t, t), RISK_LEVELS, "risk tolerance")
        elif field == "investment_horizon":
            new["horizon_years"] = parse_horizon_years(answer)
            new["investment_horizon"] = answer
        elif field == "num_positions":
            if not re.fullmatch(r"\d+", answer):
                raise ValueError("Please enter a whole number, e.g. 5.")
            n = int(answer)
            if not MIN_POSITIONS <= n <= MAX_POSITIONS:
                raise ValueError(f"Please choose between {MIN_POSITIONS} and {MAX_POSITIONS} positions.")
            new["num_positions"] = n
        elif field == "preferred_markets":
            new["preferred_markets"] = list(
                parse_choice(answer, {k: ",".join(v) for k, v in MARKETS.items()}, "market (US, India or both)")
                .split(",")
            )
        elif field == "existing_investments":
            none_words = ("none", "no", "nil", "n/a", "na", "nothing")
            new["existing_investments"] = [] if answer.lower() in none_words else [{"description": answer[:500]}]
        else:
            raise ValueError(f"Unknown field {field!r}")
    except ValueError as e:
        return profile, str(e)
    return new, None


def next_field(profile: UserProfile, answered: set[str]) -> Optional[str]:
    """Next unanswered question. Currency is skipped when the location already determined it."""
    for f in FIELDS:
        if f in answered:
            continue
        if f == "base_currency" and profile.get("base_currency") in SUPPORTED_CURRENCIES:
            continue
        return f
    return None


def is_complete(profile: UserProfile) -> bool:
    return bool(
        profile.get("base_currency") in SUPPORTED_CURRENCIES
        and (profile.get("budget") or 0) > 0
        and profile.get("budget_type") in ("lumpsum", "monthly")
        and profile.get("risk_tolerance") in RISK_LEVELS
        and (profile.get("horizon_years") or 0) > 0
        and MIN_POSITIONS <= (profile.get("num_positions") or 0) <= MAX_POSITIONS
    )


def question_text(field: str) -> str:
    return dict(QUESTIONS)[field]
