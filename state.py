"""
state.py -- State schema for the analysis graph and the user profile used by Investment Mode.
"""
from typing import TypedDict


class UserProfile(TypedDict, total=False):
    location: str               # country name as entered, title-cased
    base_currency: str          # "INR" or "USD"
    budget: float               # investable capital in base_currency (> 0)
    budget_type: str            # "lumpsum" or "monthly"
    goals: list[str]
    risk_tolerance: str         # "conservative", "moderate", "aggressive"
    investment_horizon: str     # e.g. "5 years"
    horizon_years: float
    preferred_markets: list[str]  # subset of ["us", "india"]
    preferred_sectors: list[str]
    num_positions: int
    existing_investments: list
    completed_onboarding: bool


def empty_profile() -> UserProfile:
    return UserProfile(
        location="",
        base_currency="",
        budget=0.0,
        budget_type="",
        goals=[],
        risk_tolerance="",
        investment_horizon="",
        horizon_years=0.0,
        preferred_markets=[],
        preferred_sectors=[],
        num_positions=0,  # 0 = not answered yet
        existing_investments=[],
        completed_onboarding=False,
    )


class TeamState(TypedDict, total=False):
    query: str
    user_profile: UserProfile
    ticker: str
    ticker_context: str
    ticker_result: dict
    data: dict
    data_gaps: list[str]
    sources: list[str]
    final_brief: str
    llm_trace: list[dict]


def initial_state(query: str = "", user_profile: UserProfile | None = None) -> TeamState:
    return TeamState(query=query, user_profile=user_profile or empty_profile())
