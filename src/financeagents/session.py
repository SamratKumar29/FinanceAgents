"""
session.py -- Save and load user profiles as JSON.

Uploaded content is only ever parsed as JSON (``load_session_from_str``); reading a server
path is a separate, explicit function used by the CLI.
"""
import json
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from financeagents.onboarding import MAX_POSITIONS, MIN_POSITIONS
from financeagents.state import UserProfile, empty_profile

MAX_SESSION_BYTES = 64 * 1024


class ProfileModel(BaseModel):
    model_config = ConfigDict(extra="ignore")

    location: str = Field("", max_length=100)
    base_currency: Literal["", "INR", "USD"] = ""
    budget: float = Field(0.0, ge=0)
    budget_type: Literal["", "lumpsum", "monthly"] = ""
    goals: list[str] = Field(default_factory=list, max_length=20)
    risk_tolerance: Literal["", "conservative", "moderate", "aggressive"] = ""
    investment_horizon: str = Field("", max_length=50)
    horizon_years: float = Field(0.0, ge=0, le=60)
    preferred_markets: list[Literal["us", "india"]] = Field(default_factory=list)
    preferred_sectors: list[str] = Field(default_factory=list, max_length=20)
    num_positions: int = Field(0, ge=0, le=MAX_POSITIONS)
    existing_investments: list[dict] = Field(default_factory=list, max_length=50)
    completed_onboarding: bool = False


def save_session(user_profile: UserProfile, analysis_history: Optional[list] = None, filepath: str = None) -> str:
    """Serialise the session; optionally write it to ``filepath``. Returns the JSON string."""
    session_data = {
        "version": "3.0",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "project": "FinanceAgents",
        "user_profile": dict(user_profile) if user_profile else {},
        "analysis_history": analysis_history or [],
    }
    json_str = json.dumps(session_data, indent=2, default=str)
    if filepath:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)
    return json_str


def _validate(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Session file must be a JSON object.")
    raw_profile = data.get("user_profile")
    if not isinstance(raw_profile, dict):
        raise ValueError("Invalid session file: 'user_profile' must be an object.")
    try:
        model = ProfileModel.model_validate(raw_profile)
    except ValidationError as e:
        fields = ", ".join(".".join(str(p) for p in err["loc"]) for err in e.errors())
        raise ValueError(f"Invalid profile fields: {fields}") from e
    profile = UserProfile(**{**empty_profile(), **model.model_dump()})
    if profile["completed_onboarding"] and not MIN_POSITIONS <= profile["num_positions"] <= MAX_POSITIONS:
        raise ValueError("Invalid profile fields: num_positions")
    history = data.get("analysis_history", [])
    return {
        "user_profile": profile,
        "analysis_history": history if isinstance(history, list) else [],
        "saved_at": str(data.get("saved_at", "")),
        "version": str(data.get("version", "")),
    }


def load_session_from_str(text: str) -> dict:
    """Parse and validate uploaded session JSON. Never touches the filesystem."""
    if len(text.encode("utf-8")) > MAX_SESSION_BYTES:
        raise ValueError("Session file is too large.")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Session file is not valid JSON: {e.msg}") from e
    return _validate(data)


def load_session_from_path(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return load_session_from_str(f.read(MAX_SESSION_BYTES + 1))


def get_profile_summary(profile: UserProfile) -> str:
    """Human-readable profile summary (Markdown)."""
    if not profile.get("completed_onboarding"):
        return "Profile not yet completed."
    budget = profile.get("budget") or 0
    lines = [
        "👤 **User Profile**",
        f"📍 Location: {profile.get('location') or 'N/A'}",
        f"💰 Budget: {budget:,.2f} {profile.get('base_currency') or ''} ({profile.get('budget_type') or 'N/A'})",
        f"🎯 Goals: {', '.join(profile.get('goals') or []) or 'N/A'}",
        f"⚖️ Risk: {profile.get('risk_tolerance') or 'N/A'}",
        f"📅 Horizon: {profile.get('investment_horizon') or 'N/A'}",
        f"📊 Positions: {profile.get('num_positions') or 'N/A'}",
        f"🌍 Markets: {', '.join(profile.get('preferred_markets') or []) or 'All'}",
    ]
    existing = profile.get("existing_investments") or []
    if existing:
        lines.append(f"💼 Existing: {', '.join(str(e.get('description', e)) for e in existing)}")
    return "\n".join(lines)
