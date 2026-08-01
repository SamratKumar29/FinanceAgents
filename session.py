"""
session.py -- Save and load user profiles as JSON files.

Enables session persistence: users can download their profile at the end of a session
and upload it next time to resume where they left off.
"""
import json
from datetime import datetime, timezone
from typing import Optional
from state import UserProfile


def save_session(
    user_profile: UserProfile,
    conversation_summary: str = "",
    analysis_history: list = None,
    filepath: str = None,
) -> str:
    """
    Save user session to a JSON file.

    Args:
        user_profile: The user's investment profile
        conversation_summary: Brief summary of the conversation
        analysis_history: List of past analysis results
        filepath: Optional explicit filepath (if None, returns JSON string)

    Returns:
        JSON string of the session data
    """
    session_data = {
        "version": "2.0",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "project": "FinanceAgents",
        "user_profile": dict(user_profile) if user_profile else {},
        "conversation_summary": conversation_summary,
        "analysis_history": analysis_history or [],
    }

    json_str = json.dumps(session_data, indent=2, default=str)

    if filepath:
        with open(filepath, "w") as f:
            f.write(json_str)

    return json_str


def load_session(json_input) -> dict:
    """
    Load a user session from a JSON string or file path.

    Args:
        json_input: Either a JSON string or a file path to a JSON file

    Returns:
        Dict with keys: user_profile, conversation_summary, analysis_history, saved_at

    Raises:
        ValueError: If the JSON is invalid or missing required fields
    """
    try:
        if isinstance(json_input, str) and not json_input.strip().startswith("{"):
            # It's a file path
            with open(json_input, "r") as f:
                data = json.load(f)
        else:
            data = json.loads(json_input)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError(f"Failed to load session: {e}")

    # Validate structure
    if "user_profile" not in data:
        raise ValueError("Invalid session file: missing 'user_profile' field")

    # Ensure all profile fields exist
    profile = data["user_profile"]
    defaults = {
        "location": "",
        "base_currency": "",
        "budget": 0.0,
        "budget_type": "",
        "goals": [],
        "risk_tolerance": "",
        "investment_horizon": "",
        "preferred_markets": [],
        "preferred_sectors": [],
        "num_positions": 5,
        "existing_investments": [],
        "completed_onboarding": False,
    }
    for key, default in defaults.items():
        profile.setdefault(key, default)

    return {
        "user_profile": profile,
        "conversation_summary": data.get("conversation_summary", ""),
        "analysis_history": data.get("analysis_history", []),
        "saved_at": data.get("saved_at", ""),
        "version": data.get("version", "2.0"),
    }


def get_profile_summary(profile: UserProfile) -> str:
    """Generate a human-readable summary of the user profile."""
    if not profile.get("completed_onboarding"):
        return "Profile not yet completed."

    lines = []
    lines.append("👤 **User Profile**")
    lines.append(f"📍 Location: {profile.get('location', 'N/A')}")
    lines.append(f"💰 Budget: {profile.get('budget', 0):,.2f} {profile.get('base_currency', 'N/A')} ({profile.get('budget_type', 'N/A')})")
    lines.append(f"🎯 Goals: {', '.join(profile.get('goals', [])) or 'N/A'}")
    lines.append(f"⚖️ Risk: {profile.get('risk_tolerance', 'N/A')}")
    lines.append(f"📅 Horizon: {profile.get('investment_horizon', 'N/A')}")
    lines.append(f"📊 Positions: {profile.get('num_positions', 'N/A')}")
    lines.append(f"🌍 Markets: {', '.join(profile.get('preferred_markets', [])) or 'All'}")

    existing = profile.get("existing_investments", [])
    if existing:
        lines.append(f"💼 Existing: {', '.join(str(e) for e in existing)}")

    return "\n".join(lines)