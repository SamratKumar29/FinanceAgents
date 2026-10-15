"""config.py -- Environment loading and UI model suggestions.

LLM settings live in ``llm.Settings`` (immutable, per session). Nothing here is mutated at runtime.
"""
import os

from dotenv import load_dotenv

from llm import Settings

if not os.getenv("FINANCEAGENTS_SKIP_DOTENV"):
    load_dotenv()


def server_settings() -> Settings:
    """Operator-configured settings from the environment / .env."""
    return Settings.from_env()


# Model suggestions for the Streamlit UI (names only; availability is checked at call time).
SUGGESTED_MODELS = [
    "xiaomi/mimo-v2.5",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "anthropic/claude-3.5-sonnet",
    "google/gemini-2.0-flash",
    "meta-llama/llama-3.1-70b-instruct",
]
