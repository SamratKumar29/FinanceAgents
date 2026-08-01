"""config.py -- LLM connection settings. Supports both .env and UI-configurable values."""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Core LLM settings (loaded from .env, can be overridden by UI) ---
OPENAI_API_KEY = os.getenv("LLM_API_KEY", "")
OPENAI_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2.5")


# --- Model suggestions for the Streamlit UI ---
SUGGESTED_MODELS = [
    {"name": "xiaomi/mimo-v2.5", "tool_calling": False, "note": "Free, good for synthesis (default)"},
    {"name": "openai/gpt-4o-mini", "tool_calling": True, "note": "Best budget option with tool calling"},
    {"name": "openai/gpt-4o", "tool_calling": True, "note": "High quality, full tool calling"},
    {"name": "anthropic/claude-3.5-sonnet", "tool_calling": True, "note": "Excellent reasoning + tool calling"},
    {"name": "google/gemini-2.0-flash", "tool_calling": True, "note": "Fast, good tool calling"},
    {"name": "meta-llama/llama-3.1-70b-instruct", "tool_calling": False, "note": "Large open model, no native tool calling"},
]

# Models known to support tool/function calling via OpenRouter
TOOL_CALLING_MODELS = {
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1-mini",
    "openai/gpt-4.1",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-2.0-flash",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "deepseek/deepseek-chat",
    "qwen/qwen-2.5-72b-instruct",
}


def supports_tool_calling(model: str) -> bool:
    """Check if a model is known to support tool/function calling."""
    if not model:
        return False
    model_lower = model.lower().strip()
    for known in TOOL_CALLING_MODELS:
        if known.lower() in model_lower or model_lower in known.lower():
            return True
    # Also check if any substring matches (e.g. "gpt-4o-mini" in "openrouter/openai/gpt-4o-mini")
    if any(kw in model_lower for kw in ["gpt-4o", "gpt-4.1", "claude", "gemini", "deepseek", "qwen"]):
        return True
    return False


def update_config(api_key: str = None, base_url: str = None, model: str = None):
    """Update LLM settings at runtime (called from Streamlit UI)."""
    global OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
    if api_key is not None:
        OPENAI_API_KEY = api_key
    if base_url is not None:
        OPENAI_BASE_URL = base_url
    if model is not None:
        LLM_MODEL = model