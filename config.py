"""config.py -- LLM connection settings. BYOK pattern, same as the rest of
this sprint. Set these in a .env file (see .env.example)."""
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("LLM_API_KEY")
OPENAI_BASE_URL = os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "xiaomi/mimo-v2.5")
