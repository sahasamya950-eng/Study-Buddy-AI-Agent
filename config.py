"""
Configuration management for Study Buddy AI Agent.
Loads environment variables safely from .env, system environment, or Streamlit secrets.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DATA_DIR = BASE_DIR / "sample_data"
TEST_FILES_DIR = BASE_DIR / "test_files"
ASSETS_DIR = BASE_DIR / "assets"
SCREENSHOTS_DIR = BASE_DIR / "screenshots"

# Ensure directories exist
for directory in [SAMPLE_DATA_DIR, TEST_FILES_DIR, ASSETS_DIR, SCREENSHOTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Load .env file if present (checks study buddy ai/.env and workspace root .env)
ENV_FILES = [BASE_DIR / ".env", BASE_DIR.parent / ".env"]
for env_path in ENV_FILES:
    if env_path.exists():
        load_dotenv(env_path, override=True)
load_dotenv()

# Helper function to sanitize keys and ignore dummy placeholders
PLACEHOLDER_KEYS = {
    "your_gemini_api_key_here",
    "use your gemini api key here",
    "your_api_key_here",
    "your_openai_api_key_here",
    "",
}

def clean_key(val: str) -> str:
    cleaned = (val or "").strip()
    if cleaned.lower() in PLACEHOLDER_KEYS:
        return ""
    return cleaned

# Resolve Gemini API Key (Priority: os.getenv -> Streamlit secrets)
raw_gemini = os.getenv("GEMINI_API_KEY", "")
if not clean_key(raw_gemini):
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            raw_gemini = st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass

GEMINI_API_KEY = clean_key(raw_gemini)
if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

# Resolve OpenAI API Key
raw_openai = os.getenv("OPENAI_API_KEY", "")
if not clean_key(raw_openai):
    try:
        import streamlit as st
        if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
            raw_openai = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

OPENAI_API_KEY = clean_key(raw_openai)
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Application Settings
APP_NAME = "Study Buddy AI Agent"
APP_VERSION = "1.0.0"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
DEFAULT_DIFFICULTY = os.getenv("DEFAULT_DIFFICULTY", "Medium")
try:
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
except ValueError:
    MAX_FILE_SIZE_MB = 25

# Model Configuration
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"

# Tool Settings
MAX_QUIZ_QUESTIONS = 20
MIN_QUIZ_QUESTIONS = 1
DEFAULT_QUIZ_QUESTIONS = 5
