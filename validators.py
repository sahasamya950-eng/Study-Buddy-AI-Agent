"""
Input validation utilities for Study Buddy AI Agent.
"""

import os
from typing import Tuple, Optional
from config import MAX_FILE_SIZE_MB
from utils.constants import DIFFICULTY_LEVELS, REVISION_PLAN_DAYS

def validate_file_size(file_bytes: bytes, max_mb: int = MAX_FILE_SIZE_MB) -> Tuple[bool, Optional[str]]:
    """Validates if file size is within limits."""
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > max_mb:
        return False, f"File size ({size_mb:.2f} MB) exceeds maximum allowed limit of {max_mb} MB."
    return True, None

def validate_text_input(text: str, min_chars: int = 10) -> Tuple[bool, Optional[str]]:
    """Validates non-empty text input."""
    if not text or not isinstance(text, str):
        return False, "Input text is empty or invalid."
    if len(text.strip()) < min_chars:
        return False, f"Input text must contain at least {min_chars} characters."
    return True, None

def validate_difficulty(difficulty: str) -> str:
    """Validates and normalizes difficulty level."""
    if not difficulty or not isinstance(difficulty, str):
        return "Medium"
    title_diff = difficulty.strip().capitalize()
    if title_diff in DIFFICULTY_LEVELS:
        return title_diff
    return "Medium"

def validate_summary_type(summary_type: str) -> str:
    """Validates and normalizes summary length/depth type."""
    if not summary_type or not isinstance(summary_type, str):
        return "Medium"
    st = summary_type.strip().capitalize()
    if st in ["Short", "Medium", "Detailed"]:
        return st
    return "Medium"

def validate_revision_days(days: int) -> int:
    """Validates revision plan days, ensuring a minimum of 1 and maximum of 30 days."""
    try:
        val = int(days)
        return max(1, min(val, 30))
    except (ValueError, TypeError):
        return 7

def validate_quiz_count(num_questions: int) -> int:
    """Clamps number of quiz questions between 1 and 20."""
    try:
        val = int(num_questions)
        return max(1, min(val, 20))
    except (ValueError, TypeError):
        return 5
