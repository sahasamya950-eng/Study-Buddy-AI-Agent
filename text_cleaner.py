"""
Text cleaning and normalization utility functions.
"""

import re
from typing import List

def clean_text(text: str) -> str:
    """
    Cleans raw extracted text from PDFs or text documents.
    Removes extra whitespace, non-printable characters, and standard artifacts.
    """
    if not text or not isinstance(text, str):
        return ""

    # Replace multiple spaces/newlines with clean line breaks & single spaces
    text = re.sub(r'\r\n|\r', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Remove control characters except newline
    text = re.sub(r'[\x00-\x09\x0B-\x1F\x7F]', '', text)
    
    # Remove excessive blank lines (more than 2 consecutive newlines)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def split_into_paragraphs(text: str) -> List[str]:
    """Splits cleaned text into distinct non-empty paragraphs."""
    cleaned = clean_text(text)
    if not cleaned:
        return []
    paragraphs = [p.strip() for p in cleaned.split('\n\n') if p.strip()]
    return paragraphs

def truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncates text to a maximum character length safely."""
    cleaned = clean_text(text)
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars] + "... [Truncated for brevity]"
