"""
Utilities package for Study Buddy AI Agent.
Shared helpers, validators, logging, PDF extraction, and vector store.
"""

from utils.logger import logger
from utils.constants import (
    DIFFICULTY_LEVELS,
    SUMMARY_TYPES,
    REVISION_PLAN_DAYS,
    MAX_REVISION_DAYS,
    EXPLANATION_MODES,
    SYSTEM_PROMPT_TEMPLATE,
)
from utils.text_cleaner import clean_text, split_into_paragraphs, truncate_text
from utils.validators import validate_file_size, validate_text_input, validate_summary_type
from utils.pdf_utils import extract_text_from_pdf_stream, read_pdf_file_path
from utils.vector_store import vector_store, StudyVectorStore
from utils.export_utils import create_pdf_report
from utils.helper import shuffle_flashcards, generate_fallback_quiz, generate_fallback_flashcards

__all__ = [
    "logger",
    "DIFFICULTY_LEVELS",
    "SUMMARY_TYPES",
    "REVISION_PLAN_DAYS",
    "MAX_REVISION_DAYS",
    "EXPLANATION_MODES",
    "SYSTEM_PROMPT_TEMPLATE",
    "clean_text",
    "split_into_paragraphs",
    "truncate_text",
    "validate_file_size",
    "validate_text_input",
    "validate_summary_type",
    "extract_text_from_pdf_stream",
    "read_pdf_file_path",
    "vector_store",
    "StudyVectorStore",
    "create_pdf_report",
    "shuffle_flashcards",
    "generate_fallback_quiz",
    "generate_fallback_flashcards",
]
