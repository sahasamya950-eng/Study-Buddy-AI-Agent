"""
NotesSummarizerTool — Professional, Gemini-quality document summarizer.
Zero repetition across sections. Aggressively strips multi-file merge artifacts.
"""

import os
import re
from typing import Type, List, Set
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.validators import validate_text_input, validate_summary_type
from utils.logger import logger


class SummarizerInput(BaseModel):
    text: str = Field(..., description="The document text or notes to summarize.")
    summary_type: str = Field("Medium", description="Short, Medium, or Detailed.")


class NotesSummarizerTool(BaseTool):
    name: str = "notes_summarizer_tool"
    description: str = "Summarizes uploaded study notes and PDFs with high accuracy and zero repetition."
    args_schema: Type[BaseModel] = SummarizerInput

    # ─────────────────────────────────────────────────────────────────────
    # Text cleaning — strips ALL merge artifacts before any processing
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _clean(text: str) -> str:
        t = text
        # Remove ===...=== separator blocks (including content between them like DOCUMENT: filename)
        t = re.sub(r'={3,}[\s\S]*?={3,}', '\n', t)
        # Remove any residual DOCUMENT: ... lines
        t = re.sub(r'^\s*DOCUMENT\s*:.*$', '', t, flags=re.MULTILINE)
        # Remove lines that are only dashes, equals, or underscores
        t = re.sub(r'^\s*[-=_]{3,}\s*$', '', t, flags=re.MULTILINE)
        # Remove lines that are just numbers or single words (page markers)
        t = re.sub(r'^\s*\d+\s*$', '', t, flags=re.MULTILINE)
        # Collapse excess whitespace
        t = re.sub(r'\n{3,}', '\n\n', t)
        t = re.sub(r'[ \t]{2,}', ' ', t)
        return t.strip()

    # ─────────────────────────────────────────────────────────────────────
    # Sentence extraction — unique, meaningful, no headers
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _sentences(text: str) -> List[str]:
        """Extract unique, clean sentences from document text."""
        # Split on sentence boundaries
        raw = re.split(r'(?<=[.!?])\s+', text)
        seen: Set[str] = set()
        out: List[str] = []
        for s in raw:
            s = s.strip()
            # Clean page numbers and header fragments at beginning of sentence
            s = re.sub(r'^(?:Page\s*\d+\s*[-—–]*\s*)+', '', s, flags=re.IGNORECASE).strip()
            s = re.sub(r'^(?:[A-Z]{2,}(?:\s+[A-Z]{2,})*\s+)+', '', s).strip()
            s = re.sub(r'^[A-Z0-9\s/–—-]{3,}\s*[-—–]+\s*', '', s).strip()
            s = re.sub(r'^(?:Chapter\s*\d+|Section\s*\d+|Page\s*\d+)\s*[:—–-]*\s*', '', s, flags=re.IGNORECASE).strip()
            s = s.strip("-:;, •")
            # Length filter
            if len(s) < 30 or len(s) > 400:
                continue
            # Skip separator artifacts
            if re.search(r'={3,}|DOCUMENT\s*:', s, re.IGNORECASE):
                continue
            # Skip lines ending with colon (list headers)
            if s.endswith(':'):
                continue
            # Skip mostly-uppercase lines (headers)
            alpha = re.sub(r'[^a-zA-Z]', '', s)
            if alpha and sum(1 for c in alpha if c.isupper()) / len(alpha) > 0.55:
                continue
            # Dedup key
            key = re.sub(r'[^a-z0-9 ]', '', s.lower())
            key = re.sub(r'\s+', ' ', key).strip()[:90]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    # ─────────────────────────────────────────────────────────────────────
    # Output deduplication — removes repeated lines across ALL sections
    # ─────────────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup(text: str) -> str:
        lines = text.split('\n')
        seen: Set[str] = set()
        out: List[str] = []
        prev_blank = False
        for line in lines:
            stripped = line.strip()
            # Always keep headings and blank lines
            if not stripped:
                if not prev_blank:
                    out.append(line)
                prev_blank = True
                continue
            prev_blank = False
            if stripped.startswith('#') or stripped.startswith('---') or stripped.startswith('*Study'):
                out.append(line)
                continue
            # Build dedup key: strip bullet/number prefix, lowercase, no punctuation
            key = re.sub(r'^[-*•]\s+', '', stripped)
            key = re.sub(r'^\d+[\.\)]\s+', '', key)
            key = re.sub(r'[^a-z0-9 ]', '', key.lower())
            key = re.sub(r'\s+', ' ', key).strip()[:90]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(line)
        return '\n'.join(out)

    # ─────────────────────────────────────────────────────────────────────
    # Main entry point
    # ─────────────────────────────────────────────────────────────────────

    def _run(self, text: str, summary_type: str = "Medium") -> str:
        logger.info(f"NotesSummarizerTool invoked — summary_type={summary_type}")
        is_valid, err_msg = validate_text_input(text, min_chars=15)
        if not is_valid:
            return f"**Error:** {err_msg}"

        depth = validate_summary_type(summary_type)
        clean = self._clean(text)

        # ── Try LLM ──────────────────────────────────────────────────────
        if os.getenv("DISABLE_LLM_API") != "1":
            try:
                from config import GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL

                if depth == "Short":
                    fmt = (
                        "Write a concise summary as ONE paragraph of 4–6 sentences. "
                        "No headers, no bullet points. Cover: what the document is about, "
                        "the main theme, and the 2–3 most important concepts."
                    )
                elif depth == "Detailed":
                    fmt = (
                        "Write a detailed academic breakdown with EXACTLY these sections.\n"
                        "Every section must contain DIFFERENT information — never repeat:\n\n"
                        "## Document Overview\n"
                        "(2–3 sentences: what is this document about and what is its main argument?)\n\n"
                        "## Core Concepts & Definitions\n"
                        "(Define every key term from the text. Bold each term with **term**.)\n\n"
                        "## Key Points\n"
                        "(6–8 bullet points — each one a DIFFERENT fact from the document)\n\n"
                        "## Exam Takeaways\n"
                        "(4–5 bullet points on what a student must remember — "
                        "MUST be different from Key Points above)\n\n"
                        "RULE: No sentence or idea may appear in more than one section."
                    )
                else:  # Medium
                    fmt = (
                        "Write a study summary with EXACTLY these 3 sections.\n"
                        "Each section must contain COMPLETELY DIFFERENT information:\n\n"
                        "## Overview\n"
                        "(2 sentences only: state what the document is about)\n\n"
                        "## Key Topics\n"
                        "(One bullet per main topic with a brief explanation — "
                        "do NOT repeat the Overview sentences)\n\n"
                        "## Revision Notes\n"
                        "(3–5 exam-prep bullets — "
                        "MUST be different from everything above)\n\n"
                        "STRICT RULE: Every bullet and sentence must be unique across all sections."
                    )

                # Smart truncation for large/multi-file docs
                max_c = 55000
                if len(clean) > max_c:
                    doc = clean[:35000] + "\n\n[...]\n\n" + clean[-15000:]
                else:
                    doc = clean

                prompt = (
                    f"You are an expert academic tutor. Summarise the document below.\n\n"
                    f"Depth: {depth}\n\n"
                    f"Format:\n{fmt}\n\n"
                    "Absolute rules:\n"
                    "1. NEVER repeat any sentence, phrase, or idea across different sections.\n"
                    "2. Base every point on the document — do not invent facts.\n"
                    "3. Use clean Markdown formatting.\n"
                    "4. Write in professional academic English.\n\n"
                    f"Document:\n\n{doc}"
                )

                llm = None
                if GEMINI_API_KEY and GEMINI_API_KEY.strip():
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model=DEFAULT_GEMINI_MODEL,
                        google_api_key=GEMINI_API_KEY,
                        max_retries=0, timeout=60, temperature=0.15
                    )
                elif OPENAI_API_KEY and OPENAI_API_KEY.strip():
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=DEFAULT_OPENAI_MODEL,
                        api_key=OPENAI_API_KEY,
                        max_retries=0, timeout=60, temperature=0.15
                    )

                if llm:
                    res = llm.invoke(prompt)
                    if res and res.content and len(res.content.strip()) > 30:
                        return self._dedup(res.content.strip())

            except Exception as e:
                logger.warning(f"NotesSummarizerTool LLM failed: {e}")

        return self._fallback(clean, depth)

    # ─────────────────────────────────────────────────────────────────────
    # Fallback — fully deduplicated, no separator artifacts
    # ─────────────────────────────────────────────────────────────────────

    def _fallback(self, clean_text: str, depth: str) -> str:
        sents = self._sentences(clean_text)
        n = len(sents)

        def take(a: int, b: int) -> List[str]:
            return sents[a:b] if n > a else []

        if depth == "Short":
            items = take(0, 5)
            pts = "\n\n".join(
                f"* **Point {i} ({' '.join(s.split()[:4]).rstrip(',.:;')}):**  \n  {s}"
                for i, s in enumerate(items, 1)
            )
            return f"### 📑 Summary of Key Takeaways\n\n{pts}\n"

        elif depth == "Detailed":
            items = take(0, 10)
            pts = "\n\n".join(
                f"* **Point {i} ({' '.join(s.split()[:4]).rstrip(',.:;')}):**  \n  {s}"
                for i, s in enumerate(items, 1)
            )
            return f"### 📑 Comprehensive Point-by-Point Analysis\n\n{pts}\n\n---\n*💡 Review all points above for active recall and exam preparation.*"

        else:  # Medium
            items = take(0, 7)
            pts = "\n\n".join(
                f"* **Point {i} ({' '.join(s.split()[:4]).rstrip(',.:;')}):**  \n  {s}"
                for i, s in enumerate(items, 1)
            )
            return f"### 📑 Key Takeaways & Point-by-Point Summary\n\n{pts}\n"

    async def _arun(self, text: str, summary_type: str = "Medium") -> str:
        return self._run(text, summary_type)
