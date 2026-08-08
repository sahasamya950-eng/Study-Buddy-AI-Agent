"""
TopicExtractionTool — Professional, Gemini-quality key topic, term, and formula extractor.
"""

import os
import random
from typing import Type, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.validators import validate_text_input
from utils.helper import safe_json_parse
from utils.logger import logger


class TopicExtractorInput(BaseModel):
    text: str = Field(..., description="The document or study text from which to extract key topics.")


class TopicExtractionTool(BaseTool):
    name: str = "topic_extraction_tool"
    description: str = "Extracts main topics, subtopics, important keywords, definitions, and formulas from notes."
    args_schema: Type[BaseModel] = TopicExtractorInput

    def _run(self, text: str) -> str:
        logger.info("TopicExtractionTool invoked")
        is_valid, err_msg = validate_text_input(text, min_chars=15)
        if not is_valid:
            return f"**Error:** {err_msg}"

        if os.getenv("DISABLE_LLM_API") != "1":
            try:
                from config import GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL

                prompt = (
                    "You are an expert academic analyst. Carefully read the document below and extract a structured knowledge map.\n\n"
                    "Return a valid JSON object with exactly these keys:\n"
                    "- \"main_topics\": list of strings — the primary subjects covered (3–6 topics)\n"
                    "- \"subtopics\": list of strings — supporting sub-concepts under the main topics\n"
                    "- \"key_terms\": list of strings — important vocabulary and technical terms\n"
                    "- \"definitions\": list of objects with keys \"term\" and \"definition\" — define all key terms found\n"
                    "- \"formulas\": list of strings — any equations, formulas, or rules present\n"
                    "- \"summary\": string — one-sentence description of the document's scope\n\n"
                    "Rules:\n"
                    "- Extract ONLY what is present in the document. Do not invent topics.\n"
                    "- Output ONLY the raw JSON object. No markdown, no explanation.\n\n"
                    f"Document:\n\n{text}"
                )

                llm = None
                if GEMINI_API_KEY and GEMINI_API_KEY.strip():
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model=DEFAULT_GEMINI_MODEL, google_api_key=GEMINI_API_KEY,
                        max_retries=0, timeout=30, temperature=0.1
                    )
                elif OPENAI_API_KEY and OPENAI_API_KEY.strip():
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=DEFAULT_OPENAI_MODEL, api_key=OPENAI_API_KEY,
                        max_retries=0, timeout=30, temperature=0.1
                    )

                if llm:
                    res = llm.invoke(prompt)
                    parsed = safe_json_parse(res.content)
                    if parsed and isinstance(parsed, dict):
                        return self._format_output(parsed)

            except Exception as e:
                logger.warning(f"TopicExtractionTool LLM failed: {e}")

        return self._fallback_extract(text)

    def _format_output(self, data: Dict[str, Any]) -> str:
        main_topics = "\n".join(f"- **{t}**" for t in data.get("main_topics", [])) or "- General Study Material"
        subtopics = "\n".join(f"  - {s}" for s in data.get("subtopics", [])) or "  - Fundamentals"
        key_terms = ", ".join(f"`{k}`" for k in data.get("key_terms", [])) or "`study`, `concepts`"

        defs = data.get("definitions", [])
        if defs and isinstance(defs[0], dict):
            defs_str = "\n".join(f"- **{d.get('term', '')}:** {d.get('definition', '')}" for d in defs)
        else:
            defs_str = "\n".join(f"- {d}" for d in defs) if defs else "- No explicit definitions detected."

        formulas = data.get("formulas", [])
        formulas_str = "\n".join(f"- `{f}`" for f in formulas) if formulas else "- No explicit formulas detected in the document."

        summary = data.get("summary", "")

        return (
            f"## Topics & Key Concepts\n\n"
            + (f"**Document Scope:** {summary}\n\n" if summary else "")
            + f"### Main Topics\n{main_topics}\n\n"
            f"### Subtopics\n{subtopics}\n\n"
            f"### Key Terminology\n{key_terms}\n\n"
            f"### Definitions\n{defs_str}\n\n"
            f"### Formulas & Rules\n{formulas_str}"
        )

    def _fallback_extract(self, text: str) -> str:
        import re
        words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)
        seen = list(dict.fromkeys(words))[:8]
        sentences = re.split(r'(?<=[.!?])\s+', text)
        definitions = [s.strip() for s in sentences if " is " in s.lower() or " refers to " in s.lower() or " defined as " in s.lower()]
        defs_str = "\n".join(f"- {d}" for d in definitions[:4]) if definitions else "- Review document for definitions."
        topics_str = "\n".join(f"- **{w}**" for w in seen[:5]) if seen else "- Core academic subject"

        return (
            f"## Topics & Key Concepts\n\n"
            f"### Main Topics\n{topics_str}\n\n"
            f"### Definitions Found\n{defs_str}\n\n"
            f"### Formulas & Rules\n- No explicit formulas detected in this document."
        )

    async def _arun(self, text: str) -> str:
        return self._run(text)
