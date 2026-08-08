"""
FlashcardGeneratorTool — Professional, Gemini-quality flashcard generator.
Produces document-grounded front/back flashcards for active recall learning.
"""

import os
from typing import Type, List, Dict
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.validators import validate_text_input, validate_difficulty
from utils.helper import safe_json_parse, generate_fallback_flashcards
from utils.logger import logger


class FlashcardInput(BaseModel):
    text: str = Field(..., description="The study notes or text content to generate flashcards from.")
    count: int = Field(5, description="Number of flashcards to generate (1 to 20).")
    difficulty: str = Field("Medium", description="Flashcard difficulty: Easy, Medium, or Hard.")


class FlashcardGeneratorTool(BaseTool):
    name: str = "flashcard_generator_tool"
    description: str = (
        "Generates accurate, document-grounded front-and-back study flashcards "
        "for active recall learning and exam preparation."
    )
    args_schema: Type[BaseModel] = FlashcardInput

    def _run(self, text: str, count: int = 5, difficulty: str = "Medium") -> str:
        logger.info(f"FlashcardGeneratorTool — count={count}, difficulty={difficulty}")
        is_valid, err_msg = validate_text_input(text, min_chars=15)
        if not is_valid:
            return f"**Error:** {err_msg}"

        num_c = max(1, min(count, 20))
        diff = validate_difficulty(difficulty)

        cards = []
        if os.getenv("DISABLE_LLM_API") != "1":
            try:
                from config import GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL

                prompt = (
                    f"You are an expert academic tutor. Create exactly {num_c} study flashcards based ONLY on the document below.\n\n"
                    f"**Difficulty:** {diff}\n\n"
                    "**Flashcard Rules:**\n"
                    "- The 'front' must be a clear, specific question or term from the document.\n"
                    "- The 'back' must be a concise, accurate answer directly grounded in the document text.\n"
                    "- Cover a variety of concepts: definitions, formulas, processes, comparisons.\n"
                    "- Do NOT copy generic templates — make each card specific to this document.\n\n"
                    "Output ONLY a valid JSON array of objects with keys: \"front\", \"back\", \"difficulty\"\n\n"
                    f"**Document:**\n\n{text}"
                )

                llm = None
                if GEMINI_API_KEY and GEMINI_API_KEY.strip():
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model=DEFAULT_GEMINI_MODEL, google_api_key=GEMINI_API_KEY,
                        max_retries=0, timeout=30, temperature=0.4
                    )
                elif OPENAI_API_KEY and OPENAI_API_KEY.strip():
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=DEFAULT_OPENAI_MODEL, api_key=OPENAI_API_KEY,
                        max_retries=0, timeout=30, temperature=0.4
                    )

                if llm:
                    res = llm.invoke(prompt)
                    parsed = safe_json_parse(res.content)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        cards = parsed

            except Exception as e:
                logger.warning(f"FlashcardGeneratorTool LLM failed: {e}")

        if not cards:
            cards = generate_fallback_flashcards(text, count=num_c, difficulty=diff)

        return self._format_output(cards, diff)

    def _format_output(self, cards: List[Dict[str, str]], difficulty: str) -> str:
        lines = [f"## Flashcards — {difficulty} Level ({len(cards)} Cards)\n"]
        for idx, card in enumerate(cards, start=1):
            front = card.get("front", "").strip()
            back = card.get("back", "").strip()
            lines.append(f"### Card {idx}")
            lines.append(f"**Q:** {front}")
            lines.append(f"**A:** {back}")
            lines.append("")
        return "\n".join(lines)

    async def _arun(self, text: str, count: int = 5, difficulty: str = "Medium") -> str:
        return self._run(text, count, difficulty)
