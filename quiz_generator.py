"""
QuizGeneratorTool — Professional, Gemini-quality quiz generator.
Produces document-grounded, varied, and accurately marked quiz questions.
"""

import os
import random
from typing import Type, List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.validators import validate_text_input, validate_difficulty, validate_quiz_count
from utils.helper import safe_json_parse, generate_fallback_quiz
from utils.logger import logger


class QuizGeneratorInput(BaseModel):
    text: str = Field(..., description="Study notes or source document content to base quiz on.")
    num_questions: int = Field(5, description="Number of questions to generate (1 to 20).")
    difficulty: str = Field("Medium", description="Quiz difficulty: Easy, Medium, or Hard.")
    question_type: str = Field("Mixed", description="Question type: Multiple Choice, True/False, Short Answer, or Mixed.")


class QuizGeneratorTool(BaseTool):
    name: str = "quiz_generator_tool"
    description: str = (
        "Generates high-quality, document-grounded quizzes with MCQ, True/False, and Short Answer questions. "
        "Every question is directly derived from the uploaded document content."
    )
    args_schema: Type[BaseModel] = QuizGeneratorInput

    def _run(self, text: str, num_questions: int = 5, difficulty: str = "Medium", question_type: str = "Mixed") -> str:
        logger.info(f"QuizGeneratorTool — count={num_questions}, diff={difficulty}, type={question_type}")
        is_valid, err_msg = validate_text_input(text, min_chars=15)
        if not is_valid:
            return f"**Error:** {err_msg}"

        num_q = validate_quiz_count(num_questions)
        diff = validate_difficulty(difficulty)

        quiz_items = []
        if os.getenv("DISABLE_LLM_API") != "1":
            try:
                from config import GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL

                prompt = (
                    f"You are an expert exam setter. Create exactly {num_q} quiz questions based ONLY on the document below.\n\n"
                    f"**Difficulty:** {diff}\n"
                    f"**Question Style:** {question_type}\n\n"
                    "**Requirements:**\n"
                    "- Every question must come from the actual document content — no invented facts.\n"
                    "- For Multiple Choice: provide exactly 4 options, only one correct.\n"
                    "- For True/False: provide [\"True\", \"False\"] as options.\n"
                    "- The 'explanation' field must provide a deep, educational rationale explaining WHY the answer is correct, what concept it illustrates, and why alternative options are incorrect. Do NOT simply write 'Directly stated in the document'.\n"
                    "- Vary question types for Mixed mode.\n\n"
                    "Output ONLY a valid JSON array. Each object must have:\n"
                    "  { \"id\": number, \"question\": string, \"options\": list, \"correct_answer\": string, \"explanation\": string, \"type\": string, \"difficulty\": string }\n\n"
                    f"**Document:**\n\n{text}"
                )

                llm = None
                if GEMINI_API_KEY and GEMINI_API_KEY.strip():
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(
                        model=DEFAULT_GEMINI_MODEL, google_api_key=GEMINI_API_KEY,
                        max_retries=0, timeout=30, temperature=0.5
                    )
                elif OPENAI_API_KEY and OPENAI_API_KEY.strip():
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(
                        model=DEFAULT_OPENAI_MODEL, api_key=OPENAI_API_KEY,
                        max_retries=0, timeout=30, temperature=0.5
                    )

                if llm:
                    res = llm.invoke(prompt)
                    parsed = safe_json_parse(res.content)
                    if isinstance(parsed, list) and len(parsed) > 0:
                        quiz_items = parsed

            except Exception as e:
                logger.warning(f"QuizGeneratorTool LLM failed: {e}")

        if not quiz_items:
            quiz_items = generate_fallback_quiz(text, num_questions=num_q, difficulty=diff, question_type=question_type)

        return self._format_output(quiz_items, diff)

    def _format_output(self, questions: List[Dict[str, Any]], difficulty: str) -> str:
        lines = [f"## Quiz — {difficulty} Level ({len(questions)} Questions)\n"]
        for i, q in enumerate(questions, start=1):
            q_text = q.get("question", "")
            q_type = q.get("type", "Question")
            opts = q.get("options", [])
            answer = q.get("correct_answer", "")
            explanation = q.get("explanation", "")

            lines.append(f"**Q{i}. {q_text}**")
            lines.append(f"*Type: {q_type} | Difficulty: {q.get('difficulty', difficulty)}*")
            if opts:
                option_letters = ["A", "B", "C", "D"]
                for j, opt in enumerate(opts):
                    letter = option_letters[j] if j < len(option_letters) else str(j + 1)
                    lines.append(f"  {letter}) {opt}")
            lines.append(f"\n✅ **Answer:** {answer}")
            if explanation:
                lines.append(f"💡 **Explanation:** {explanation}")
            lines.append("")
        return "\n".join(lines)

    async def _arun(self, text: str, num_questions: int = 5, difficulty: str = "Medium", question_type: str = "Mixed") -> str:
        return self._run(text, num_questions, difficulty, question_type)
