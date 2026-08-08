"""
AnswerEvaluationTool — Professional, Gemini-quality answer evaluator.
Provides precise scoring, detailed feedback, and actionable improvement suggestions.
"""

import os
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.validators import validate_text_input
from utils.helper import safe_json_parse
from utils.logger import logger


class AnswerEvaluationInput(BaseModel):
    question: str = Field(..., description="The quiz or study question asked.")
    student_answer: str = Field(..., description="The answer submitted by the student.")
    expected_answer: str = Field(..., description="The reference or correct answer.")


class AnswerEvaluationTool(BaseTool):
    name: str = "answer_evaluation_tool"
    description: str = (
        "Evaluates student answers against reference answers. "
        "Returns a score (0–100), correctness status, specific feedback, and improvement suggestions."
    )
    args_schema: Type[BaseModel] = AnswerEvaluationInput

    def _run(self, question: str, student_answer: str, expected_answer: str) -> str:
        logger.info(f"AnswerEvaluationTool — question: {question[:40]}...")
        is_q_valid, err1 = validate_text_input(question, min_chars=3)
        is_sa_valid, err2 = validate_text_input(student_answer, min_chars=1)
        if not is_q_valid or not is_sa_valid:
            return f"**Input Error:** {err1 or err2}"

        if os.getenv("DISABLE_LLM_API") != "1":
            try:
                from config import GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL

                prompt = (
                    "You are an expert academic examiner. Evaluate the student's answer against the expected answer.\n\n"
                    f"**Question:** {question}\n"
                    f"**Student's Answer:** {student_answer}\n"
                    f"**Expected Answer:** {expected_answer}\n\n"
                    "**Evaluation Criteria:**\n"
                    "- Factual accuracy (is the core concept correct?)\n"
                    "- Completeness (did they cover all key points?)\n"
                    "- Clarity and precision of language\n\n"
                    "Return ONLY a valid JSON object with these exact keys:\n"
                    "- \"score\": integer 0–100\n"
                    "- \"is_correct\": boolean (true if score >= 70)\n"
                    "- \"correct_answer\": string (the ideal complete answer)\n"
                    "- \"feedback\": string (specific, constructive praise + what was missing)\n"
                    "- \"suggestions\": string (concrete, actionable next steps to improve)"
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
                    if parsed and isinstance(parsed, dict) and "score" in parsed:
                        return self._format_output(parsed, expected_answer)

            except Exception as e:
                logger.warning(f"AnswerEvaluationTool LLM failed: {e}")

        # Deterministic word-overlap fallback
        result = self._deterministic_eval(student_answer, expected_answer)
        return self._format_output(result, expected_answer)

    def _deterministic_eval(self, student_ans: str, expected_ans: str) -> dict:
        sa = student_ans.strip().lower()
        ea = expected_ans.strip().lower()

        if sa == ea:
            return {
                "score": 100, "is_correct": True,
                "correct_answer": expected_ans,
                "feedback": "Perfect answer. You captured every key point accurately.",
                "suggestions": "Excellent work! Challenge yourself with harder questions on this topic."
            }

        sa_words = set(sa.split())
        ea_words = set(ea.split())
        overlap = len(sa_words & ea_words)
        score = min(90, int((overlap / max(len(ea_words), 1)) * 100))

        if score >= 70:
            return {
                "score": score, "is_correct": True,
                "correct_answer": expected_ans,
                "feedback": f"Good answer. You captured the main idea but could be more precise. Score: {score}/100.",
                "suggestions": "Add specific terminology from the reference answer to strengthen your response."
            }
        elif score >= 40:
            return {
                "score": score, "is_correct": False,
                "correct_answer": expected_ans,
                "feedback": f"Partial answer. Some key points are present but important details are missing. Score: {score}/100.",
                "suggestions": "Re-read the relevant section of your notes and focus on the exact definition and its components."
            }
        else:
            return {
                "score": score if score > 0 else 10, "is_correct": False,
                "correct_answer": expected_ans,
                "feedback": "Your answer does not align with the expected response. The core concept may be unclear.",
                "suggestions": "Return to your study notes for this topic. Focus on understanding the definition before attempting practice questions."
            }

    def _format_output(self, data: dict, expected_answer: str) -> str:
        score = data.get("score", 0)
        is_correct = data.get("is_correct", score >= 70)

        if score >= 80:
            badge = "✅ Excellent"
        elif score >= 60:
            badge = "🟡 Good — Minor Gaps"
        elif score >= 40:
            badge = "🟠 Partial — Needs Work"
        else:
            badge = "❌ Incorrect — Review Required"

        correct_ans = data.get("correct_answer") or expected_answer

        return (
            f"## Answer Evaluation\n\n"
            f"**Result:** {badge}  |  **Score:** {score}/100\n\n"
            f"---\n\n"
            f"### Correct Answer\n{correct_ans}\n\n"
            f"### Feedback\n{data.get('feedback', 'No feedback available.')}\n\n"
            f"### How to Improve\n{data.get('suggestions', 'Review your study notes for this topic.')}"
        )

    async def _arun(self, question: str, student_answer: str, expected_answer: str) -> str:
        return self._run(question, student_answer, expected_answer)
