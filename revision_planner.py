"""
RevisionPlannerTool — Professional, Gemini-quality revision plan generator.
Creates personalised, topic-specific 3-Day, 7-Day, and 15-Day study plans.
"""

import os
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.validators import validate_revision_days
from utils.logger import logger


class RevisionPlannerInput(BaseModel):
    days: int = Field(7, ge=1, le=30, description="Duration of revision plan in days (minimum 1, maximum 30 days).")
    topics: str = Field("General Topics", description="Comma-separated topics or subject summary to revise.")
    hours_per_day: float = Field(2.0, ge=0.5, le=14.0, description="Available study hours per day.")


class RevisionPlannerTool(BaseTool):
    name: str = "revision_planner_tool"
    description: str = "Creates a personalised, structured day-by-day revision plan (up to 30 days maximum) for exam preparation."
    args_schema: Type[BaseModel] = RevisionPlannerInput

    def _run(self, days: int = 7, topics: str = "General Topics", hours_per_day: float = 2.0) -> str:
        logger.info(f"RevisionPlannerTool — {days} days, topics: {topics[:40]}")
        valid_days = validate_revision_days(days)
        hrs = max(0.5, min(hours_per_day, 12.0))
        topic_list = [t.strip() for t in topics.split(",") if t.strip()] or ["Core Concepts", "Problem Solving", "Mock Testing"]

        if os.getenv("DISABLE_LLM_API") != "1":
            try:
                from config import GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL

                prompt = (
                    f"You are an expert academic coach. Create a highly personalised, actionable {valid_days}-day revision plan.\n\n"
                    f"**Topics to Cover:** {', '.join(topic_list)}\n"
                    f"**Daily Study Time:** {hrs} hours\n\n"
                    "**Plan Requirements:**\n"
                    "- Allocate specific topics to specific days in a logical progression.\n"
                    "- Include: active recall sessions, problem-solving practice, spaced repetition checkpoints, and a final mock review day.\n"
                    "- For each day provide: Focus topic, learning tasks (bullet list), time breakdown, and a self-test task.\n"
                    "- Use a Pomodoro rhythm: 25-min study + 5-min break cycles.\n"
                    "- Format using proper Markdown with `##` for day headers and bullet lists for tasks.\n"
                    "- Do NOT use generic filler — make the plan specific to the topics provided."
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
                    if res and res.content and len(res.content.strip()) > 50:
                        return res.content.strip()

            except Exception as e:
                logger.warning(f"RevisionPlannerTool LLM failed: {e}")

        return self._structured_plan(valid_days, topic_list, hrs)

    def _structured_plan(self, days: int, topics: list, hrs: float) -> str:
        lines = [
            f"## {days}-Day Revision Plan\n",
            f"**Topics:** {', '.join(topics)}  |  **Daily Time:** {hrs} hrs\n",
            "---\n"
        ]
        topic_count = len(topics)

        for d in range(1, days + 1):
            if d == 1:
                focus = topics[0] if topics else "Foundations"
                lines += [
                    f"### Day 1 — Foundation & Core Concepts",
                    f"**Focus:** {focus}",
                    f"- Read and annotate key definitions and principles",
                    f"- Create a concept map linking main ideas",
                    f"- Write summary notes in your own words",
                    f"- **Self-test:** Answer 5 questions on {focus} from memory",
                    f"- **Time:** {hrs} hrs  (25-min Pomodoro cycles)\n"
                ]
            elif d == days:
                lines += [
                    f"### Day {d} — Mock Exam & Final Review",
                    f"**Focus:** Full-syllabus active recall",
                    f"- Attempt a timed past paper or full mock quiz",
                    f"- Review all flagged weak areas from previous days",
                    f"- Revisit key formulas and definitions",
                    f"- **Self-test:** Score your mock and identify gaps",
                    f"- **Time:** {hrs} hrs\n"
                ]
            elif d == days - 1:
                lines += [
                    f"### Day {d} — Timed Practice & Exam Technique",
                    f"**Focus:** Speed, accuracy, and exam strategy",
                    f"- Practice questions under timed conditions",
                    f"- Review mark schemes and model answers",
                    f"- Identify question patterns and common traps",
                    f"- **Self-test:** Complete a mini-mock in half the exam time",
                    f"- **Time:** {hrs} hrs\n"
                ]
            else:
                idx = (d - 1) % topic_count
                t = topics[idx]
                lines += [
                    f"### Day {d} — Deep Dive: {t}",
                    f"**Focus:** {t}",
                    f"- Study core theory and worked examples for {t}",
                    f"- Solve practice problems at {('Easy' if d <= days // 3 else 'Medium' if d <= 2 * days // 3 else 'Hard')} difficulty",
                    f"- Update flashcards with new terms and formulas",
                    f"- **Self-test:** Quiz yourself on {t} without notes",
                    f"- **Time:** {hrs} hrs  (25-min Pomodoro cycles)\n"
                ]

        lines.append("---")
        lines.append("**Study Tips:**")
        lines.append("- Use spaced repetition — review previous days' material briefly each morning.")
        lines.append("- Prioritise active recall over passive re-reading.")
        lines.append("- Get 7–8 hours of sleep; consolidation happens during rest.")
        return "\n".join(lines)

    async def _arun(self, days: int = 7, topics: str = "General Topics", hours_per_day: float = 2.0) -> str:
        return self._run(days, topics, hours_per_day)
