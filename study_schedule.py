"""
StudyScheduleGeneratorTool - Creates daily study timetables based on available hours, topics, and exam date.
"""

import os
from typing import Type, Optional
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.logger import logger

class StudyScheduleInput(BaseModel):
    hours_per_day: float = Field(3.0, description="Available study hours per day.")
    subjects: str = Field("General Subjects", description="Subjects or topics to schedule.")
    exam_date: str = Field("In 2 weeks", description="Target exam date or deadline.")

class StudyScheduleGeneratorTool(BaseTool):
    name: str = "study_schedule_generator_tool"
    description: str = "Generates customized daily timetables and hour allocations based on exam date and available hours."
    args_schema: Type[BaseModel] = StudyScheduleInput

    def _run(self, hours_per_day: float = 3.0, subjects: str = "General Subjects", exam_date: str = "In 2 weeks") -> str:
        """Generates daily schedule."""
        logger.info(f"StudyScheduleGeneratorTool invoked for {hours_per_day}h/day until {exam_date}")
        
        hrs = max(1.0, min(hours_per_day, 14.0))
        subject_list = [s.strip() for s in subjects.split(",") if s.strip()] or ["Core Subject"]

        if os.getenv("DISABLE_LLM_API") != "1":
            try:
                from config import GEMINI_API_KEY, OPENAI_API_KEY, DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL
                import random
                seed_val = random.randint(1000, 999999)
                prompt = (
                    f"Create a balanced, fresh, and unique daily study schedule template for a student with {hrs} hours per day available.\n"
                    f"Subjects: {', '.join(subject_list)}\n"
                    f"Exam Target: {exam_date}\n"
                    f"Randomization Seed: {seed_val}\n\n"
                    "Include breakdown of time slots (e.g., Morning/Evening blocks), break intervals, and study methods."
                )
                
                res_text = ""
                if GEMINI_API_KEY and GEMINI_API_KEY.strip():
                    from langchain_google_genai import ChatGoogleGenerativeAI
                    llm = ChatGoogleGenerativeAI(model=DEFAULT_GEMINI_MODEL, google_api_key=GEMINI_API_KEY, max_retries=0, timeout=10, temperature=0.8)
                    res_text = llm.invoke(prompt).content
                elif OPENAI_API_KEY and OPENAI_API_KEY.strip():
                    from langchain_openai import ChatOpenAI
                    llm = ChatOpenAI(model=DEFAULT_OPENAI_MODEL, api_key=OPENAI_API_KEY, max_retries=0, timeout=10, temperature=0.8)
                    res_text = llm.invoke(prompt).content

                if res_text:
                    return res_text
            except Exception as e:
                logger.warning(f"LLM study schedule generation failed, using fallback template: {e}")

        return self._generate_fallback_schedule(hrs, subject_list, exam_date)

    def _generate_fallback_schedule(self, hrs: float, subjects: list, exam_date: str) -> str:
        slot_hours = round(hrs / max(len(subjects), 1), 1)
        
        out = [f"### ⏱️ Custom Daily Study Timetable ({hrs} Hours/Day)\n"]
        out.append(f"**Target Exam Deadline:** {exam_date}")
        out.append(f"**Subjects:** {', '.join(subjects)}\n")
        
        out.append("#### 📋 Recommended Time Slot Allocation:")
        for idx, subj in enumerate(subjects, start=1):
            out.append(f"- **Block {idx} ({slot_hours} hrs):** {subj}")
            out.append(f"  * 25 mins active reading/note review")
            out.append(f"  * 5 mins break")
            out.append(f"  * 25 mins problem solving & flashcards")
            out.append(f"  * 5 mins progress logging\n")

        out.append("#### 🛡️ Daily Best Practices:")
        out.append("- Start daily sessions with the most challenging subject.")
        out.append("- Hydrate and step away from screens during 5-minute breaks.")
        out.append("- Spend the final 15 minutes of your daily quota performing active recall self-tests.")
        return "\n".join(out)

    async def _arun(self, hours_per_day: float = 3.0, subjects: str = "General Subjects", exam_date: str = "In 2 weeks") -> str:
        return self._run(hours_per_day, subjects, exam_date)
