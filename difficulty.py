"""
DifficultyLevelTool - Configures and applies difficulty levels across study content.
"""

from typing import Type, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from utils.validators import validate_difficulty
from utils.logger import logger

class DifficultyInput(BaseModel):
    level: str = Field("Medium", description="Desired difficulty level: Easy, Medium, or Hard.")
    target_feature: str = Field("Quiz", description="Feature to apply difficulty to: Quiz, Flashcards, Explanations, or Summaries.")

class DifficultyLevelTool(BaseTool):
    name: str = "difficulty_level_tool"
    description: str = "Adjusts difficulty parameters (Easy, Medium, Hard) for quizzes, flashcards, and explanations."
    args_schema: Type[BaseModel] = DifficultyInput

    def _run(self, level: str = "Medium", target_feature: str = "Quiz") -> str:
        """Applies difficulty level parameters."""
        logger.info(f"DifficultyLevelTool configuring level={level} for feature={target_feature}")
        
        valid_level = validate_difficulty(level)
        
        settings = {
            "Easy": {
                "question_depth": "Basic recall, definitions, direct facts",
                "explanation_tone": "Simple, beginner-friendly with analogies",
                "time_per_question": "1-2 minutes"
            },
            "Medium": {
                "question_depth": "Application of concepts, moderate analytical reasoning",
                "explanation_tone": "Standard academic level with practical examples",
                "time_per_question": "2-3 minutes"
            },
            "Hard": {
                "question_depth": "Complex multi-step problem solving, critical evaluation, edge cases",
                "explanation_tone": "Rigorous, formal, in-depth technical breakdown",
                "time_per_question": "4-5 minutes"
            }
        }

        conf = settings.get(valid_level, settings["Medium"])
        
        return (
            f"### ⚙️ Difficulty Settings Applied\n\n"
            f"- **Target Feature:** {target_feature}\n"
            f"- **Selected Level:** `{valid_level}`\n"
            f"- **Cognitive Focus:** {conf['question_depth']}\n"
            f"- **Explanation Style:** {conf['explanation_tone']}\n"
            f"- **Pacing Recommendation:** {conf['time_per_question']} per task item"
        )

    async def _arun(self, level: str = "Medium", target_feature: str = "Quiz") -> str:
        return self._run(level, target_feature)
