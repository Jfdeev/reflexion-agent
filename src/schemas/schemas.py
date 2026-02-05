from typing import List
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser


class Reflection(BaseModel):
    missing: str = Field(description="Critique of what is missing")
    superfluous: str = Field(description="Critique of what is superfluous")


class AnswerQuestion(BaseModel):
    """Answer the qusetion"""
    answer: str = Field(description="~250 word detailed answer to the question")
    reflection: Reflection = Field(description="your reflection on the initial answer")
    search_queries: List[str] = Field(
        description="1-3 search queries for researching improvements to address the critique of your current answer"
    )

class RevisedAnswer(BaseModel):
    """Revise your original answer to the question"""
    references: List[str] = Field(
        description="Citations motivating your updated answer"
    )