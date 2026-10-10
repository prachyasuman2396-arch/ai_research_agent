from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str
    url: str


class ModelComparison(BaseModel):
    model: str
    advantages: list[str]
    disadvantages: list[str]


class ResearchReport(BaseModel):
    research_question: str
    summary: str
    comparison: list[ModelComparison]
    recommendation: str
    sources: list[Source]