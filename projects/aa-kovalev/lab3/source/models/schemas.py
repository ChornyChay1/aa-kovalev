"""Валидация данных квиза, API и результатов прогона."""
from typing import Self

from pydantic import BaseModel, Field, model_validator

from constants import MAX_OPTIONS
from models.domain import SearchHit, ToolArguments, ToolResult, TurnStatus


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    options: list[str] = Field(min_length=2, max_length=MAX_OPTIONS)
    correct: int = Field(
        ge=1, le=MAX_OPTIONS,
        description='Номер правильного варианта в options: 1 — первый, 2 — второй и так далее. Нуля быть не может.',
    )

    @model_validator(mode='after')
    def validate_options(self) -> Self:
        if self.correct > len(self.options):
            raise ValueError('Номер правильного варианта вне диапазона')
        if any(not option.strip() or len(option) > 250 for option in self.options):
            raise ValueError('Варианты должны содержать от 1 до 250 символов')
        return self


class ToolCall(BaseModel):
    name: str
    arguments: ToolArguments
    result: ToolResult | None = None
    error: str | None = None


class AgentStats(BaseModel):
    status: TurnStatus
    input_tokens: int | None
    output_tokens: int | None
    steps: int | None = None
    trimmed: bool | None = None
    max_context: int | None = None
    context_tokens: int | None = None
    trimmed_before_answer: bool | None = None


class TurnMetrics(BaseModel):
    latency_seconds: float
    tool_calls: int
    tool_errors: int
    cost_usd: float | None


class TurnResult(AgentStats, TurnMetrics):
    output: str
    calls: list[ToolCall]
    retrievals: list[list[SearchHit]]
    trace_id: str
    observation_id: str
    trace_url: str | None
    session_id: str


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=6000)
    session: str | None = None


class BookPage(BaseModel):
    page: int
    url: str


class ChatResponse(BaseModel):
    session: str
    answer: str
    status: str
    trace_url: str | None
    book_pages: list[BookPage]


class ChatTurn(BaseModel):
    message: str
    result: ChatResponse


class Transcript(BaseModel):
    session: str
    title: str
    turns: list[ChatTurn] = Field(default_factory=list)


class Attempt(BaseModel):
    answers: list[int]
