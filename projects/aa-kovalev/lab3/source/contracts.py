"""Типы на общей границе агента, инструментов и трассировки."""
from typing import Any, Literal, Protocol

from pydantic import BaseModel

ToolArguments = dict[str, Any]  # JSON, полученный от модели.
SearchHit = dict[str, Any]  # Формат выдачи RAG из Лабы 2.
ToolResult = dict[str, Any] | list[dict[str, Any]]
TurnStatus = Literal['success', 'error', 'timeout', 'max_steps']


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


class DialogueAgent(Protocol):
    def answer(self, question: str) -> tuple[str, AgentStats]: ...
