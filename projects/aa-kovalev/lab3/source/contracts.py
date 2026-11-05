"""Типы на общей границе агента, инструментов и трассировки."""
from typing import Any, Literal, NotRequired, Protocol, TypedDict

ToolArguments = dict[str, Any]  # JSON, полученный от модели.
SearchHit = dict[str, Any]  # Формат выдачи RAG из Лабы 2.
ToolResult = dict[str, Any] | list[dict[str, Any]]
TurnStatus = Literal['success', 'error', 'timeout', 'max_steps']


class ToolCall(TypedDict):
    name: str
    arguments: ToolArguments
    result: NotRequired[ToolResult]
    error: NotRequired[str]


class AgentStats(TypedDict):
    status: TurnStatus
    input_tokens: int | None
    output_tokens: int | None
    steps: NotRequired[int]
    trimmed: NotRequired[bool | None]
    max_context: NotRequired[int | None]
    context_tokens: NotRequired[int | None]
    trimmed_before_answer: NotRequired[bool]


class TurnMetrics(TypedDict):
    latency_seconds: float
    tool_calls: int
    tool_errors: int
    cost_usd: float | None


class TurnResult(AgentStats, TurnMetrics):
    output: str
    calls: list[ToolCall]
    retrievals: list[list[SearchHit]]
    trace_id: str
    trace_url: str | None
    session_id: str


class DialogueAgent(Protocol):
    def answer(self, question: str) -> tuple[str, AgentStats]: ...
