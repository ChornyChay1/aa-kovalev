"""Типы данных, общие для хранилища и агентов."""
from typing import Any, Literal, TYPE_CHECKING, Protocol, TypedDict

if TYPE_CHECKING:
    from models.schemas import AgentStats

ToolArguments = dict[str, Any]  # JSON, полученный от модели.
SearchHit = dict[str, Any]  # Формат выдачи RAG из Лабы 2.
ToolResult = dict[str, Any] | list[dict[str, Any]]
TurnStatus = Literal['success', 'error', 'timeout', 'max_steps']


class QuizSummary(TypedDict):
    quiz_id: str
    title: str
    url: str


class QuizQuestion(TypedDict):
    text: str
    options: list[str]
    correct: int


class Quiz(QuizSummary):
    questions: list[QuizQuestion]


class DialogueAgent(Protocol):
    prompt_version: str

    def answer(self, question: str) -> tuple[str, "AgentStats"]: ...
