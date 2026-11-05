"""Адаптеры общих инструментов для smolagents."""
from typing import Any

from smolagents import Tool

from constants import GET_QUIZ_INFO, LIST_QUIZES, MANAGE_QUIZ, SEARCH_BOOK
from models.domain import ToolResult
from settings import SCHEMAS
from tools.book import BookTools


class BookTool(Tool):
    def __init__(self, schema: dict[str, Any], tools: BookTools) -> None:
        spec = schema['function']
        self.name, self.description = spec['name'], spec['description']
        self.inputs = spec['parameters']['properties']
        self.output_type = 'any'
        self.tools = tools
        super().__init__()


class SearchBookTool(BookTool):
    def forward(self, query: str) -> ToolResult:
        return self.tools.call(self.name, {'query': query})


class ManageQuizTool(BookTool):
    def forward(self, action: str, quiz_id: str | None = None,
                title: str | None = None, questions: list[dict[str, Any]] | None = None,
                question_number: int | None = None, question: dict[str, Any] | None = None,
                option: str | None = None, option_number: int | None = None) -> ToolResult:
        return self.tools.call(self.name, dict(
            action=action, quiz_id=quiz_id, title=title, questions=questions,
            question_number=question_number, question=question, option=option, option_number=option_number))


class ListQuizesTool(BookTool):
    def forward(self) -> ToolResult:
        return self.tools.call(self.name, {})


class GetQuizInfoTool(BookTool):
    def forward(self, quiz_id: str, question_number: int | None = None) -> ToolResult:
        return self.tools.call(self.name, dict(quiz_id=quiz_id, question_number=question_number))


TOOL_CLASSES = {
    SEARCH_BOOK: SearchBookTool,
    MANAGE_QUIZ: ManageQuizTool,
    LIST_QUIZES: ListQuizesTool,
    GET_QUIZ_INFO: GetQuizInfoTool,
}


def build_tools(tools: BookTools) -> list[Tool]:
    return [TOOL_CLASSES[schema['function']['name']](schema, tools) for schema in SCHEMAS]
