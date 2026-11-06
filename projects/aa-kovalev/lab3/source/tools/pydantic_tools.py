"""PydanticAI строит схемы и валидацию из типов и docstrings общих функций."""
from pydantic_ai import Tool

from constants import MANAGE_QUIZ
from tools.book import BookTools
from tools.definitions import build_tool_definitions


def build_tools(tools: BookTools) -> list[Tool]:
    return [
        Tool(handler, docstring_format='google', require_parameter_descriptions=True,
             sequential=handler.__name__ == MANAGE_QUIZ)
        for handler in build_tool_definitions(tools)
    ]
