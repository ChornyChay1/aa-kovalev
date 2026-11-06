"""Нативная генерация smolagents с мостом для вложенных Pydantic-моделей."""
from pydantic import validate_call
from smolagents import Tool, tool

from constants import MANAGE_QUIZ
from models.schemas import Question
from tools.book import BookTools
from tools.definitions import build_tool_definitions


def build_tools(tools: BookTools) -> list[Tool]:
    result = []
    for handler in build_tool_definitions(tools):
        item = tool(handler)
        if item.name == MANAGE_QUIZ:
            # smolagents 1.22.0 распознаёт BaseModel лишь как object.
            # Поля и ограничения берём из модели, не дублируем вручную.
            item.inputs['question'].update(Question.model_json_schema())
            item.inputs['questions']['items'] = Question.model_json_schema()

            # Нативный @tool не преобразует входные dict в BaseModel.
            # Валидация сохраняет сигнатуру и даёт общим функциям Question.
            item.forward = validate_call(handler)
        result.append(item)
    return result
