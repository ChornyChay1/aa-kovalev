"""Адаптеры общих инструментов для PydanticAI."""
from pydantic_ai import ModelRetry, Tool

from constants import CREATE, GET_QUIZ_INFO, LIST_QUIZES, MANAGE_QUIZ, MAX_QUESTIONS, MAX_QUIZ_TITLE, SEARCH_BOOK, QuizAction
from models.domain import ToolResult
from models.schemas import Question
from settings import SCHEMAS
from tools.book import BookTools

DESCRIPTIONS = {item['function']['name']: item['function']['description'] for item in SCHEMAS}


def build_tools(tools: BookTools) -> list[Tool]:
    def search_book(query: str) -> ToolResult:
        return tools.call(SEARCH_BOOK, {'query': query})

    def manage_quiz(action: QuizAction,
                    quiz_id: str | None = None, title: str | None = None,
                    questions: list[Question] | None = None,
                    question_number: int | None = None, question: Question | None = None,
                    option: str | None = None, option_number: int | None = None) -> ToolResult:
        if action == CREATE and (not title or not title.strip() or len(title) > MAX_QUIZ_TITLE
                                   or not questions or len(questions) > MAX_QUESTIONS):
            raise ModelRetry(
                'Для create придумай непустой title до 150 символов и передай questions '
                'с 1–10 полностью заполненными вопросами. Соблюдай количество из запроса.'
            )
        args = dict(action=action, quiz_id=quiz_id, title=title, question_number=question_number,
                    questions=[q.model_dump() for q in questions] if questions else None,
                    question=question.model_dump() if question else None, option=option, option_number=option_number)
        return tools.call(MANAGE_QUIZ, args)

    def list_quizes() -> ToolResult:
        return tools.call(LIST_QUIZES, {})

    def get_quiz_info(quiz_id: str, question_number: int | None = None) -> ToolResult:
        return tools.call(GET_QUIZ_INFO, dict(quiz_id=quiz_id, question_number=question_number))

    handlers = (search_book, manage_quiz, list_quizes, get_quiz_info)
    return [Tool(handler, description=DESCRIPTIONS[handler.__name__]) for handler in handlers]
