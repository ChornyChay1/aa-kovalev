"""Адаптеры общих инструментов для PydanticAI."""
from pydantic_ai import ModelRetry, Tool

from constants import CREATE, GET_QUIZ_INFO, LIST_QUIZES, MANAGE_QUIZ, MAX_QUESTIONS, MAX_QUIZ_TITLE, SEARCH_BOOK, QuizAction
from models.domain import ToolResult
from models.schemas import Question
from services.quizzes import get_quiz
from settings import SCHEMAS
from tools.book import BookTools

DESCRIPTIONS = {item['function']['name']: item['function']['description'] for item in SCHEMAS}


def build_tools(tools: BookTools) -> list[Tool]:
    def search_book(query: str) -> ToolResult:
        return tools.call(SEARCH_BOOK, {'query': query})

    def select_quiz_for_edit(quiz_id: str) -> ToolResult:
        if tools.edit_quiz_id is not None and tools.edit_quiz_id != quiz_id:
            raise ModelRetry('На этот ход уже выбран другой квиз. Не меняй цель. Уточни запрос у пользователя.')
        if tools.history_trimmed and tools.edit_quiz_id is None and quiz_id not in tools.explicit_quiz_ids:
            raise ModelRetry('История сократилась до выбора квиза. Попроси пользователя указать полный ID или ссылку.')
        try:
            quiz = get_quiz(quiz_id, tools.session)
        except ValueError as error:
            raise ModelRetry(str(error)) from error
        tools.edit_quiz_id = quiz_id
        return {'quiz_id': quiz_id, 'title': quiz['title']}

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
        if action != CREATE:
            if tools.edit_quiz_id is None:
                raise ModelRetry('Сначала выбери целевой квиз через select_quiz_for_edit.')
            if quiz_id != tools.edit_quiz_id:
                raise ModelRetry(f'Целевой квиз на этот ход: {tools.edit_quiz_id}. Передай этот quiz_id.')
        args = dict(action=action, quiz_id=quiz_id, title=title, question_number=question_number,
                    questions=[q.model_dump() for q in questions] if questions else None,
                    question=question.model_dump() if question else None, option=option, option_number=option_number)
        return tools.call(MANAGE_QUIZ, args)

    def list_quizes() -> ToolResult:
        return tools.call(LIST_QUIZES, {})

    def get_quiz_info(quiz_id: str, question_number: int | None = None) -> ToolResult:
        return tools.call(GET_QUIZ_INFO, dict(quiz_id=quiz_id, question_number=question_number))

    handlers = (search_book, manage_quiz, list_quizes, get_quiz_info)
    descriptions = {**DESCRIPTIONS, MANAGE_QUIZ: DESCRIPTIONS[MANAGE_QUIZ] +
                    ' Перед update или delete сначала вызови select_quiz_for_edit.'}
    return [Tool(handler, description=descriptions[handler.__name__]) for handler in handlers] + [
        Tool(select_quiz_for_edit, description=(
            'Перед правкой или удалением закрепи полный ID квиза, который указал пользователь. '
            'Если запрос ссылается на историю, выбери квиз по этой истории. '
            'При неоднозначности попроси пользователя уточнить цель.'
        )),
    ]
