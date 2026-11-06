"""Общие функции инструментов: типы и docstrings вместо ручных JSON-схем."""
from collections.abc import Callable

from constants import GET_QUIZ_INFO, LIST_QUIZES, MANAGE_QUIZ, SEARCH_BOOK, QuizAction
from models.domain import ToolResult
from models.schemas import Question
from tools.book import BookTools


def build_tool_definitions(tools: BookTools) -> list[Callable[..., ToolResult]]:
    def search_book(query: str) -> ToolResult:
        """Ищет по запросу в учебнике «Укус Питона».

        Возвращает фрагменты текста и номера PDF-страниц. Если ответ опирается
        на учебник, укажи страницу из поля page подтверждающего фрагмента.

        Args:
            query: Поисковый запрос по теме исходного вопроса пользователя;
                включи конкретные термины, в том числе при уточнении в диалоге.
        """
        return tools.call(SEARCH_BOOK, {'query': query})

    def manage_quiz(action: QuizAction,
                    quiz_id: str | None = None, title: str | None = None,
                    questions: list[Question] | None = None,
                    question_number: int | None = None, question: Question | None = None,
                    option: str | None = None, option_number: int | None = None) -> ToolResult:
        """Создаёт, изменяет или удаляет квиз.

        create принимает название и готовые вопросы, возвращает ID и ссылку.
        append_questions добавляет вопросы в конец без question_number.
        update заменяет вопрос через question или изменяет один вариант через
        option и option_number; option без номера добавляет вариант в конец.
        delete_question удаляет один вопрос, delete — весь квиз только по явной
        просьбе пользователя. Номера вопросов и вариантов начинаются с 1.

        Args:
            action: Операция: create, update, append_questions, delete_question или delete.
            quiz_id: Полный ID из результата предыдущего вызова или списка квизов,
                не порядковый номер.
            title: Название нового квиза для create, от 1 до 150 символов.
            questions: Готовые вопросы для create или append_questions.
            question_number: Номер существующего вопроса с 1; обязателен для update
                и delete_question.
            question: Полный заменяющий вопрос для update: text, options и correct.
                Остальные вопросы сохраняются.
            option: Текст варианта для update; передавай вместо question.
                Без option_number добавляет вариант в конец, сохраняя правильный ответ.
            option_number: Позиция существующего варианта с 1 при замене.
        """
        args = dict(
            action=action, quiz_id=quiz_id, title=title, question_number=question_number,
            questions=[q.model_dump() for q in questions] if questions is not None else None,
            question=question.model_dump() if question is not None else None,
            option=option, option_number=option_number,
        )
        return tools.call(MANAGE_QUIZ, args)

    def list_quizes() -> ToolResult:
        """Возвращает список квизов доступной коллекции с названиями, ID и ссылками.

        Позволяет найти ID квиза по названию.
        """
        return tools.call(LIST_QUIZES, {})

    def get_quiz_info(quiz_id: str, question_number: int | None = None) -> ToolResult:
        """Возвращает вопросы квиза и варианты без ключей ответов.

        Args:
            quiz_id: Полный ID из результата предыдущего вызова или списка квизов,
                не порядковый номер.
            question_number: Номер вопроса с 1; без номера возвращаются все вопросы.
        """
        return tools.call(GET_QUIZ_INFO, dict(quiz_id=quiz_id, question_number=question_number))

    return [search_book, manage_quiz, list_quizes, get_quiz_info]
