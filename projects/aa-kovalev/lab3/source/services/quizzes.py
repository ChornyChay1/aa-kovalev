"""Хранилище квизов в SQLite."""
import json
import sqlite3
from contextlib import closing
from typing import Any
from uuid import uuid4

from constants import CREATE, DELETE, MAX_OPTIONS, MAX_QUESTIONS, MAX_QUIZ_TITLE, UPDATE, WEB_QUIZ_COLLECTION
from models.domain import Quiz, QuizSummary
from models.schemas import Question
from settings import ROOT


def connect() -> sqlite3.Connection:
    db = sqlite3.connect(ROOT / 'quizzes.sqlite3')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS quizzes (id TEXT PRIMARY KEY, session TEXT, title TEXT, questions TEXT)')
    return db


def migrate_web_quizzes() -> None:
    # Старые веб-диалоги использовали uuid4().hex. У CLI ID сессии содержит дефисы.
    with closing(connect()) as db, db:
        db.execute("""UPDATE quizzes SET session = ?
                      WHERE length(session) = 32 AND session NOT GLOB '*[^0-9a-f]*'""",
                   (WEB_QUIZ_COLLECTION,))


def public(row: sqlite3.Row) -> QuizSummary:
    return {'quiz_id': row['id'], 'title': row['title'], 'url': '/quiz/' + row['id']}


def get_quiz(quiz_id: str, session: str | None = None) -> Quiz:
    with closing(connect()) as db:
        row = db.execute('SELECT * FROM quizzes WHERE id = ?', (quiz_id,)).fetchone()
    if row is None or (session is not None and row['session'] != session):
        raise ValueError('Квиз не найден')
    return {**public(row), 'questions': json.loads(row['questions'])}


def get_quiz_info(session: str, quiz_id: str, question_number: int | None = None) -> dict[str, Any]:
    quiz = get_quiz(quiz_id, session)
    questions = quiz['questions']
    if question_number is not None:
        question = questions[question_index(question_number, len(questions))]
        return {'quiz_id': quiz_id, 'title': quiz['title'], 'question_number': question_number,
                'question': {'text': question['text'], 'options': question['options']}}
    return {
        **quiz,
        'questions': [{'text': q['text'], 'options': q['options']} for q in questions],
    }


def question_index(number: int | None, count: int) -> int:
    if type(number) is not int or not 1 <= number <= count:
        raise ValueError('Номер вопроса вне диапазона')
    return number - 1


def list_quizes(session: str) -> list[QuizSummary]:
    with closing(connect()) as db:
        rows = db.execute('SELECT id, title FROM quizzes WHERE session = ? ORDER BY rowid', (session,))
        return [public(row) for row in rows]


def manage_quiz(
    session: str, action: str, quiz_id: str | None = None, title: str | None = None,
    questions: list[dict[str, Any]] | None = None, question_number: int | None = None,
    question: dict[str, Any] | None = None, option: str | None = None,
    option_number: int | None = None,
) -> dict[str, Any]:
    with closing(connect()) as db, db:
        if action == CREATE:
            if not title or not title.strip() or len(title) > MAX_QUIZ_TITLE or not questions or len(questions) > MAX_QUESTIONS:
                raise ValueError('Укажите название и от 1 до 10 вопросов')
            quiz_id = uuid4().hex
            data = [Question.model_validate(q).model_dump() for q in questions]
            db.execute('INSERT INTO quizzes VALUES (?, ?, ?, ?)',
                       (quiz_id, session, title, json.dumps(data)))
            return {'quiz_id': quiz_id, 'title': title, 'url': '/quiz/' + quiz_id}
        if not quiz_id:
            raise ValueError('Для правки или удаления нужен quiz_id')
        db.execute('BEGIN IMMEDIATE')
        row = db.execute('SELECT * FROM quizzes WHERE id = ? AND session = ?', (quiz_id, session)).fetchone()
        if row is None:
            raise ValueError('Квиз не найден в доступной коллекции')
        if action == DELETE:
            db.execute('DELETE FROM quizzes WHERE id = ?', (quiz_id,))
            return {'quiz_id': quiz_id, 'deleted': True}
        if action != UPDATE:
            raise ValueError('Допустимые действия: create, update, delete')
        data = json.loads(row['questions'])
        index = question_index(question_number, len(data))
        if option_number is not None and option is None:
            raise ValueError('Укажите текст варианта')
        changed_option = None
        if option is not None:
            if question is not None:
                raise ValueError('Передайте либо вариант, либо вопрос целиком')
            question = data[index]
            options = question['options']
            if option_number is None and len(options) >= MAX_OPTIONS:
                raise ValueError('В вопросе уже пять вариантов')
            if option_number is not None and (type(option_number) is not int or not 1 <= option_number <= len(options)):
                raise ValueError('Номер варианта вне диапазона')
            position = len(options) if option_number is None else option_number - 1
            other_options = options[:position] + options[position + 1:]
            if option.strip().casefold() in {text.strip().casefold() for text in other_options}:
                raise ValueError('Такой вариант уже есть')
            options[position:position + 1] = [option]
            changed_option = {'option_number': position + 1, 'option': option}
        data[index] = Question.model_validate(question).model_dump()
        db.execute('UPDATE quizzes SET questions = ? WHERE id = ?', (json.dumps(data), quiz_id))
        return {**public(row), 'updated_question': question_number, **(changed_option or {})}
