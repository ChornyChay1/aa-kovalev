"""Хранилище квизов, общее для веб-формы и инструментов агента."""
import json
import sqlite3
from contextlib import closing
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from settings import ROOT


class Question(BaseModel):
    text: str = Field(min_length=1, max_length=500)
    options: list[str] = Field(min_length=2, max_length=5)
    correct: int = Field(ge=1, le=5)

    @model_validator(mode='after')
    def validate_options(self):
        if self.correct > len(self.options):
            raise ValueError('Номер правильного варианта вне диапазона')
        if any(not option.strip() or len(option) > 250 for option in self.options):
            raise ValueError('Варианты должны содержать от 1 до 250 символов')
        return self


def connect():
    db = sqlite3.connect(ROOT / 'quizzes.sqlite3')
    db.row_factory = sqlite3.Row
    db.execute('CREATE TABLE IF NOT EXISTS quizzes (id TEXT PRIMARY KEY, session TEXT, title TEXT, questions TEXT)')
    return db


def public(row):
    return {'quiz_id': row['id'], 'title': row['title'], 'url': '/quiz/' + row['id']}


def get_quiz(quiz_id, session=None):
    with closing(connect()) as db:
        row = db.execute('SELECT * FROM quizzes WHERE id = ?', (quiz_id,)).fetchone()
    if row is None or (session is not None and row['session'] != session):
        raise ValueError('Квиз не найден')
    return {**public(row), 'questions': json.loads(row['questions'])}


def get_quiz_info(session, quiz_id):
    quiz = get_quiz(quiz_id, session)
    for question in quiz['questions']:
        del question['correct']
    return quiz


def list_quizes(session):
    with closing(connect()) as db:
        rows = db.execute('SELECT id, title FROM quizzes WHERE session = ? ORDER BY rowid', (session,))
        return [public(row) for row in rows]


def manage_quiz(session, action, quiz_id=None, title=None, questions=None,
                question_number=None, question=None):
    with closing(connect()) as db, db:
        if action == 'create':
            if not title or not title.strip() or len(title) > 150 or not questions or len(questions) > 10:
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
            raise ValueError('Квиз не найден в этой сессии')
        if action == 'delete':
            db.execute('DELETE FROM quizzes WHERE id = ?', (quiz_id,))
            return {'quiz_id': quiz_id, 'deleted': True}
        if action != 'update':
            raise ValueError('Допустимые действия: create, update, delete')
        data = json.loads(row['questions'])
        if type(question_number) is not int or not 1 <= question_number <= len(data):
            raise ValueError('Номер вопроса вне диапазона')
        data[question_number - 1] = Question.model_validate(question).model_dump()
        db.execute('UPDATE quizzes SET questions = ? WHERE id = ?', (json.dumps(data), quiz_id))
        return {**public(row), 'updated_question': question_number}
