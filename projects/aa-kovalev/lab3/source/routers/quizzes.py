"""Страница квиза и проверка ответов."""
from html import escape
from string import Template

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from models.domain import Quiz
from models.schemas import Attempt
from services.quizzes import get_quiz
from settings import ROOT

router = APIRouter()


def quiz_or_404(quiz_id: str) -> Quiz:
    try:
        return get_quiz(quiz_id)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@router.get('/quiz/{quiz_id}', response_class=HTMLResponse)
def quiz_page(quiz_id: str) -> str:
    quiz = quiz_or_404(quiz_id)
    fields = []
    for i, question in enumerate(quiz['questions']):
        options = ''.join(
            f'<label><input type="radio" name="q{i}" value="{n}" required> {escape(option)}</label>'
            for n, option in enumerate(question['options'], 1))
        fields.append(f'<fieldset><legend>{i+1}. {escape(question["text"])}</legend>{options}</fieldset>')
    template = Template((ROOT / 'web/quiz.html').read_text(encoding='utf-8'))
    return template.substitute(title=escape(quiz['title']), fields=''.join(fields))


@router.post('/quiz/{quiz_id}/grade')
def grade(quiz_id: str, attempt: Attempt) -> dict[str, int]:
    questions = quiz_or_404(quiz_id)['questions']
    if len(attempt.answers) != len(questions) or any(
        not 1 <= a <= len(q['options']) for a, q in zip(attempt.answers, questions)
    ):
        raise HTTPException(422, 'Укажите по одному допустимому ответу на каждый вопрос.')
    return {'score': sum(a == q['correct'] for a, q in zip(attempt.answers, questions)),
            'total': len(questions)}
