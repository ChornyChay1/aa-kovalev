"""Local quiz forms and chat API."""
from contextlib import asynccontextmanager
from html import escape
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from langfuse import get_client
from pydantic import BaseModel, Field

from agent import Agent
from book_tools import BookTools
from quizzes import get_quiz
from run import turn
from settings import ROOT

sessions = {}
sessions_lock = Lock()


@asynccontextmanager
async def lifespan(app):
    yield
    for agent, tools, _ in sessions.values():
        agent.close()
        tools.close()
    sessions.clear()
    get_client().flush()


app = FastAPI(lifespan=lifespan)


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=6000)
    session: str | None = None


class Attempt(BaseModel):
    answers: list[int]


@app.get('/', response_class=HTMLResponse)
def home():
    return (ROOT / 'web/index.html').read_text(encoding='utf-8')


@app.post('/api/chat')
def chat(request: Chat):
    with sessions_lock:
        session = request.session or uuid4().hex
        if request.session and session not in sessions:
            raise HTTPException(404, 'Сессия завершилась. Начните новый диалог.')
        if session not in sessions:
            try:
                tools = BookTools()
            except FileNotFoundError as error:
                raise HTTPException(503, 'Подготовьте активный индекс Лабы 2.') from error
            sessions[session] = Agent(tools), tools, Lock()
        agent, tools, lock = sessions[session]
    with lock:
        row = turn(agent, tools, request.message, session, 'custom', 'web')
        return {'session': session, 'answer': row['output'], 'status': row['status'],
                'trace_url': row['trace_url']}


def quiz_or_404(quiz_id):
    try:
        return get_quiz(quiz_id)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@app.get('/quiz/{quiz_id}', response_class=HTMLResponse)
def quiz_page(quiz_id: str):
    quiz = quiz_or_404(quiz_id)
    fields = []
    for i, question in enumerate(quiz['questions']):
        options = ''.join(
            f'<label><input type="radio" name="q{i}" value="{n}" required> {escape(option)}</label>'
            for n, option in enumerate(question['options'], 1))
        fields.append(f'<fieldset><legend>{i+1}. {escape(question["text"])}</legend>{options}</fieldset>')
    title = escape(quiz['title'])
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1">
    <title>{title}</title>
    <style>body{{font:17px system-ui;max-width:720px;margin:2rem auto;padding:0 1rem}}
    label{{display:block;padding:.45rem}}fieldset{{margin:1rem 0}}</style>
    <a href="/">← Чат</a><h1>{title}</h1><form>{''.join(fields)}
    <button>Проверить</button></form><p role="status"></p>
    <script>
    document.querySelector('form').onsubmit = async (event) => {{
        event.preventDefault();
        const answers = [...new FormData(event.target).values()].map(Number);
        const status = document.querySelector('[role=status]');
        try {{
            const response = await fetch(location.pathname + '/grade', {{
                method: 'POST', headers: {{'Content-Type': 'application/json'}},
                body: JSON.stringify({{answers}})
            }});
            const result = await response.json();
            status.textContent = response.ok ? `Результат: ${{result.score}} из ${{result.total}}` : result.detail;
        }} catch {{ status.textContent = 'Сервис недоступен'; }}
    }};
    </script></html>'''


@app.post('/quiz/{quiz_id}/grade')
def grade(quiz_id: str, attempt: Attempt):
    questions = quiz_or_404(quiz_id)['questions']
    if len(attempt.answers) != len(questions) or any(
        not 1 <= a <= len(q['options']) for a, q in zip(attempt.answers, questions)
    ):
        raise HTTPException(422, 'Укажите по одному допустимому ответу на каждый вопрос.')
    return {'score': sum(a == q['correct'] for a, q in zip(attempt.answers, questions)),
            'total': len(questions)}
