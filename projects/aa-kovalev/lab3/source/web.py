"""Local quiz forms and chat API."""
from collections.abc import AsyncIterator, Iterator
from contextlib import ExitStack, asynccontextmanager
from dataclasses import dataclass
from html import escape
from string import Template
from threading import Lock
from uuid import uuid4

from dishka import Container, Provider, Scope, make_container, provide
from dishka.integrations.fastapi import FastapiProvider, FromDishka, inject_sync, setup_dishka
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from langfuse import get_client
from pydantic import BaseModel, Field

from agent import Agent
from book_tools import BookTools
from dependencies import LabProvider
from quizzes import Quiz, get_quiz
from turns import turn
from settings import ROOT


@dataclass
class Dialogue:
    agent: Agent
    tools: BookTools
    lock: Lock


class ChatSessions:
    def __init__(self, container: Container, scopes: ExitStack) -> None:
        self.container = container
        self.scopes = scopes
        self.dialogues: dict[str, Dialogue] = {}
        self.lock = Lock()

    def get(self, session: str | None) -> tuple[str, Dialogue]:
        with self.lock:
            if session is not None:
                if session not in self.dialogues:
                    raise HTTPException(404, 'Сессия завершилась. Начните новый диалог.')
                return session, self.dialogues[session]
            session = uuid4().hex
            # Register a scope only after all resources were created successfully.
            with ExitStack() as opening:
                scope = opening.enter_context(self.container(scope=Scope.REQUEST))
                try:
                    dialogue = Dialogue(scope.get(Agent), scope.get(BookTools), Lock())
                except FileNotFoundError as error:
                    raise HTTPException(503, 'Подготовьте активный индекс Лабы 2.') from error
                self.scopes.enter_context(opening.pop_all())
            self.dialogues[session] = dialogue
            return session, dialogue


class WebProvider(Provider):
    @provide(scope=Scope.APP)
    def chat_sessions(self, container: Container) -> Iterator[ChatSessions]:
        with ExitStack() as scopes:
            yield ChatSessions(container, scopes)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    try:
        yield
    finally:
        try:
            # Agent.close() runs its own asyncio.Runner.
            await run_in_threadpool(app.state.dishka_container.close)
        finally:
            get_client().flush()


app = FastAPI(lifespan=lifespan)
setup_dishka(container=make_container(LabProvider(), WebProvider(), FastapiProvider(), lock_factory=Lock), app=app)


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=6000)
    session: str | None = None


class ChatResponse(BaseModel):
    session: str
    answer: str
    status: str
    trace_url: str | None


class Attempt(BaseModel):
    answers: list[int]


@app.get('/', response_class=HTMLResponse)
def home() -> str:
    return (ROOT / 'web/index.html').read_text(encoding='utf-8')


@app.post('/api/chat')
@inject_sync
def chat(request: Chat, sessions: FromDishka[ChatSessions]) -> ChatResponse:
    session, dialogue = sessions.get(request.session)
    with dialogue.lock:
        row = turn(dialogue.agent, dialogue.tools, request.message, session, 'custom', 'web')
    return ChatResponse(session=session, answer=row['output'], status=row['status'],
                        trace_url=row['trace_url'])


def quiz_or_404(quiz_id: str) -> Quiz:
    try:
        return get_quiz(quiz_id)
    except ValueError as error:
        raise HTTPException(404, str(error)) from error


@app.get('/quiz/{quiz_id}', response_class=HTMLResponse)
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


@app.post('/quiz/{quiz_id}/grade')
def grade(quiz_id: str, attempt: Attempt) -> dict[str, int]:
    questions = quiz_or_404(quiz_id)['questions']
    if len(attempt.answers) != len(questions) or any(
        not 1 <= a <= len(q['options']) for a, q in zip(attempt.answers, questions)
    ):
        raise HTTPException(422, 'Укажите по одному допустимому ответу на каждый вопрос.')
    return {'score': sum(a == q['correct'] for a, q in zip(attempt.answers, questions)),
            'total': len(questions)}
