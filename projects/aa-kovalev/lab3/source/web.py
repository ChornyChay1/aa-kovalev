"""Local quiz forms and chat API."""
from collections.abc import AsyncIterator
from contextlib import ExitStack, asynccontextmanager, closing
from dataclasses import dataclass
from html import escape
from string import Template
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from starlette.concurrency import run_in_threadpool
from langfuse import get_client
from pydantic import BaseModel, Field

from agent import Agent
from book_tools import BookTools
from quizzes import Quiz, get_quiz
from turns import turn
from settings import LAB2, ROOT


@dataclass
class Dialogue:
    agent: Agent
    tools: BookTools
    lock: Lock


class ChatSessions:
    def __init__(self) -> None:
        self.resources = ExitStack()
        self.dialogues: dict[str, Dialogue] = {}
        self.lock = Lock()

    def get(self, session: str | None) -> tuple[str, Dialogue]:
        with self.lock:
            if session is not None:
                if session not in self.dialogues:
                    raise HTTPException(404, 'Сессия завершилась. Начните новый диалог.')
                return session, self.dialogues[session]
            session = uuid4().hex
            # Keep resources only after both objects were created successfully.
            with ExitStack() as opening:
                try:
                    tools = opening.enter_context(closing(BookTools()))
                    agent = opening.enter_context(closing(Agent(tools)))
                except FileNotFoundError as error:
                    raise HTTPException(503, 'Подготовьте активный индекс Лабы 2.') from error
                self.resources.enter_context(opening.pop_all())
            dialogue = Dialogue(agent, tools, Lock())
            self.dialogues[session] = dialogue
            return session, dialogue


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    sessions = ChatSessions()
    app.state.chat_sessions = sessions
    try:
        yield
    finally:
        try:
            # Agent.close() runs its own asyncio.Runner.
            await run_in_threadpool(sessions.resources.close)
        finally:
            get_client().flush()


app = FastAPI(lifespan=lifespan)


class Chat(BaseModel):
    message: str = Field(min_length=1, max_length=6000)
    session: str | None = None


class BookPage(BaseModel):
    page: int
    url: str


class ChatResponse(BaseModel):
    session: str
    answer: str
    status: str
    trace_url: str | None
    book_pages: list[BookPage]


class Attempt(BaseModel):
    answers: list[int]


@app.get('/', response_class=HTMLResponse)
def home() -> str:
    return (ROOT / 'web/index.html').read_text(encoding='utf-8')


@app.post('/api/chat')
def chat(request: Request, message: Chat) -> ChatResponse:
    sessions: ChatSessions = request.app.state.chat_sessions
    session, dialogue = sessions.get(message.session)
    with dialogue.lock:
        row = turn(dialogue.agent, dialogue.tools, message.message, session, 'custom', 'web')
        pages = dict.fromkeys(hit['page'] for hits in row.retrievals for hit in hits)
    book_pages = [BookPage(page=page, url=f'/document/{session}#page={page}') for page in pages]
    return ChatResponse(session=session, answer=row.output, status=row.status,
                        trace_url=row.trace_url, book_pages=book_pages)


@app.get('/document/{session}', response_class=FileResponse)
def document(request: Request, session: str) -> FileResponse:
    sessions: ChatSessions = request.app.state.chat_sessions
    _, dialogue = sessions.get(session)
    pdf = LAB2 / dialogue.tools.rag.manifest['corpus_path'] / 'book.pdf'
    if not pdf.is_file():
        raise HTTPException(404, 'PDF учебника не найден в корпусе Лабы 2.')
    return FileResponse(pdf, media_type='application/pdf')


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
