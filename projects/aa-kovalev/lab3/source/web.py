"""Local quiz forms and chat API."""
import asyncio
import json
import re
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import ExitStack, asynccontextmanager, closing
from dataclasses import dataclass
from html import escape
from string import Template
from threading import Lock
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from langfuse import get_client
from pydantic import BaseModel, Field
from pydantic_ai.messages import ModelMessagesTypeAdapter

from agent import Agent
from book_tools import BookTools
from quizzes import Quiz, get_quiz, list_quizes
from turns import turn
from settings import LAB2, ROOT


DIALOGUES_DB = ROOT / 'dialogues.sqlite3'
QUIZ_LINK = re.compile(
    r'(?P<label>\[[^\]\n]+\])\((?:https?://[^/\s)]+)?(?P<path>/quiz/[^\s)]+)\)'
)


def connect_dialogues() -> sqlite3.Connection:
    db = sqlite3.connect(DIALOGUES_DB)
    db.execute("""CREATE TABLE IF NOT EXISTS dialogues (
        session TEXT PRIMARY KEY, history BLOB NOT NULL, transcript TEXT NOT NULL
    )""")
    return db


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
            history = []
            if session is None:
                session = uuid4().hex
            elif session in self.dialogues:
                return session, self.dialogues[session]
            else:
                with closing(connect_dialogues()) as db:
                    row = db.execute('SELECT history FROM dialogues WHERE session = ?', (session,)).fetchone()
                if row is None:
                    raise HTTPException(404, 'Диалог не найден.')
                history = ModelMessagesTypeAdapter.validate_json(row[0])
            # Keep resources only after both objects were created successfully.
            with ExitStack() as opening:
                try:
                    tools = opening.enter_context(closing(BookTools()))
                    agent = opening.enter_context(closing(Agent(tools)))
                except FileNotFoundError as error:
                    raise HTTPException(503, 'Подготовьте активный индекс Лабы 2.') from error
                self.resources.enter_context(opening.pop_all())
            agent.history = history
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


class ChatTurn(BaseModel):
    message: str
    result: ChatResponse


class Transcript(BaseModel):
    session: str
    title: str
    turns: list[ChatTurn] = Field(default_factory=list)


class Attempt(BaseModel):
    answers: list[int]


@app.get('/', response_class=HTMLResponse)
def home() -> str:
    return (ROOT / 'web/index.html').read_text(encoding='utf-8')


def verify_quiz_links(answer: str, session: str) -> str:
    if not QUIZ_LINK.search(answer):
        return answer
    urls = {quiz['url'] for quiz in list_quizes(session)}
    return QUIZ_LINK.sub(
        lambda match: f"{match['label']}({match['path']})" if match['path'] in urls
        else 'Ссылка на квиз недоступна.',
        answer,
    )


@app.get('/api/dialogues')
def dialogues() -> list[Transcript]:
    with closing(connect_dialogues()) as db:
        rows = db.execute('SELECT transcript FROM dialogues ORDER BY rowid DESC').fetchall()
    transcripts = [Transcript.model_validate_json(row[0]) for row in rows]
    for transcript in transcripts:
        for entry in transcript.turns:
            entry.result.answer = verify_quiz_links(entry.result.answer, transcript.session)
    return transcripts


@app.post('/api/chat')
def chat(request: Request, message: Chat) -> ChatResponse:
    sessions: ChatSessions = request.app.state.chat_sessions
    session, dialogue = sessions.get(message.session)
    return run_chat(session, dialogue, message.message)


def run_chat(session: str, dialogue: Dialogue, message: str,
             emit: Callable[[str, str], None] | None = None) -> ChatResponse:
    with dialogue.lock:
        dialogue.agent.on_event = emit
        try:
            row = turn(dialogue.agent, dialogue.tools, message, session, 'custom', 'web')
        finally:
            dialogue.agent.on_event = None
        pages = dict.fromkeys(hit['page'] for hits in row.retrievals for hit in hits)
        book_pages = [BookPage(page=page, url=f'/document/{session}#page={page}') for page in pages]
        result = ChatResponse(session=session, answer=verify_quiz_links(row.output, session),
                              status=row.status,
                              trace_url=row.trace_url, book_pages=book_pages)
        with closing(connect_dialogues()) as db, db:
            saved = db.execute('SELECT transcript FROM dialogues WHERE session = ?', (session,)).fetchone()
            transcript = (Transcript.model_validate_json(saved[0]) if saved
                          else Transcript(session=session, title=message.strip()[:70]))
            transcript.turns.append(ChatTurn(message=message, result=result))
            db.execute("""INSERT INTO dialogues (session, history, transcript) VALUES (?, ?, ?)
                          ON CONFLICT(session) DO UPDATE SET
                              history = excluded.history, transcript = excluded.transcript""",
                       (session, ModelMessagesTypeAdapter.dump_json(dialogue.agent.history),
                        transcript.model_dump_json()))
    return result


@app.post('/api/chat/stream')
async def chat_stream(request: Request, message: Chat) -> StreamingResponse:
    sessions: ChatSessions = request.app.state.chat_sessions
    session, dialogue = await run_in_threadpool(sessions.get, message.session)

    async def events() -> AsyncIterator[str]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, object]] = asyncio.Queue()
        connected = True

        def emit(kind: str, value: str) -> None:
            if connected:
                loop.call_soon_threadsafe(queue.put_nowait, (kind, value))

        async def run() -> None:
            try:
                result = await run_in_threadpool(run_chat, session, dialogue, message.message, emit)
                queue.put_nowait(('result', result.model_dump(mode='json')))
            except Exception:
                queue.put_nowait(('error', 'Не удалось завершить запрос.'))

        worker = asyncio.create_task(run())
        try:
            while True:
                kind, value = await queue.get()
                yield json.dumps({'kind': kind, 'value': value}, ensure_ascii=False) + '\n'
                if kind in {'result', 'error'}:
                    break
        finally:
            connected = False
            await asyncio.shield(worker)

    return StreamingResponse(events(), media_type='application/x-ndjson')


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
