"""Сессии веб-чата и хранение истории в SQLite."""
import sqlite3
from collections.abc import Callable
from contextlib import ExitStack, closing
from dataclasses import dataclass
from threading import Lock
from uuid import uuid4

from fastapi import HTTPException
from pydantic_ai.messages import ModelMessagesTypeAdapter

from agents.custom import Agent
from constants import CUSTOM_AGENT, QUIZ_LINK, WEB_QUIZ_COLLECTION, WEB_RUN_ID
from models.schemas import BookPage, ChatResponse, ChatTurn, Transcript
from services.quizzes import list_quizes
from services.turns import turn
from settings import ROOT
from tools.book import BookTools

DIALOGUES_DB = ROOT / 'dialogues.sqlite3'


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


def verify_quiz_links(answer: str) -> str:
    if not QUIZ_LINK.search(answer):
        return answer
    urls = {quiz['url'] for quiz in list_quizes(WEB_QUIZ_COLLECTION)}
    return QUIZ_LINK.sub(
        lambda match: f"{match['label']}({match['path']})" if match['path'] in urls
        else 'Ссылка на квиз недоступна.',
        answer,
    )


def list_dialogues() -> list[Transcript]:
    with closing(connect_dialogues()) as db:
        rows = db.execute('SELECT transcript FROM dialogues ORDER BY rowid DESC').fetchall()
    transcripts = [Transcript.model_validate_json(row[0]) for row in rows]
    for transcript in transcripts:
        for entry in transcript.turns:
            entry.result.answer = verify_quiz_links(entry.result.answer)
    return transcripts


def run_chat(session: str, dialogue: Dialogue, message: str,
             emit: Callable[[str, str], None] | None = None) -> ChatResponse:
    with dialogue.lock:
        dialogue.agent.on_event = emit
        try:
            row = turn(
                dialogue.agent,
                dialogue.tools,
                message,
                session,
                CUSTOM_AGENT,
                WEB_RUN_ID,
                quiz_collection=WEB_QUIZ_COLLECTION
            )
        finally:
            dialogue.agent.on_event = None
        pages = dict.fromkeys(hit['page'] for hits in row.retrievals for hit in hits)
        book_pages = [BookPage(page=page, url=f'/document/{session}#page={page}') for page in pages]
        result = ChatResponse(
            session=session,
            answer=verify_quiz_links(row.output),
            status=row.status,
            trace_url=row.trace_url,
            book_pages=book_pages
        )
        with closing(connect_dialogues()) as db, db:
            saved = db.execute('SELECT transcript FROM dialogues WHERE session = ?', (session,)).fetchone()
            transcript = (
                Transcript.model_validate_json(saved[0])
                if saved else Transcript(session=session, title=message.strip()[:70])
            )
            transcript.turns.append(ChatTurn(message=message, result=result))
            db.execute(
                """INSERT INTO dialogues (session, history, transcript) VALUES (?, ?, ?)
                          ON CONFLICT(session) DO UPDATE SET
                              history = excluded.history, transcript = excluded.transcript""",
                       (session, ModelMessagesTypeAdapter.dump_json(dialogue.agent.history),
                        transcript.model_dump_json())
            )
    return result
