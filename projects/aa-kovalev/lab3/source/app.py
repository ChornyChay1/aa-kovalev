"""Точка входа веб-приложения Lab 3."""
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langfuse import get_client
from starlette.concurrency import run_in_threadpool

from routers import chat, quizzes
from services.dialogues import ChatSessions
from services.quizzes import migrate_web_quizzes


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    await run_in_threadpool(migrate_web_quizzes)
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
app.include_router(chat.router)
app.include_router(quizzes.router)
