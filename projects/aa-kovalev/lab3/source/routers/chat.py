"""HTTP-маршруты чата и учебника."""
import asyncio
import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from models.schemas import Chat, ChatResponse, Transcript
from services.dialogues import ChatSessions, list_dialogues, run_chat
from settings import LAB2, ROOT

router = APIRouter()


@router.get('/', response_class=HTMLResponse)
def home() -> str:
    return (ROOT / 'web/index.html').read_text(encoding='utf-8')


@router.get('/api/dialogues')
def dialogues() -> list[Transcript]:
    return list_dialogues()


@router.post('/api/chat')
def chat(request: Request, message: Chat) -> ChatResponse:
    sessions: ChatSessions = request.app.state.chat_sessions
    session, dialogue = sessions.get(message.session)
    return run_chat(session, dialogue, message.message)


@router.post('/api/chat/stream')
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


@router.get('/document/{session}', response_class=FileResponse)
def document(request: Request, session: str) -> FileResponse:
    sessions: ChatSessions = request.app.state.chat_sessions
    _, dialogue = sessions.get(session)
    pdf = LAB2 / dialogue.tools.rag.manifest['corpus_path'] / 'book.pdf'
    if not pdf.is_file():
        raise HTTPException(404, 'PDF учебника не найден в корпусе Лабы 2.')
    return FileResponse(pdf, media_type='application/pdf')
