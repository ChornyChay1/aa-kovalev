"""Локальный HTML-интерфейс и REST API RAG."""
from contextlib import asynccontextmanager
from threading import Lock
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from common import ROOT, ARTIFACTS, read
from rag import RAG

@asynccontextmanager
async def lifespan(app):
    active = read(ARTIFACTS / 'active.json')
    app.state.rag = RAG(ROOT / active['index'])
    app.state.lock = Lock()
    yield
    app.state.rag.close()

app = FastAPI(title='Лаба 2: RAG по учебнику Python', lifespan=lifespan)

class Question(BaseModel):
    question: str = Field(min_length=1, max_length=2000)

@app.get('/')
def home():
    return FileResponse(ROOT / 'source' / 'index.html')

@app.get('/document')
def document():
    folder = ROOT / app.state.rag.manifest['corpus_path']
    return FileResponse(folder / 'book.pdf', media_type='application/pdf')

@app.post('/ask')
def ask(body: Question):
    if not body.question.strip():
        raise HTTPException(400, 'Введите вопрос')
    with app.state.lock:
        try:
            hits, elapsed = app.state.rag.retrieve(body.question, app.state.rag.config['context_chunks'])
            return {**app.state.rag.answer(body.question, hits), 'retrieval_s': elapsed}
        except Exception as error:
            raise HTTPException(502, 'Ошибка Qdrant/Ollama: ' + str(error)) from error
