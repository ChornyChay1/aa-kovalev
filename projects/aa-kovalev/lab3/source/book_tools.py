"""Общие инструменты агентов и их трассировка."""
import json
import sys
from pathlib import Path
from time import monotonic

from contracts import SearchHit, ToolArguments, ToolCall, ToolResult
import httpx
from langfuse import get_client
from settings import LAB2, CONFIG
from quizzes import manage_quiz, list_quizes, get_quiz_info

sys.path.append(str(LAB2 / 'source'))
from rag import RAG

class BookTools:
    session: str

    def __init__(self, index: Path | None = None) -> None:
        if index is None:
            active = json.loads((LAB2 / 'artifacts/active.json').read_text(encoding='utf-8'))
            index = LAB2 / active['index']
        self.rag = RAG(index)
        self.chunks = [p['payload'] for p in json.loads(
            (self.rag.folder / 'points.json').read_text(encoding='utf-8'))]
        self.reset()

    def reset(self) -> None:
        self.calls: list[ToolCall] = []
        self.retrievals: list[list[SearchHit]] = []
        self.deadline = monotonic() + CONFIG['timeout_seconds']

    def call(self, name: str, arguments: ToolArguments) -> ToolResult:
        lf = get_client()
        with lf.start_as_current_observation(name=name, as_type='tool', input=arguments) as span:
            record: ToolCall = {'name': name, 'arguments': arguments}
            self.calls.append(record)
            try:
                result = self.dispatch(name, arguments)
                record['result'] = result
                span.update(output=result)
                return result
            except Exception as error:
                record['error'] = str(error)
                result = {'error': str(error), 'hint': 'Исправь аргументы или используй другой инструмент.'}
                span.update(output=result, level='ERROR', status_message=str(error))
                return result

    def dispatch(self, name: str, arguments: ToolArguments) -> ToolResult:
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('Исчерпан таймаут реплики')
        if name == 'search_book':
            return self.search_book(arguments, remaining)
        if name == 'manage_quiz':
            return manage_quiz(self.session, **arguments)
        if name == 'list_quizes':
            return [dict(quiz) for quiz in list_quizes(self.session)]
        if name == 'get_quiz_info':
            return get_quiz_info(self.session, **arguments)
        raise ValueError('Неизвестный инструмент: ' + name)

    def search_book(self, arguments: ToolArguments, remaining: float) -> list[SearchHit]:
        # Retrieval делает два последовательных HTTP-запроса.
        self.rag.api.client.timeout = httpx.Timeout(remaining / 2)
        with get_client().start_as_current_observation(
            name='retrieval', as_type='retriever', input=arguments,
            metadata={'source': self.rag.config['source'], 'index': self.rag.manifest['version']},
        ) as span:
            hits, _ = self.rag.retrieve(arguments['query'], CONFIG['retrieval_k'])
            self.retrievals.append(hits)
            span.update(output=hits)
            return hits

    def close(self) -> None:
        self.rag.close()
