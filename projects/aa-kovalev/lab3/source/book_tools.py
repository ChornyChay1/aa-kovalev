"""Общие инструменты агентов и их трассировка."""
import json
import sys
from time import monotonic
import httpx
from langfuse import get_client
from settings import LAB2, CONFIG
from quizzes import manage_quiz, list_quizes, get_quiz_info

sys.path.insert(0, str(LAB2 / 'source'))
from rag import RAG

class BookTools:
    def __init__(self, index=None):
        if index is None:
            active = json.loads((LAB2 / 'artifacts/active.json').read_text(encoding='utf-8'))
            index = LAB2 / active['index']
        self.rag = RAG(index)
        self.chunks = [p['payload'] for p in json.loads(
            (self.rag.folder / 'points.json').read_text(encoding='utf-8'))]
        self.reset()

    def reset(self):
        self.calls, self.retrievals = [], []
        self.deadline = monotonic() + CONFIG['timeout_seconds']

    def call(self, name, arguments):
        lf = get_client()
        with lf.start_as_current_observation(name=name, as_type='tool', input=arguments) as span:
            record = {'name': name, 'arguments': arguments}
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

    def dispatch(self, name, arguments):
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('Исчерпан таймаут реплики')
        if name == 'search_book':
            return self.search_book(arguments, remaining)
        if name == 'manage_quiz':
            return manage_quiz(self.session, **arguments)
        if name == 'list_quizes':
            return list_quizes(self.session)
        if name == 'get_quiz_info':
            return get_quiz_info(self.session, **arguments)
        raise ValueError('Неизвестный инструмент: ' + name)

    def search_book(self, arguments, remaining):
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

    def close(self):
        self.rag.close()
