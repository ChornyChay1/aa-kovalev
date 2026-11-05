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
                remaining = self.deadline - monotonic()
                if remaining <= 0:
                    raise TimeoutError('Исчерпан таймаут реплики')
                if name == 'search_book':
                    # Retrieval делает два последовательных HTTP-запроса.
                    self.rag.api.client.timeout = httpx.Timeout(remaining / 2)
                    with lf.start_as_current_observation(name='retrieval', as_type='retriever',
                            input=arguments, metadata={'source': self.rag.config['source'],
                                                     'index': self.rag.manifest['version']}) as retrieval:
                        hits, _ = self.rag.retrieve(arguments['query'], CONFIG['retrieval_k'])
                        self.retrievals.append(hits)
                        retrieval.update(output=hits)
                        result = hits
                elif name == 'manage_quiz':
                    result = manage_quiz(self.session, **arguments)
                elif name == 'list_quizes':
                    result = list_quizes(self.session)
                elif name == 'get_quiz_info':
                    result = get_quiz_info(self.session, **arguments)
                else:
                    raise ValueError('Неизвестный инструмент: ' + name)
                record['result'] = result
                span.update(output=result)
                return result
            except Exception as error:
                record['error'] = str(error)
                result = {'error': str(error), 'hint': 'Исправь аргументы или используй другой инструмент.'}
                span.update(output=result, level='ERROR', status_message=str(error))
                return result

    def close(self):
        self.rag.close()
