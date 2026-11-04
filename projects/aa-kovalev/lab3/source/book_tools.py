"""Три общих инструмента; поиск импортируется из Лабы 2 без копирования."""
import ast
import json
import operator
import re
import sys
from time import monotonic
import httpx
from langfuse import get_client
from simpleeval import SimpleEval
from settings import LAB2, CONFIG

sys.path.insert(0, str(LAB2 / 'source'))
from rag import RAG

OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
             ast.Mod: operator.mod, ast.USub: operator.neg, ast.UAdd: operator.pos}
MAX_EXPRESSION_LENGTH = 200


class BookTools:
    def __init__(self, index=None):
        if index is None:
            active = json.loads((LAB2 / 'artifacts/active.json').read_text(encoding='utf-8'))
            index = LAB2 / active['index']
        self.rag = RAG(index)
        corpus = LAB2 / self.rag.manifest['corpus_path']
        parts = re.split(r'(?m)^<!-- PAGE:(\d+) -->\s*',
                         (corpus / 'book.md').read_text(encoding='utf-8'))
        self.pages = dict(zip(map(int, parts[1::2]), parts[2::2]))
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
                elif name == 'read_page':
                    page = arguments['page']
                    result = {'page': page, 'text': self.pages[page]}
                elif name == 'calculate':
                    expression = arguments['expression']
                    if len(expression) > MAX_EXPRESSION_LENGTH or not re.fullmatch(r'[\d\s.+*/%()\-]+', expression):
                        raise ValueError('Разрешены только короткие арифметические выражения')
                    result = SimpleEval(operators=OPERATORS, names={}, functions={}).eval(expression)
                else:
                    raise ValueError('Неизвестный инструмент: ' + name)
                span.update(output=result)
                return result
            except Exception as error:
                record['error'] = str(error)
                result = {'error': str(error), 'hint': 'Исправь аргументы или используй другой инструмент.'}
                span.update(output=result, level='ERROR', status_message=str(error))
                return result

    def close(self):
        self.rag.close()
