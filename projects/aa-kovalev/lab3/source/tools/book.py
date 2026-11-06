"""Общие инструменты агентов и их трассировка."""
import json
from pathlib import Path
from time import monotonic
from threading import Lock

from constants import GET_QUIZ_INFO, LIST_QUIZES, MANAGE_QUIZ, SEARCH_BOOK
from models.domain import SearchHit, ToolArguments, ToolResult
from models.schemas import ToolCall
import httpx
from langfuse import get_client
from settings import LAB2, CONFIG
from services.quizzes import manage_quiz, list_quizes, get_quiz_info

from rag import RAG
from tools.result_budget import count_result, limit_result

class BookTools:
    session: str

    def __init__(self, index: Path | None = None) -> None:
        if index is None:
            active = json.loads((LAB2 / 'artifacts/active.json').read_text(encoding='utf-8'))
            index = LAB2 / active['index']
        self.rag = RAG(index)
        self.reset()

    def reset(self) -> None:
        self.calls: list[ToolCall] = []
        self.retrievals: list[list[SearchHit]] = []
        self.returned_tool_tokens = 0
        self._output_lock = Lock()
        self.deadline = monotonic() + CONFIG['timeout_seconds']

    def call(self, name: str, arguments: ToolArguments) -> ToolResult:
        lf = get_client()
        with lf.start_as_current_observation(name=name, as_type='tool', input=arguments) as span:
            record = ToolCall(name=name, arguments=arguments)
            self.calls.append(record)
            try:
                result = self.dispatch(name, arguments)
                record.result = result  # Полный результат для отчёта и оценок.
                original_tokens = count_result(result)
                # Общий предел: вся доступная входная часть окна модели.
                # Если результат сам больше неё, очистка истории не спасёт.
                input_budget = (CONFIG['context_tokens']
                                - CONFIG['max_output_tokens']
                                - CONFIG['context_reserve_tokens']) * 0.7
                with self._output_lock:
                    remaining = input_budget - self.returned_tool_tokens
                    budget = max(0, remaining)
                    delivered, truncated = limit_result(result, budget)
                    delivered_tokens = count_result(delivered)
                    self.returned_tool_tokens += delivered_tokens
                record.truncated = truncated
                record.returned_tokens = delivered_tokens
                span.update(output=result, metadata={
                    'full_result_tokens': original_tokens,
                    'returned_result_tokens': delivered_tokens,
                    'returned_tool_tokens_total': self.returned_tool_tokens,
                    'truncated_for_model': truncated,
                })
                return delivered
            except Exception as error:
                record.error = str(error)
                result = {'error': str(error), 'hint': 'Исправь аргументы или используй другой инструмент.'}
                span.update(output=result, level='ERROR', status_message=str(error))
                return result

    def dispatch(self, name: str, arguments: ToolArguments) -> ToolResult:
        remaining = self.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('Исчерпан таймаут реплики')
        if name == SEARCH_BOOK:
            return self.search_book(arguments, remaining)
        if name == MANAGE_QUIZ:
            return manage_quiz(self.session, **arguments)
        if name == LIST_QUIZES:
            quizzes = list_quizes(self.session)
            result: ToolArguments = {'quizzes': quizzes, 'count': len(quizzes)}
            if not quizzes:
                result['message'] = 'В доступной коллекции пока нет квизов.'
            return result
        if name == GET_QUIZ_INFO:
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
            # Обрезка всех ответов инструментов выполняется по токенам
            # в call(); полные найденные фрагменты остаются в retrieval.
            return [
                {'id': hit['id'], 'page': hit['page'],
                 'text': hit['text'], 'truncated': False}
                for hit in hits
            ]

    def close(self) -> None:
        self.rag.close()
