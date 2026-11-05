"""PydanticAI agent with a token-budgeted history processor."""
import asyncio
import json
from contextvars import copy_context
from functools import lru_cache
from time import monotonic
from typing import Literal

import httpx2
from langfuse import get_client
from pydantic_ai import Agent as PydanticAgent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.capabilities import Instrumentation, ProcessHistory
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded
from pydantic_ai.messages import (
    ModelMessage, ModelMessagesTypeAdapter, ModelRequest, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai_harness import (
    ClearToolResults, FallbackCompaction, SlidingWindowCompaction, SummarizingCompaction,
)
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from book_tools import BookTools
from contracts import AgentStats, ToolResult

from quizzes import Question
from settings import API_KEY, BASE_URL, CONFIG, SCHEMAS, SYSTEM

INPUT_BUDGET = CONFIG['context_tokens'] - CONFIG['max_output_tokens'] - CONFIG['context_reserve_tokens']
SUMMARY_TOKENS = 240
RECENT_HISTORY_TOKENS = 900
FALLBACK_HISTORY_TOKENS = 600
TOOL_OUTPUT_CHARS = 1400


@lru_cache(maxsize=1)
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(CONFIG['tokenizer'])


def count_text(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


def starts_a_turn(message: ModelMessage) -> bool:
    return (isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) for part in message.parts)
            and not any(isinstance(part, ToolReturnPart) for part in message.parts))


class Agent:
    def __init__(self, tools: BookTools) -> None:
        self.tools = tools
        self.history: list[ModelMessage] = []
        self.trimmed = False
        self.loop = asyncio.Runner()
        self.http = httpx2.AsyncClient(trust_env=False)
        model = OllamaModel(CONFIG['model'], provider=OllamaProvider(
            base_url=BASE_URL, api_key=API_KEY, http_client=self.http))
        self.sliding_window = SlidingWindowCompaction(
            max_tokens=INPUT_BUDGET, keep_tokens=FALLBACK_HISTORY_TOKENS, tokenizer=count_text,
            preserve_first_user_message=False)
        self.clear_tool_results = ClearToolResults(max_tokens=INPUT_BUDGET, keep_pairs=0)
        self.compaction = FallbackCompaction(
            fallback_chain=[
                SummarizingCompaction(
                    max_tokens=INPUT_BUDGET, keep_tokens=RECENT_HISTORY_TOKENS, tokenizer=count_text,
                    preserve_first_user_message=False, retries=0,
                    model_settings={'max_tokens': SUMMARY_TOKENS,
                                    'temperature': CONFIG['temperature'],
                                    'timeout': CONFIG['timeout_seconds']},
                    summarization_capabilities=[Instrumentation()]),
                self.sliding_window,
            ],
            fallback_on=(ModelAPIError, UnexpectedModelBehavior, UsageLimitExceeded),
            tokenizer=count_text)
        self.agent = PydanticAgent(
            model, instructions=SYSTEM, retries=2,
            capabilities=[
                Instrumentation(),
                ToolOutputLimits(bands=[Band(TOOL_OUTPUT_CHARS, Truncate(max_chars=TOOL_OUTPUT_CHARS))]),
                ProcessHistory(self.fit_context),
            ],
            tools=[self.search_book, self.manage_quiz, self.list_quizes, self.get_quiz_info])

    def search_book(self, query: str) -> ToolResult:
        """Найти материал в учебнике через RAG Лабы 2."""
        return self.tools.call('search_book', {'query': query})

    def manage_quiz(self, action: Literal['create', 'update', 'delete'],
                    quiz_id: str | None = None, title: str | None = None,
                    questions: list[Question] | None = None,
                    question_number: int | None = None, question: Question | None = None) -> ToolResult:
        """Создать квиз, заменить один вопрос или удалить квиз. Нумерация с 1."""
        if action == 'create' and (not title or not title.strip() or len(title) > 150
                                   or not questions or len(questions) > 10):
            raise ModelRetry(
                'Для create придумай непустой title до 150 символов и передай questions '
                'с 1–10 полностью заполненными вопросами. Соблюдай количество из запроса.'
            )
        args = dict(action=action, quiz_id=quiz_id, title=title, question_number=question_number,
                    questions=[q.model_dump() for q in questions] if questions else None,
                    question=question.model_dump() if question else None)
        return self.tools.call('manage_quiz', args)

    def list_quizes(self) -> ToolResult:
        """Получить ID, названия и ссылки квизов в этой сессии."""
        return self.tools.call('list_quizes', {})

    def get_quiz_info(self, quiz_id: str) -> ToolResult:
        """Получить вопросы и варианты ответов квиза по ID."""
        return self.tools.call('get_quiz_info', {'quiz_id': quiz_id})

    def count(self, messages: list[ModelMessage]) -> int:
        payload = ModelMessagesTypeAdapter.dump_json(messages, exclude_defaults=True, exclude_none=True).decode()
        return count_text(SYSTEM + json.dumps(SCHEMAS, ensure_ascii=False) + payload)

    async def fit_context(self, ctx: RunContext[None], messages: list[ModelMessage]) -> list[ModelMessage]:
        before = self.count(messages)
        after = before
        if before > INPUT_BUDGET:
            with get_client().start_as_current_observation(name='context-trim',
                    input={'tokens': before, 'budget': INPUT_BUDGET}) as span:
                latest_turn = max((i for i, message in enumerate(messages) if starts_a_turn(message)), default=0)
                current_turn = messages[latest_turn:]
                if self.count(current_turn) > INPUT_BUDGET:
                    current_turn = await self.clear_tool_results.compact(current_turn, ctx)
                if self.count(current_turn) > INPUT_BUDGET:
                    raise ValueError('Текущая реплика с результатами инструментов слишком длинная. Уточните запрос.')
                compacted = await self.compaction.compact(messages, ctx)
                after = self.count(compacted)
                # Harness's trigger is an estimate, and its retained tail can exceed it.
                # Keep the current complete turn, as in the ProcessHistory example.
                if after > INPUT_BUDGET or messages[latest_turn] not in compacted:
                    compacted = current_turn
                    after = self.count(compacted)
                if compacted != messages:
                    self.trimmed = self.stats.trimmed = True
                span.update(output={'tokens': after, 'messages_before': len(messages),
                                    'messages_after': len(compacted)})
                messages = compacted
        self.stats.max_context = max(self.stats.max_context or 0, after)
        return messages

    def model_settings(self, max_tokens: int) -> ModelSettings:
        remaining = self.tools.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('Исчерпан таймаут реплики')
        return {'max_tokens': max_tokens, 'temperature': CONFIG['temperature'], 'timeout': remaining}

    def add_usage(self, usage: RunUsage) -> None:
        self.stats.input_tokens = (self.stats.input_tokens or 0) + usage.input_tokens
        self.stats.output_tokens = (self.stats.output_tokens or 0) + usage.output_tokens
        self.stats.steps = (self.stats.steps or 0) + usage.requests

    def answer(self, question: str) -> tuple[str, AgentStats]:
        self.stats = AgentStats(
            input_tokens=0, output_tokens=0, steps=0,
            trimmed=False, max_context=0, status='success',
        )
        try:
            request = self.agent.run(question, message_history=self.history,
                usage_limits=UsageLimits(request_limit=CONFIG['max_steps']),
                model_settings=self.model_settings(CONFIG['max_output_tokens']))
            result = self.loop.run(asyncio.wait_for(request, CONFIG['timeout_seconds']), context=copy_context())
            self.history = result.all_messages()
            self.add_usage(result.usage)
            answer = result.output
        except Exception as error:
            name = type(error).__name__.lower()
            self.stats.status = ('max_steps' if name == 'usagelimitexceeded'
                                    else 'timeout' if 'timeout' in name else 'error')
            answer = str(error)
        self.stats.trimmed_before_answer = self.trimmed
        self.stats.context_tokens = self.count(self.history)
        return answer, self.stats

    def close(self) -> None:
        self.loop.run(self.http.aclose())
        self.loop.close()
