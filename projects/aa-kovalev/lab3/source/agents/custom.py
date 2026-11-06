"""PydanticAI agent with official tiered history compaction."""
import asyncio
import json
from collections.abc import AsyncIterable, Callable
from contextvars import copy_context
from functools import lru_cache
from time import monotonic

import httpx2
from langfuse import get_client
from pydantic_ai import Agent as PydanticAgent, ModelRetry, RunContext, UsageLimits, capture_run_messages
from pydantic_ai.capabilities import Instrumentation, PrepareTools
from pydantic_ai.messages import (
    AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent, ModelMessage,
    ModelMessagesTypeAdapter, repair_messages,
)
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai_harness.compaction import (
    ClearToolResults, SlidingWindowCompaction, SummarizingCompaction, TieredCompaction,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from constants import (
    APPEND_QUESTIONS, CREATE, DELETE, DELETE_QUESTION, MANAGE_QUIZ,
    GET_QUIZ_INFO, LIST_QUIZES, MAX_TOOL_CALLS_PER_TURN,
    TURN_RESERVE_TOKENS, UPDATE,
)
from models.schemas import AgentStats
from tools.book import BookTools
from tools.pydantic_tools import build_tools
from services.prompts import system_prompt
from settings import API_KEY, BASE_URL, CONFIG

INPUT_BUDGET = CONFIG['context_tokens'] - CONFIG['max_output_tokens'] - CONFIG['context_reserve_tokens']
START_BUDGET = INPUT_BUDGET - TURN_RESERVE_TOKENS


@lru_cache(maxsize=1)
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(CONFIG['tokenizer'])


def count_text(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


def validate_answer(output: str) -> str:
    if not output.strip():
        raise ModelRetry('Продолжи задачу пользователя. Вызови нужный инструмент или дай итоговый ответ.')
    return output


class ObservedCompaction(TieredCompaction):
    def __init__(self, *, observe, **kwargs):
        super().__init__(**kwargs)
        self.observe = observe

    async def before_model_request(self, ctx, request_context):
        before = ModelMessagesTypeAdapter.dump_json(ctx.messages)
        result = await super().before_model_request(ctx, request_context)
        changed = before != ModelMessagesTypeAdapter.dump_json(ctx.messages)
        self.observe(ctx, changed)
        return result


class Agent:
    def __init__(self, tools: BookTools) -> None:
        self.instructions, self.prompt_version = system_prompt()
        self.tools = tools
        self.on_event: Callable[[str, str], None] | None = None
        self.history: list[ModelMessage] = []
        self.trimmed = False
        self.loop = asyncio.Runner()
        self.http = httpx2.AsyncClient(trust_env=False)
        model = OllamaModel(CONFIG['model'], provider=OllamaProvider(
            base_url=BASE_URL, api_key=API_KEY, http_client=self.http))
        # PydanticAI Harness: clear old search results, summarize older turns,
        # and use a sliding window only as the last-resort fallback. Placing
        # the sliding window before summarization would normally skip the LLM tier.
        self.compaction = ObservedCompaction(
            observe=self.record_context,
            tiers=[
                ClearToolResults(
                    max_tokens=1, keep_pairs=2, tokenizer=count_text,
                    exclude_tools=frozenset({GET_QUIZ_INFO, LIST_QUIZES, MANAGE_QUIZ}),
                ),
                SummarizingCompaction(
                    max_tokens=1, keep_tokens=4000, tokenizer=count_text,
                    preserve_first_user_message=False,
                    model_settings={'max_tokens': CONFIG['max_output_tokens'],
                                    'temperature': CONFIG['temperature']},
                ),
                SlidingWindowCompaction(
                    max_tokens=1, keep_tokens=4000, tokenizer=count_text,
                    preserve_first_user_message=False,
                ),
            ],
            target_tokens=START_BUDGET,
            tokenizer=count_text,
        )
        registered_tools = build_tools(tools)
        self.agent = PydanticAgent(
            model, instructions=self.instructions, retries=2,
            capabilities=[
                Instrumentation(),
                self.compaction,
                PrepareTools(self.prepare_tools),
            ],
            tools=registered_tools)
        self.agent.output_validator(validate_answer)
        # Считаем контекст по реальным схемам, сгенерированным PydanticAI.
        self._tool_schemas = json.dumps([
            {'type': 'function', 'function': {
                'name': t.tool_def.name,
                'description': t.tool_def.description,
                'parameters': t.tool_def.parameters_json_schema,
            }}
            for t in registered_tools
        ], ensure_ascii=False)

    def count(self, messages: list[ModelMessage]) -> int:
        payload = ModelMessagesTypeAdapter.dump_json(messages, exclude_defaults=True, exclude_none=True).decode()
        return count_text(self.instructions + self._tool_schemas + payload)

    async def prepare_tools(
            self, ctx: RunContext[None],
            tool_defs: list[ToolDefinition]
    ) -> list[ToolDefinition]:
        
        if ctx.usage.tool_calls >= MAX_TOOL_CALLS_PER_TURN:
            return []
        if ctx.run_step >= CONFIG['max_steps']:
            return []
        return tool_defs

    def record_context(self, ctx, changed: bool) -> None:
        tokens = self.count(ctx.messages)
        self.stats.max_context = max(self.stats.max_context or 0, tokens)
        self.stats.trimmed = bool(self.stats.trimmed or changed)
        self.trimmed = self.trimmed or changed
        with get_client().start_as_current_observation(
            name='context-check', input={'step': ctx.run_step, 'tokens': tokens,
                'budget': INPUT_BUDGET, 'count_method': 'tokenized_serialized_messages'},
        ) as span:
            span.update(output={'compacted': changed, 'fits': tokens <= INPUT_BUDGET})
        if tokens > INPUT_BUDGET:
            raise ValueError('Контекст после сокращения превышает входной бюджет.')

    def model_settings(self, max_tokens: int) -> ModelSettings:
        remaining = self.tools.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('Исчерпан таймаут реплики')
        return {'max_tokens': max_tokens, 'temperature': CONFIG['temperature'], 'timeout': remaining}

    def add_usage(self, usage: RunUsage) -> None:
        self.stats.input_tokens = (self.stats.input_tokens or 0) + usage.input_tokens
        self.stats.output_tokens = (self.stats.output_tokens or 0) + usage.output_tokens
        self.stats.steps = (self.stats.steps or 0) + usage.requests

    async def stream_events(self, ctx: RunContext, events: AsyncIterable[AgentStreamEvent]) -> None:
        emit = self.on_event
        if emit is None:
            return
        async for event in events:
            if isinstance(event, FunctionToolCallEvent):
                emit('tool', event.part.tool_name)
            elif isinstance(event, FunctionToolResultEvent):
                emit('tool_done', event.part.tool_name)

    async def run_turn(self, question: str) -> str:
        previous = self.history
        usage = RunUsage()
        with capture_run_messages() as messages:
            try:
                result = await self.agent.run(
                    question,
                    message_history=previous,
                    usage=usage,
                    usage_limits=UsageLimits(
                        request_limit=CONFIG['max_steps'], tool_calls_limit=MAX_TOOL_CALLS_PER_TURN,
                    ),
                    model_settings=self.model_settings(CONFIG['max_output_tokens']),
                    event_stream_handler=self.stream_events if self.on_event else None,
                )
                return result.output
            finally:
                if messages:
                    self.history = repair_messages(messages)
                self.add_usage(usage)

    def answer(self, question: str) -> tuple[str, AgentStats]:
        self.stats = AgentStats(
            input_tokens=0, output_tokens=0, steps=0,
            trimmed=False, max_context=0, status='success',
        )
        try:
            remaining = self.tools.deadline - monotonic()
            answer = self.loop.run(
                asyncio.wait_for(self.run_turn(question), max(0, remaining)), context=copy_context()
            )
        except Exception as error:
            name = type(error).__name__.lower()
            self.stats.status = (
                'max_steps' if name == 'usagelimitexceeded' else 'timeout' if 'timeout' in name else 'error'
            )
            actions = {
                CREATE: 'Создан квиз',
                UPDATE: 'Изменён квиз',
                DELETE: 'Удалён квиз',
                APPEND_QUESTIONS: 'Добавлены вопросы в квиз',
                DELETE_QUESTION: 'Удалён вопрос в квизе'
            }
            completed = [
                f"{actions[call.arguments['action']]} `{call.result['quiz_id']}`."
                for call in self.tools.calls
                if call.name == MANAGE_QUIZ and call.error is None
                and isinstance(call.result, dict) and 'quiz_id' in call.result
            ]
            answer = '\n'.join([str(error), *completed])
        
        self.stats.trimmed_before_answer = self.trimmed
        self.stats.context_tokens = self.count(self.history)
        return answer, self.stats

    def close(self) -> None:
        self.loop.run(self.http.aclose())
        self.loop.close()
