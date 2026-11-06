"""PydanticAI agent with official tiered history compaction."""
import asyncio
import json
import re
from collections.abc import AsyncIterable, Callable
from contextvars import copy_context
from dataclasses import replace
from time import monotonic

import httpx2
from langfuse import get_client
from pydantic_ai import Agent as PydanticAgent, ModelRetry, RunContext, UsageLimits, capture_run_messages
from pydantic_ai.capabilities import Instrumentation, PrepareTools
from pydantic_ai.messages import (
    AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent, ModelMessage,
    ModelMessagesTypeAdapter, ModelRequest, ToolReturnPart, UserPromptPart, repair_messages,
)
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai_harness.compaction import (
    ClearToolResults, SlidingWindowCompaction, SummarizingCompaction, TieredCompaction,
)
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.usage import RunUsage

from constants import (
    APPEND_QUESTIONS, CREATE, DELETE, DELETE_QUESTION, MANAGE_QUIZ,
    GET_QUIZ_INFO, LIST_QUIZES, MAX_TOOL_CALLS_PER_TURN,
    TURN_RESERVE_TOKENS, UPDATE,
)
from models.schemas import AgentStats
from tools.book import BookTools
from tools.result_budget import count_text, count_result, limit_result
from tools.pydantic_tools import build_tools
from services.prompts import system_prompt
from settings import API_KEY, BASE_URL, CONFIG, ROOT

INPUT_BUDGET = CONFIG['context_tokens'] - CONFIG['max_output_tokens'] - CONFIG['context_reserve_tokens']
START_BUDGET = INPUT_BUDGET - TURN_RESERVE_TOKENS


QUIZ_WRITE_CLAIM = re.compile(
    r'(?<!не )(?:добавлен|создан|удал[её]н|измен[её]н|обновл[её]н|замен[её]н)(?:а|о|ы)?\b',
    re.IGNORECASE,
)
QUIZ_ACTION_LABELS = {
    CREATE: 'Создан квиз',
    UPDATE: 'Изменён квиз',
    DELETE: 'Удалён квиз',
    APPEND_QUESTIONS: 'Добавлены вопросы в квиз',
    DELETE_QUESTION: 'Удалён вопрос в квизе',
}


class ObservedCompaction(TieredCompaction):
    """Budget adapter: native strategies operate only on completed turns."""
    def __init__(self, *, observe, measure, **kwargs):
        super().__init__(**kwargs)
        self.observe = observe
        self.measure = measure

    async def before_model_request(self, ctx, request_context):
        messages = list(request_context.messages)
        before = self.measure(messages)
        # The current user turn is an indivisible suffix, including its tool calls.
        boundary = next(
            (i for i, message in enumerate(messages)
            if message.run_id == ctx.run_id), None
        )
        if boundary is None:
            boundary = next(
                (i for i in range(len(messages) - 1, -1, -1)
                 if isinstance(messages[i], ModelRequest)
                and any(isinstance(p, UserPromptPart) for p in messages[i].parts)), 0
            )
        
        # вот тут оч хотелось сохранить последние мысли, иначе чет плохое сделаем.
        old, current = messages[:boundary], messages[boundary:]
        changed = False
        if before > self.target_tokens and old:
            with get_client().start_as_current_observation(
                name='context-trim',
                input={
                    'tokens': before,
                    'budget': self.target_tokens,
                    'protected_messages': len(current)},
            ) as span:
                strategies = []
                for tier in self.tiers:
                    if self.measure([*old, *current]) <= self.target_tokens:
                        break
                    old = await tier.compact(old, replace(ctx, messages=old))
                    strategies.append(type(tier).__name__)
                # A retained tail is a floor, not a ceiling. Remove complete old
                # turns if native strategies still cannot satisfy our input budget.
                while old and self.measure([*old, *current]) > self.target_tokens:
                    cut = next(
                        (i for i in range(1, len(old))
                        if isinstance(old[i], ModelRequest)
                        and any(isinstance(p, UserPromptPart) for p in old[i].parts)), len(old)
                    )
                    old = old[cut:]
                    strategies.append('drop_oldest_turn')
                compacted = [*old, *current]
                changed = ModelMessagesTypeAdapter.dump_json(messages) != ModelMessagesTypeAdapter.dump_json(compacted)
                messages = compacted
                span.update(
                    output={
                        'tokens': self.measure(messages),
                        'strategies': strategies,
                        'messages_after': len(messages)
                    }
                )
        # Если новый результат инструмента переполняет уже очищенный
        # контекст, ограничиваем именно его, не стирая текущий запрос и
        # вызовы инструментов. Полный результат остался в BookTools / Langfuse.
        if self.measure(messages) > INPUT_BUDGET:
            with get_client().start_as_current_observation(
                name='tool-output-trim',
                input={'tokens': self.measure(messages), 'budget': INPUT_BUDGET},
            ) as span:
                reduced = []
                while self.measure(messages) > INPUT_BUDGET:
                    overflow = self.measure(messages) - INPUT_BUDGET
                    candidates = [
                        (i, j, part, count_result(part.content))
                        for i, message in enumerate(messages)
                        if isinstance(message, ModelRequest)
                        for j, part in enumerate(message.parts)
                        if isinstance(part, ToolReturnPart)
                        and isinstance(part.content, (dict, list))
                    ]
                    # Сначала результаты поиска, затем другие объёмные
                    # данные; квитанции manage_quiz — последними.
                    candidates.sort(key=lambda x: (
                        x[2].tool_name == MANAGE_QUIZ,
                        x[2].tool_name != 'search_book',
                        -x[3],
                    ))
                    for i, j, part, size in candidates:
                        budget = max(128, size - overflow - 32)
                        if budget >= size:
                            continue
                        try:
                            content, truncated = limit_result(part.content, budget)
                        except ValueError:
                            continue
                        if not truncated or count_result(content) >= size:
                            continue
                        message = messages[i]
                        parts = list(message.parts)
                        parts[j] = replace(part, content=content)
                        messages[i] = replace(message, parts=parts)
                        reduced.append({
                            'tool': part.tool_name,
                            'before_tokens': size,
                            'after_tokens': count_result(content),
                        })
                        # trimmed в метриках истории учитывает только compaction
                        # старых ходов, не ограничение нового tool-ответа.
                        break
                    else:
                        break
                span.update(output={
                    'tokens': self.measure(messages),
                    'fits': self.measure(messages) <= INPUT_BUDGET,
                    'truncated_tools': reduced,
                })
        ctx.messages[:] = messages
        self.observe(ctx, changed)
        return replace(request_context, messages=messages)


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
            observe=self.record_context, measure=self.count,
            tiers=[
                ClearToolResults(
                    max_tokens=1, keep_pairs=2, tokenizer=count_text,
                    exclude_tools=frozenset({MANAGE_QUIZ}),
                ),
                SummarizingCompaction(
                    max_tokens=1, keep_messages=0, tokenizer=count_text,
                    preserve_first_user_message=False,
                    model_settings={'max_tokens': CONFIG['max_output_tokens'],
                                    'temperature': CONFIG['temperature']},
                ).with_focus((ROOT / 'prompts/summary_focus.txt').read_text(encoding='utf-8')),
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
        self.agent.output_validator(self.finalize_answer)
        # Считаем контекст по реальным схемам, сгенерированным PydanticAI.
        self._tool_schemas = json.dumps([
            {'type': 'function', 'function': {
                'name': t.tool_def.name,
                'description': t.tool_def.description,
                'parameters': t.tool_def.parameters_json_schema,
            }}
            for t in registered_tools
        ], ensure_ascii=False)

    def completed_quiz_actions(self) -> list[str]:
        completed = []
        for call in self.tools.calls:
            if call.name != MANAGE_QUIZ or call.error is not None or not isinstance(call.result, dict):
                continue
            label = QUIZ_ACTION_LABELS.get(call.arguments.get('action'))
            quiz_id = call.result.get('quiz_id')
            if label is None or not quiz_id:
                continue
            title, url = call.result.get('title'), call.result.get('url')
            quiz = f'[{title}]({url}) · ID `{quiz_id}`' if title and url else f'`{quiz_id}`'
            line = f'{label} {quiz}.'
            numbers = call.result.get('added_question_numbers')
            if numbers:
                line += ' Номера: ' + ', '.join(map(str, numbers)) + '.'
            completed.append(line)
        return completed

    def finalize_answer(self, ctx: RunContext[None], output: str) -> str:
        if self.on_event:
            self.on_event('finalizing', '')
        completed = self.completed_quiz_actions()
        if completed:
            return '\n'.join(completed)
        if not output.strip():
            raise ModelRetry('Продолжи задачу пользователя. Вызови нужный инструмент или дай итоговый ответ.')
        prompt = ctx.prompt if isinstance(ctx.prompt, str) else ''
        if (re.search(r'\b(?:добавь|добавляй|создай|создавай|удали|удаляй|измени|замени|обнови)\b', prompt, re.I)
                and QUIZ_WRITE_CLAIM.search(output)):
            raise ModelRetry('В этом ходе нет успешного вызова manage_quiz. Выполни запрошенное '
                             'действие через инструмент или прямо сообщи, что оно не выполнено.')
        return output

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
            raise ValueError('Текущий ход превышает бюджет даже без старой истории. '
                             'Уменьшите объём запроса или результатов инструментов.')

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
            answer = '\n'.join([str(error), *self.completed_quiz_actions()])

        self.stats.trimmed_before_answer = self.trimmed
        self.stats.context_tokens = self.count(self.history)
        return answer, self.stats

    def close(self) -> None:
        self.loop.run(self.http.aclose())
        self.loop.close()
