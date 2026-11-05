"""PydanticAI agent with a token-budgeted history processor."""
import asyncio
import json
from collections.abc import AsyncIterable, Callable
from contextvars import copy_context
from functools import lru_cache
from time import monotonic

import httpx2
from langfuse import get_client
from pydantic_ai import Agent as PydanticAgent, ModelRetry, RunContext, UsageLimits
from pydantic_ai.capabilities import Instrumentation, ProcessHistory
from pydantic_ai.messages import (
    AgentStreamEvent, FunctionToolCallEvent, FunctionToolResultEvent, ModelMessage,
    ModelMessagesTypeAdapter, ModelRequest, ToolReturnPart, UserPromptPart,
)
from pydantic_ai.models.ollama import OllamaModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai_harness.tool_output_limits import Band, ToolOutputLimits, Truncate
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from constants import CREATE, DELETE, MANAGE_QUIZ, TOOL_OUTPUT_CHARS, TURN_RESERVE_TOKENS, UPDATE
from models.schemas import AgentStats
from tools.book import BookTools
from tools.pydantic_tools import build_tools
from services.prompts import system_prompt
from settings import API_KEY, BASE_URL, CONFIG, SCHEMAS

INPUT_BUDGET = CONFIG['context_tokens'] - CONFIG['max_output_tokens'] - CONFIG['context_reserve_tokens']


@lru_cache(maxsize=1)
def tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(CONFIG['tokenizer'])


def count_text(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


def starts_a_turn(message: ModelMessage) -> bool:
    return (isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) for part in message.parts)
            and not any(isinstance(part, ToolReturnPart) for part in message.parts))


class ContextBudgetExceeded(ValueError):
    pass


def require_nonempty_answer(output: str) -> str:
    if not output.strip():
        raise ModelRetry('Продолжи задачу пользователя. Вызови нужный инструмент или дай итоговый ответ.')
    return output


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
        self.agent = PydanticAgent(
            model, instructions=self.instructions, retries=2,
            capabilities=[
                Instrumentation(),
                ToolOutputLimits(bands=[Band(TOOL_OUTPUT_CHARS, Truncate(max_chars=TOOL_OUTPUT_CHARS))]),
                ProcessHistory(self.fit_context),
            ],
            tools=build_tools(tools))
        self.agent.output_validator(require_nonempty_answer)

    def count(self, messages: list[ModelMessage]) -> int:
        payload = ModelMessagesTypeAdapter.dump_json(messages, exclude_defaults=True, exclude_none=True).decode()
        return count_text(self.instructions + json.dumps(SCHEMAS, ensure_ascii=False) + payload)

    async def fit_context(self, ctx: RunContext[None], messages: list[ModelMessage]) -> list[ModelMessage]:
        before = self.count(messages)
        after = before
        budget = INPUT_BUDGET
        if self.first_request:
            self.first_request = False
            budget -= TURN_RESERVE_TOKENS
            if before > budget:
                with get_client().start_as_current_observation(name='context-trim',
                        input={'tokens': before, 'budget': budget,
                               'turn_reserve_tokens': TURN_RESERVE_TOKENS}) as span:
                    original_count = len(messages)
                    for i in range(1, original_count):
                        if starts_a_turn(messages[i]):
                            remaining = messages[i:]
                            remaining_tokens = self.count(remaining)
                            if remaining_tokens <= budget:
                                messages = remaining
                                after = remaining_tokens
                                self.trimmed = self.stats.trimmed = True
                                break
                    span.update(output={'tokens': after,
                                        'messages_before': original_count,
                                        'messages_after': len(messages)})
        if after > budget:
            raise ContextBudgetExceeded('Не хватило места в контексте для продолжения хода.')
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

    async def stream_events(self, ctx: RunContext, events: AsyncIterable[AgentStreamEvent]) -> None:
        emit = self.on_event
        if emit is None:
            return
        async for event in events:
            if isinstance(event, FunctionToolCallEvent):
                emit('tool', event.part.tool_name)
            elif isinstance(event, FunctionToolResultEvent):
                emit('tool_done', event.part.tool_name)

    def answer(self, question: str) -> tuple[str, AgentStats]:
        self.first_request = True
        self.stats = AgentStats(
            input_tokens=0, output_tokens=0, steps=0,
            trimmed=False, max_context=0, status='success',
        )
        try:
            request = self.agent.run(question, message_history=self.history,
                usage_limits=UsageLimits(request_limit=CONFIG['max_steps']),
                model_settings=self.model_settings(CONFIG['max_output_tokens']),
                event_stream_handler=self.stream_events if self.on_event else None)
            result = self.loop.run(asyncio.wait_for(request, CONFIG['timeout_seconds']), context=copy_context())
            self.history = result.all_messages()
            self.add_usage(result.usage)
            answer = result.output
        except ContextBudgetExceeded as error:
            self.stats.status = 'error'
            actions = {CREATE: 'Создан квиз', UPDATE: 'Изменён квиз', DELETE: 'Удалён квиз'}
            completed = [
                f"{actions[call.arguments['action']]} `{call.result['quiz_id']}`."
                for call in self.tools.calls
                if call.name == MANAGE_QUIZ and call.error is None
                and isinstance(call.result, dict) and 'quiz_id' in call.result
            ]
            answer = '\n'.join([str(error), *completed])
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
