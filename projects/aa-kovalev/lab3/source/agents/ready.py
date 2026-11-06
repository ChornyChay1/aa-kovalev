"""Готовый smolagents, без изменения его цикла и стратегии памяти."""
import json
from uuid import uuid4
from time import monotonic

import httpx

from models.domain import TurnStatus
from models.schemas import AgentStats
from tools.book import BookTools
from tools.smolagents_tools import build_tools
from smolagents import CodeAgent, ToolCallingAgent, OpenAIServerModel
from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from services.prompts import system_prompt
from settings import API_KEY, BASE_URL, CONFIG


class OllamaToolModel(OpenAIServerModel):
    def __init__(self, tools: BookTools, **kwargs) -> None:
        super().__init__(**kwargs)
        self.book_tools = tools

    def generate(self, messages, *, tools_to_call_from=None, **kwargs) -> ChatMessage:
        remaining = self.book_tools.deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError('Исчерпан таймаут реплики')
        kwargs['timeout'] = remaining
        message = super().generate(messages, tools_to_call_from=tools_to_call_from, **kwargs)
        # Ollama sometimes selects a tool with extra arguments copied from
        # another tool (notably 'title'). Ignore only unrecognized keys when
        # every required input is present; never guess a missing quiz ID/action.
        available = {tool.name: tool for tool in (tools_to_call_from or [])}
        for call in message.tool_calls or []:
            arguments = call.function.arguments
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    continue  # Let smolagents report malformed JSON.
            if not isinstance(arguments, dict):
                continue
            selected = available.get(call.function.name)
            if selected is None:
                continue
            allowed = getattr(selected, 'inputs', None)
            if allowed is None:
                continue
            required = {key for key, spec in allowed.items() if not spec.get('nullable')}
            if required <= arguments.keys():
                extras = arguments.keys() - allowed.keys()
                if extras:
                    call.function.arguments = {
                        key: value for key, value in arguments.items() if key in allowed
                    }

        # The framework cannot execute final_answer alongside other tools.
        # Keep actionable calls and let the agent request a final answer next.
        calls = message.tool_calls or []
        actions = [call for call in calls if call.function.name != 'final_answer']
        if actions:
            message.tool_calls = actions
        elif len(calls) > 1:
            message.tool_calls = calls[:1]

        # Ollama occasionally replies in plain text despite required tool_choice.
        # Convert only completed text: truncated output or malformed JSON must
        # remain visible as a parsing error, not masquerade as a successful answer.
        if (
            any(tool.name == 'final_answer' for tool in (tools_to_call_from or []))
            and not message.tool_calls
            and isinstance(message.content, str)
            and message.content.strip()
            and message.raw is not None
            and message.raw.choices[0].finish_reason == 'stop'
        ):
            message.tool_calls = [ChatMessageToolCall(
                id=uuid4().hex,
                type='function',
                function=ChatMessageToolCallFunction(
                    name='final_answer', arguments={'answer': message.content},
                ),
            )]
        return message



# CodeAgent receives tool signatures as Python pseudocode. Unlike the
# ToolCallingAgent's JSON schemas, these omit nested Question fields.
CODE_INSTRUCTIONS = """
Для каждого действия вызывай инструменты как Python-функции внутри блока
```python ... ```, а не описывай предполагаемый вызов словами.
Пример формата для создания квиза:
result = manage_quiz(action="create", title="Название", questions=[
    {"text": "Вопрос?", "options": ["Первый", "Второй"], "correct": 1}
])
print(result)
Поле questions — список словарей, question — один такой словарь.
correct — номер правильного ответа начиная с 1.
После вызова проверь результат: {"error": "..."} означает ошибку,
а подтверждение успеха и ссылки бери только из результата инструмента.
Когда задача выполнена, вызывай final_answer(...).
"""


class ReadyAgent:
    agent_class = ToolCallingAgent

    def __init__(self, tools: BookTools) -> None:
        self.tools = tools
        instructions, self.prompt_version = system_prompt()
        self.model = OllamaToolModel(
            tools,
            model_id=CONFIG['model'],
            api_base=BASE_URL,
            api_key=API_KEY,
            client_kwargs={
                'timeout': CONFIG['timeout_seconds'],
                'max_retries': 0,
                'http_client': httpx.Client(trust_env=False)
            },
            temperature=CONFIG['temperature'],
            max_tokens=CONFIG['max_output_tokens']
        )
        agent_kwargs = dict(
            model=self.model, tools=build_tools(tools),
            instructions=instructions, max_steps=CONFIG['max_steps'],
            verbosity_level=0,
        )
        if self.agent_class is ToolCallingAgent:
            agent_kwargs['max_tool_threads'] = 1
        else:
            # Use native smolagents code parsing with a format Qwen follows
            # more consistently. The default <code> tags are also supported
            # by the library, but often conflict with model-generated Markdown.
            agent_kwargs['executor_type'] = 'local'
            agent_kwargs['code_block_tags'] = 'markdown'
            agent_kwargs['instructions'] = instructions + '\n\n' + CODE_INSTRUCTIONS
        self.agent = self.agent_class(**agent_kwargs)
        self.first = True

    def answer(self, question: str) -> tuple[str, AgentStats]:
        status: TurnStatus
        start = len(self.agent.memory.steps)
        reset, self.first = self.first, False
        try:
            result = self.agent.run(question, reset=reset, return_full_result=True)
            output = str(result.output)
            status = 'success' if result.state == 'success' else 'max_steps'
        except Exception as error:
            output = str(error)
            status = 'timeout' if monotonic() >= self.tools.deadline else 'error'
        steps = self.agent.memory.steps[start:]
        usage = [s.token_usage for s in steps if isinstance(s, ActionStep)]
        complete = bool(usage) and all(u is not None for u in usage)
        return output, AgentStats(
            input_tokens=sum(u.input_tokens for u in usage if u is not None) if complete else None,
            output_tokens=sum(u.output_tokens for u in usage if u is not None) if complete else None,
            steps=len(usage), status=status,
            trimmed=None, max_context=None, context_tokens=None,
        )

    def close(self) -> None:
        self.model.client.close()


class CodingAgent(ReadyAgent):
    """smolagents CodeAgent using the same Ollama connection and quiz tools."""
    agent_class = CodeAgent


def instrument() -> None:
    # Официальная интеграция: LLM, итерации и инструменты → OpenTelemetry → Langfuse.
    SmolagentsInstrumentor().instrument()
