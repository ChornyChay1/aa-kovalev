"""Готовый smolagents, без изменения его цикла и стратегии памяти."""
from typing import Any

import httpx

from book_tools import BookTools
from contracts import AgentStats, ToolResult, TurnStatus
from smolagents import Tool, ToolCallingAgent, OpenAIServerModel
from smolagents.memory import ActionStep
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from settings import CONFIG, SYSTEM, SCHEMAS, BASE_URL, API_KEY


class BookTool(Tool):
    def __init__(self, schema: dict[str, Any], tools: BookTools) -> None:
        spec = schema['function']
        self.name, self.description = spec['name'], spec['description']
        self.inputs = spec['parameters']['properties']
        self.output_type = 'any'
        self.tools = tools
        super().__init__()


class SearchBookTool(BookTool):
    def forward(self, query: str) -> ToolResult:
        return self.tools.call(self.name, {'query': query})


class ManageQuizTool(BookTool):
    def forward(self, action: str, quiz_id: str | None = None,
                title: str | None = None, questions: list[dict[str, Any]] | None = None,
                question_number: int | None = None, question: dict[str, Any] | None = None) -> ToolResult:
        return self.tools.call(self.name, dict(
            action=action, quiz_id=quiz_id, title=title, questions=questions,
            question_number=question_number, question=question))


class ListQuizesTool(BookTool):
    def forward(self) -> ToolResult:
        return self.tools.call(self.name, {})


class GetQuizInfoTool(BookTool):
    def forward(self, quiz_id: str) -> ToolResult:
        return self.tools.call(self.name, {'quiz_id': quiz_id})


TOOL_CLASSES = {
    'search_book': SearchBookTool,
    'manage_quiz': ManageQuizTool,
    'list_quizes': ListQuizesTool,
    'get_quiz_info': GetQuizInfoTool,
}


class ReadyAgent:
    def __init__(self, tools: BookTools) -> None:
        self.model = OpenAIServerModel(model_id=CONFIG['model'], api_base=BASE_URL,
            api_key=API_KEY, client_kwargs={'timeout': CONFIG['timeout_seconds'], 'max_retries': 0,
                'http_client': httpx.Client(trust_env=False)},
            temperature=CONFIG['temperature'], max_tokens=CONFIG['max_output_tokens'])
        self.agent = ToolCallingAgent(model=self.model,
            tools=[TOOL_CLASSES[schema['function']['name']](schema, tools) for schema in SCHEMAS], instructions=SYSTEM,
            max_steps=CONFIG['max_steps'], verbosity_level=0)
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
            output, status = str(error), 'error'
        steps = self.agent.memory.steps[start:]
        usage = [s.token_usage for s in steps if isinstance(s, ActionStep)]
        complete = all(u is not None for u in usage)
        return output, AgentStats(
            input_tokens=sum(u.input_tokens for u in usage if u is not None) if complete else None,
            output_tokens=sum(u.output_tokens for u in usage if u is not None) if complete else None,
            steps=len(usage), status=status,
            trimmed=None, max_context=None, context_tokens=None,
        )

    def close(self) -> None:
        self.model.client.close()


def instrument() -> None:
    # Официальная интеграция: LLM, итерации и инструменты → OpenTelemetry → Langfuse.
    SmolagentsInstrumentor().instrument()
