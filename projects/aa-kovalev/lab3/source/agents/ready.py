"""Готовый smolagents, без изменения его цикла и стратегии памяти."""
import httpx

from models.domain import TurnStatus
from models.schemas import AgentStats
from tools.book import BookTools
from tools.smolagents_tools import build_tools
from smolagents import ToolCallingAgent, OpenAIServerModel
from smolagents.memory import ActionStep
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from services.prompts import system_prompt
from settings import API_KEY, BASE_URL, CONFIG


class ReadyAgent:
    def __init__(self, tools: BookTools) -> None:
        instructions, self.prompt_version = system_prompt()
        self.model = OpenAIServerModel(model_id=CONFIG['model'], api_base=BASE_URL,
            api_key=API_KEY, client_kwargs={'timeout': CONFIG['timeout_seconds'], 'max_retries': 0,
                'http_client': httpx.Client(trust_env=False)},
            temperature=CONFIG['temperature'], max_tokens=CONFIG['max_output_tokens'])
        self.agent = ToolCallingAgent(model=self.model,
            tools=build_tools(tools), instructions=instructions,
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
