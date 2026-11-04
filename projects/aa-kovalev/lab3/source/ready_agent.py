"""Готовый smolagents, без изменения его цикла и стратегии памяти."""
import httpx
from smolagents import Tool, ToolCallingAgent, OpenAIServerModel
from smolagents.memory import ActionStep
from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from settings import CONFIG, SYSTEM, SCHEMAS, BASE_URL, API_KEY


class BookTool(Tool):
    skip_forward_signature_validation = True
    # Имя, схема и описание читаются из того же файла, что у своего агента.
    def __init__(self, schema, tools):
        spec = schema['function']
        self.name, self.description = spec['name'], spec['description']
        self.inputs = spec['parameters']['properties']
        self.output_type = 'any'
        self.tools = tools
        super().__init__()

    def forward(self, **kwargs):
        return self.tools.call(self.name, kwargs)


class ReadyAgent:
    def __init__(self, tools):
        self.model = OpenAIServerModel(model_id=CONFIG['model'], api_base=BASE_URL,
            api_key=API_KEY, client_kwargs={'timeout': CONFIG['timeout_seconds'], 'max_retries': 0,
                'http_client': httpx.Client(trust_env=False)},
            temperature=CONFIG['temperature'], max_tokens=CONFIG['max_output_tokens'])
        self.agent = ToolCallingAgent(model=self.model,
            tools=[BookTool(schema, tools) for schema in SCHEMAS], instructions=SYSTEM,
            max_steps=CONFIG['max_steps'], verbosity_level=0)
        self.first = True

    def answer(self, question):
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
        return output, {
            'input_tokens': sum(u.input_tokens for u in usage) if complete else None,
            'output_tokens': sum(u.output_tokens for u in usage) if complete else None,
            'steps': sum(isinstance(s, ActionStep) for s in steps),
            'status': status,
            'trimmed': None, 'max_context': None, 'context_tokens': None,
        }

    def close(self):
        self.model.client.close()


def instrument():
    # Официальная интеграция: LLM, итерации и инструменты → OpenTelemetry → Langfuse.
    SmolagentsInstrumentor().instrument()
