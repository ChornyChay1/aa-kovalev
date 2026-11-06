"""Выполнение и трассировка одной реплики в веб-чате и CLI."""
from time import perf_counter

from langfuse import get_client, propagate_attributes

from tools.book import BookTools
from models.domain import DialogueAgent
from models.schemas import AgentStats, TurnMetrics, TurnResult
from services.observability import trace_url
from settings import CONFIG, cost


def turn(
        agent: DialogueAgent,
        tools: BookTools,
        question: str,
        session: str,
        agent_name: str,
        run_id: str,
        *, quiz_collection: str | None = None
) -> TurnResult:
    lf = get_client()
    tools.reset()
    tools.session = quiz_collection if quiz_collection is not None else session
    started = perf_counter()
    
    tags = [agent_name, CONFIG['version'], CONFIG['model'], run_id]
    trace_name = f'{agent_name}-turn'
    
    with lf.start_as_current_observation(name=trace_name, input=question) as span:
        with propagate_attributes(
                user_id=CONFIG['user_id'], session_id=session,
                tags=tags, trace_name=trace_name, version=CONFIG['version'],
                metadata={'run_id': run_id, 'agent': agent_name, 'system_prompt': agent.prompt_version}
        ):
            try:
                answer, stats = agent.answer(question)
            except Exception as error:
                answer = str(error)
                stats = AgentStats(status='error', input_tokens=None, output_tokens=None)
            
            latency = perf_counter() - started
            tool_errors = sum(call.error is not None for call in tools.calls)
            input_tokens = stats.input_tokens
            output_tokens = stats.output_tokens
            cost_usd = None
            if input_tokens is not None and output_tokens is not None:
                cost_usd = cost(input_tokens, output_tokens)
            if stats.status != 'success':
                span.update(level='WARNING', status_message=stats.status)
            elif tool_errors:
                span.update(level='WARNING', status_message='tool_errors')
            metrics = TurnMetrics(
                latency_seconds=latency, tool_calls=len(tools.calls),
                tool_errors=tool_errors, cost_usd=cost_usd,
            )
            metadata = {
                **stats.model_dump(exclude_unset=True),
                **metrics.model_dump(),
            }
            span.update(output=answer, metadata={**metadata, 'config': CONFIG})
            # Trace-level I/O is also used by older hosted evaluators.
            set_trace_io = getattr(span, 'set_trace_io', None) or span.update_trace
            set_trace_io(input=question, output=answer)
            return TurnResult(
                **metadata, output=answer, calls=tools.calls.copy(),
                retrievals=tools.retrievals.copy(), trace_id=span.trace_id, observation_id=span.id,
                trace_url=trace_url(span.trace_id), session_id=session,
            )
