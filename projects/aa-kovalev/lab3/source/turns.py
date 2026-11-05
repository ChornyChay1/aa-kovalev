"""Выполнение и трассировка одной реплики в веб-чате и CLI."""
from time import perf_counter

from langfuse import get_client, propagate_attributes

from book_tools import BookTools
from contracts import AgentStats, DialogueAgent, TurnMetrics, TurnResult
from observability import trace_url
from settings import CONFIG, cost


def turn(agent: DialogueAgent, tools: BookTools, question: str, session: str,
         agent_name: str, run_id: str) -> TurnResult:
    lf = get_client()
    tools.reset()
    tools.session = session
    started = perf_counter()
    tags = [agent_name, CONFIG['version'], CONFIG['model'], run_id]
    with lf.start_as_current_observation(name='user-turn', input=question) as span:
        with propagate_attributes(user_id=CONFIG['user_id'], session_id=session,
                tags=tags,
                metadata={'run_id': run_id, 'agent': agent_name}):
            span.update_trace(input=question, user_id=CONFIG['user_id'], session_id=session,
                              tags=tags)
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
            span.update_trace(output=answer)
            return TurnResult(
                **metadata, output=answer, calls=tools.calls.copy(),
                retrievals=tools.retrievals.copy(), trace_id=span.trace_id,
                trace_url=trace_url(span.trace_id), session_id=session,
            )
