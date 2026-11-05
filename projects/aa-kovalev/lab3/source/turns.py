"""Выполнение и трассировка одной реплики в веб-чате и CLI."""
from time import perf_counter

from langfuse import get_client, propagate_attributes

from settings import CONFIG, cost


def turn(agent, tools, question, session, agent_name, run_id):
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
                answer, stats = str(error), {'status': 'error', 'input_tokens': None, 'output_tokens': None}
            stats['latency_seconds'] = perf_counter() - started
            stats['tool_calls'] = len(tools.calls)
            stats['tool_errors'] = sum('error' in c for c in tools.calls)
            input_tokens = stats['input_tokens']
            output_tokens = stats['output_tokens']
            stats['cost_usd'] = None
            if input_tokens is not None and output_tokens is not None:
                stats['cost_usd'] = cost(input_tokens, output_tokens)
            if stats['status'] != 'success':
                span.update(level='WARNING', status_message=stats['status'])
            elif stats['tool_errors']:
                span.update(level='WARNING', status_message='tool_errors')
            span.update(output=answer, metadata={**stats, 'config': CONFIG})
            span.update_trace(output=answer)
            return {'output': answer, **stats, 'calls': tools.calls.copy(),
                    'retrievals': tools.retrievals.copy(), 'trace_id': span.trace_id,
                    'trace_url': lf.get_trace_url(trace_id=span.trace_id), 'session_id': session}
