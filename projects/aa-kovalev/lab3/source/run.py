"""Прогон обоих агентов или интерактивная демонстрация своего."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4
from dishka import Scope, make_container
from langfuse import get_client, propagate_attributes
from settings import ROOT, CONFIG, cost
from book_tools import BookTools
from dependencies import LabProvider
from agent import Agent
from quizzes import get_quiz, list_quizes
from ready_agent import ReadyAgent, instrument

AGENTS = {'custom': Agent, 'smolagents': ReadyAgent}


def turn(agent, tools, question, session, agent_name, run_id):
    lf = get_client()
    tools.reset()
    tools.session = session
    started = perf_counter()
    with lf.start_as_current_observation(name='user-turn', input=question) as span:
        with propagate_attributes(user_id=CONFIG['user_id'], session_id=session,
                tags=[agent_name, CONFIG['version'], CONFIG['model'], run_id],
                metadata={'run_id': run_id, 'agent': agent_name}):
            span.update_trace(input=question, user_id=CONFIG['user_id'], session_id=session,
                              tags=[agent_name, CONFIG['version'], CONFIG['model'], run_id])
            try:
                answer, stats = agent.answer(question)
            except Exception as error:
                answer, stats = str(error), {'status': 'error', 'input_tokens': None, 'output_tokens': None}
            stats['latency_seconds'] = perf_counter() - started
            stats['tool_calls'] = len(tools.calls)
            stats['tool_errors'] = sum('error' in c for c in tools.calls)
            stats['cost_usd'] = (cost(stats['input_tokens'], stats['output_tokens'])
                if stats['input_tokens'] is not None and stats['output_tokens'] is not None else None)
            if stats['status'] != 'success' or stats['tool_errors']:
                span.update(level='WARNING', status_message=stats['status'] if stats['status'] != 'success' else 'tool_errors')
            span.update(output=answer, metadata={**stats, 'config': CONFIG})
            span.update_trace(output=answer)
            return {'output': answer, **stats, 'calls': tools.calls.copy(),
                    'retrievals': tools.retrievals.copy(), 'trace_id': span.trace_id,
                    'trace_url': lf.get_trace_url(trace_id=span.trace_id), 'session_id': session}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=[*AGENTS, 'both'], default='both')
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--tasks', nargs='+', help='Необязательный список ID для пробного запуска')
    args = parser.parse_args()
    lf = get_client()
    if not lf.auth_check():
        raise SystemExit('Langfuse недоступен: проверьте docker compose и .env')
    instrument()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:6]
    container = make_container(LabProvider(args.index))
    try:
        with container(scope=Scope.SESSION) as batch:
            run_batch(args, run_id, batch)
    finally:
        try:
            container.close()
        finally:
            lf.flush()


def run_batch(args, run_id, batch):
    tools = batch.get(BookTools)
    if args.chat:
        agent_name = 'custom' if args.agent == 'both' else args.agent
        with batch() as dialogue:
            agent = dialogue.get(AGENTS[agent_name])
            while question := input('Вы (пустая строка — выход): ').strip():
                row = turn(agent, tools, question, run_id, agent_name, run_id)
                print(row['output'], '\n', row['trace_url'])
        return
    tasks = json.loads((ROOT / 'tasks.json').read_text(encoding='utf-8'))
    if args.tasks:
        tasks = [t for t in tasks if t['id'] in args.tasks]
        if {t['id'] for t in tasks} != set(args.tasks):
            raise ValueError('Неизвестный ID задания')
    names = list(AGENTS) if args.agent == 'both' else [args.agent]
    folder = ROOT / 'results' / run_id
    folder.mkdir(parents=True)
    (folder / 'metadata.json').write_text(json.dumps({'run_id': run_id, 'config': CONFIG,
        'tasks': tasks, 'agents': names, 'rag_manifest': tools.rag.manifest}, ensure_ascii=False, indent=2), encoding='utf-8')
    # Эталонные чанки вычисляются по тем же PDF-страницам, что в Лабе 2.
    (folder / 'chunks.json').write_text(json.dumps([
        {k: c[k] for k in ('id', 'source_id', 'start_page', 'end_page')}
        for c in tools.chunks]), encoding='utf-8')
    with (folder / 'runs.jsonl').open('w', encoding='utf-8') as output, \
            (folder / 'review.csv').open('w', encoding='utf-8-sig', newline='') as review:
        writer = csv.DictWriter(review, fieldnames=['agent', 'task_id', 'turn', 'input',
            'expected_output', 'output', 'success', 'error_category', 'comment', 'trace_url'])
        writer.writeheader()
        for name in names:
            for task in tasks:
                session = f'{run_id}-{name}-{task["id"]}'
                with batch() as dialogue:
                    agent = dialogue.get(AGENTS[name])
                    for i, question in enumerate(task['input']):
                        row = turn(agent, tools, question, session, name, run_id)
                        row['quizzes'] = [get_quiz(quiz['quiz_id'], session) for quiz in list_quizes(session)]
                        row.update(agent=name, task_id=task['id'], turn=i, input=question,
                            expected_output=task['expected_output'][i], metadata=task['metadata'])
                        output.write(json.dumps(row, ensure_ascii=False) + '\n'); output.flush()
                        writer.writerow({k: row[k] for k in writer.fieldnames if k in row})
                        review.flush()
                        print(name, task['id'], i, row['status'], flush=True)
    print('Результаты:', folder)


if __name__ == '__main__':
    main()
