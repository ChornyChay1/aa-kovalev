"""Метрики и таблица отчёта после оценки ответов в Langfuse."""
import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean
from typing import Any
from uuid import NAMESPACE_URL, uuid5
from langfuse import Langfuse, get_client, propagate_attributes
from services.observability import require_project
from tool_metrics import check_tools

from settings import LAB2  # Подключает модули Лабы 2.
from retrieval_metrics import metrics, relevant_chunk_ids

def average(values: Iterable[int | float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def retrieval_metrics(row: dict[str, Any], chunks: list[dict[str, Any]], k: int) -> dict[str, float]:
    pages = row['metadata']['relevant_pages'][row['turn']]
    if not pages:
        return {}
    relevant = relevant_chunk_ids(chunks, row['metadata']['source_id'], pages)
    if not relevant:
        raise ValueError(f'В индексе нет эталонных чанков: {row["task_id"]}, {pages}')
    # Как в Лабе 2: фиксированный k и эталоны по PDF-страницам.
    # При нескольких запросах — среднее по запросам; без поиска — нули.
    results = [metrics(hits, relevant, k) for hits in row['retrievals']]
    return {
        name: mean(result[key] for result in results) if results else 0
        for name, key in (('recall@k', 'recall'), ('precision@k', 'precision'), ('mrr', 'mrr'))
    }


def score(lf: Langfuse, trace_id: str, name: str, value: int | float | None) -> None:
    if value is not None:
        lf.create_score(trace_id=trace_id, name=name, value=float(value),
            score_id=str(uuid5(NAMESPACE_URL, trace_id + '/' + name)), data_type='NUMERIC')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--judge-score', default='QWEN-Evaluator',
                        help='Generated Score Name в настройках оценщика Langfuse')
    args = parser.parse_args()
    if args.judge_score == 'success':
        parser.error('Используйте отдельное имя оценки судьи, например QWEN-Evaluator.')
    folder = args.folder
    meta = json.loads((folder / 'metadata.json').read_text(encoding='utf-8'))
    config = meta['config']
    rows = [json.loads(s) for s in (folder / 'runs.jsonl').read_text(encoding='utf-8').splitlines()]
    expected_rows = {(agent, task['id'], i) for agent in meta['agents']
                     for task in meta['tasks'] for i in range(len(task['input']))}
    actual_rows = {(r['agent'], r['task_id'], r['turn']) for r in rows}
    if actual_rows != expected_rows or len(rows) != len(expected_rows):
        raise SystemExit('Прогон неполный или содержит дубликаты: сравнение не строится.')
    chunks = json.loads((folder / 'chunks.json').read_text(encoding='utf-8'))
    lf = get_client()
    require_project()
    missing = []
    external = args.judge_score == 'external_judge'
    for row in rows:
        scores = lf.api.score_v_2.get(
            trace_id=row['trace_id'], name=args.judge_score,
            source=None if external else 'EVAL',
            score_ids=str(uuid5(NAMESPACE_URL, row['trace_id'] + '/external_judge')) if external else None,
            limit=2,
        ).data
        label = f"{row['agent']}/{row['task_id']}/{row['turn']}"
        if not scores:
            missing.append(label)
            continue
        if len(scores) != 1 or scores[0].value not in (0, 1):
            raise SystemExit(f'{label}: нужна одна оценка 0 или 1, получено: '
                             f'{[score.value for score in scores]}.')
        judge = scores[0]
        row['tool_correctness'], missing_tools = check_tools(row)
        row['success'] = int(row['status'] == 'success' and not missing_tools and judge.value == 1)
        row['comment'] = judge.comment or 'См. ответ и вызовы в трейсе'
        if row['status'] != 'success':
            row['error_category'] = row['status']
        elif missing_tools:
            row['error_category'] = 'tool_failure'
            row['comment'] = 'Нет успешного выполнения: ' + ', '.join(missing_tools)
        elif row['success']:
            row['error_category'] = None
        else:
            row['error_category'] = 'judge_rejected'
        row.update(retrieval_metrics(row, chunks, config['retrieval_k']))
    if missing:
        raise SystemExit(f'Нет оценок {args.judge_score} для {len(missing)} реплик. '
                         f'Дождитесь судьи и повторите команду. Примеры: {", ".join(missing[:8])}.')
    tasks = defaultdict(list)
    summaries, failures = [], []
    try:
        for row in rows:
            tasks[row['agent'], row['task_id']].append(row)
            for key in ('success', 'input_tokens', 'output_tokens', 'cost_usd', 'latency_seconds',
                        'steps', 'tool_calls', 'tool_errors', 'max_context', 'tool_correctness',
                        'recall@k', 'precision@k', 'mrr'):
                score(lf, row['trace_id'], key, row.get(key))
            if row['turn'] in row['metadata']['history_dependent']:
                score(lf, row['trace_id'], 'context_success', row['success'])
                if row.get('trimmed_before_answer') is not None:
                    score(lf, row['trace_id'], 'context_success_' + ('after_trim' if row['trimmed_before_answer'] else 'before_trim'), row['success'])
        # Полный диалог успешен, только если успешны все его проверяемые реплики.
        task_rows = []
        for (agent, task_id), turns in tasks.items():
            item = {'agent': agent, 'task_id': task_id, **turns[0]['metadata'],
                'success': int(all(r['success'] for r in turns)),
                'cost_usd': sum(r['cost_usd'] for r in turns) if all(r['cost_usd'] is not None for r in turns) else None,
                'latency_seconds': sum(r['latency_seconds'] for r in turns),
                'input_tokens': sum(r['input_tokens'] for r in turns) if all(r['input_tokens'] is not None for r in turns) else None,
                'output_tokens': sum(r['output_tokens'] for r in turns) if all(r['output_tokens'] is not None for r in turns) else None,
                'error': int(any(r['status'] != 'success' or r['tool_errors'] for r in turns)),
                'max_steps': int(any(r['status'] == 'max_steps' for r in turns))}
            task_rows.append(item)
            for name in ('success', 'cost_usd', 'latency_seconds', 'input_tokens',
                         'output_tokens', 'error', 'max_steps'):
                if item[name] is not None:
                    session_id = turns[0]['session_id']
                    lf.create_score(session_id=session_id, name=f'task_{name}', value=item[name],
                        score_id=str(uuid5(NAMESPACE_URL, f'{session_id}/task_{name}')))
        for agent in sorted({r['agent'] for r in rows}):
            for kind in ('all', 'single', 'dialogue'):
                for difficulty in ('all', 'easy', 'medium', 'hard'):
                    selected = [r for r in task_rows if r['agent'] == agent
                        and (kind == 'all' or r['type'] == kind)
                        and (difficulty == 'all' or r['difficulty'] == difficulty)]
                    if not selected:
                        continue
                    ids = {r['task_id'] for r in selected}
                    turns = [r for r in rows if r['agent'] == agent and r['task_id'] in ids]
                    context = [r for r in turns if r['turn'] in r['metadata']['history_dependent']]
                    summary = {'agent': agent, 'type': kind, 'difficulty': difficulty, 'tasks': len(selected),
                        'success_rate': mean(r['success'] for r in selected),
                        'context_success': average(r['success'] for r in context),
                        'context_before_trim': average(r['success'] for r in context if r.get('trimmed_before_answer') is False),
                        'context_after_trim': average(r['success'] for r in context if r.get('trimmed_before_answer') is True),
                        'mean_cost_usd': average(r['cost_usd'] for r in selected),
                        'cost_coverage': mean(r['cost_usd'] is not None for r in selected),
                        'token_coverage': mean(r['input_tokens'] is not None for r in selected),
                        'mean_latency_seconds': mean(r['latency_seconds'] for r in selected),
                        'mean_input_tokens': average(r['input_tokens'] for r in selected),
                        'mean_output_tokens': average(r['output_tokens'] for r in selected),
                        'error_rate': mean(r['error'] for r in selected),
                        'max_steps_rate': mean(r['max_steps'] for r in selected),
                        'mean_steps_per_turn': average(r.get('steps') for r in turns),
                        'mean_tools_per_turn': mean(r['tool_calls'] for r in turns),
                        'max_context': max((r['max_context'] for r in turns if r.get('max_context') is not None), default=None)}
                    for metric in ('tool_correctness', 'recall@k', 'precision@k', 'mrr'):
                        summary[metric] = average(r.get(metric) for r in turns)
                    summaries.append(summary)
                    with lf.start_as_current_observation(name='run-summary', input={'run_id': meta['run_id'], 'config': config}, output=summary) as span:
                        with propagate_attributes(tags=[agent, config['version'], meta['run_id'], 'aggregate']):
                            span.update_trace(tags=[agent, config['version'], config['model'], meta['run_id'], 'aggregate'])
                            for name, value in summary.items():
                                if isinstance(value, (int, float)):
                                    score(lf, span.trace_id, name, value)
            # Один случай на задачу, а не пять одинаковых ошибок одного диалога.
            seen = set()
            for row in rows:
                if row['agent'] != agent or row['success'] or row['task_id'] in seen:
                    continue
                failures.append(row); seen.add(row['task_id'])
        with (folder / 'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summaries[0])); writer.writeheader(); writer.writerows(summaries)
        lines = ['# Результаты прогона ' + meta['run_id'], '',
            '| Агент | Тип | Сложность | N | Успех | Контекст | Стоимость/задачу, USD | Время/задачу, с |',
            '|---|---|---|---:|---:|---:|---:|---:|']
        for s in summaries:
            values = [s[k] for k in ('agent','type','difficulty','tasks','success_rate','context_success','mean_cost_usd','mean_latency_seconds')]
            lines.append('| ' + ' | '.join('—' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v) for v in values) + ' |')
        lines += ['', '## Разбор ошибок', '']
        for (agent, task_id), turns in tasks.items():
            if (agent == 'custom' and turns[0]['metadata'].get('requires_context_trim')
                    and not any(r.get('trimmed_before_answer')
                                and r['turn'] in r['metadata']['history_dependent'] for r in turns)):
                lines.append(f'- {task_id}: сокращение перед контекстной репликой не подтверждено. '
                             'Для требования длинного диалога нужен отдельный прогон с меньшим бюджетом '
                             'или более длинной историей; шесть реплик сами по себе этого не гарантируют.')
        for r in failures:
            lines.append(f'- **{r["agent"]} / {r["task_id"]} / реплика {r["turn"]}** — {r["error_category"]}: {r["comment"]}. [Трейс]({r["trace_url"]})')
        for agent in sorted({r['agent'] for r in rows}):
            count = sum(r['agent'] == agent for r in failures)
            if count < 5:
                lines.append(f'\n{agent}: найдено только {count} неудачных задач; требование разбора пяти пока не выполнено. Неудачи не подделываем.')
        if len({r['error_category'] for r in failures}) < 3:
            lines.append('\nМенее трёх наблюдавшихся категорий ошибок: требуется дальнейший разбор трейсов.')
        (folder / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        print(folder / 'report.md')
    finally:
        lf.flush()


if __name__ == '__main__':
    main()
