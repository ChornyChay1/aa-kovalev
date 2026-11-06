"""Подготовить два runs.jsonl для внешнего судьи."""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def cases(path: Path):
    dialogues = defaultdict(list)
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        if line.strip():
            row = json.loads(line)
            dialogues[row['agent'], row['task_id'], row['session_id']].append(row)

    for turns in dialogues.values():
        history = []
        previous_quizzes = []
        for row in sorted(turns, key=lambda row: row['turn']):
            calls = row['calls']
            quizzes = row.get('quizzes')
            evidence = {
                'question': row['input'],
                'answer': row['output'],
                'expected_output': row['expected_output'],
                'expected_tools': row['metadata']['expected_tools'][row['turn']],
                'expected_actions': (row['metadata']['expected_actions'][row['turn']]
                                     if 'expected_actions' in row['metadata'] else []),
                'status': row['status'],
                'tool_calls': calls,
            }
            if row['turn'] in row['metadata']['history_dependent']:
                evidence['history'] = history.copy()
            if any(call['name'] == 'manage_quiz' for call in calls):
                evidence['quizzes_before'] = previous_quizzes
                evidence['quizzes_after'] = quizzes
            yield {'trace_id': row['trace_id'], 'case': evidence}
            history.append({'question': row['input'], 'answer': row['output']})
            previous_quizzes = quizzes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs=2, type=Path)
    parser.add_argument('-o', '--output', type=Path, default=Path('judge_cases.jsonl'))
    args = parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in args.runs}:
        parser.error('Выходной файл должен отличаться от входных.')

    rows = [row for path in args.runs for row in cases(path)]
    if len({row['trace_id'] for row in rows}) != len(rows):
        parser.error('Во входных файлах есть повторяющиеся trace_id.')

    with args.output.open('w', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{args.output}: {len(rows)} реплик.')


if __name__ == '__main__':
    main()
