"""Прогон обоих агентов или интерактивная демонстрация своего."""
import argparse
import csv
import json
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import NAMESPACE_URL, uuid4, uuid5
from langfuse import get_client
from langfuse.api.resources.dataset_run_items.types.create_dataset_run_item_request import (
    CreateDatasetRunItemRequest,
)
from services.observability import project_id, require_project
from settings import ROOT, CONFIG
from tools.book import BookTools
from agents.custom import Agent
from services.quizzes import get_quiz, list_quizes
from agents.ready import ReadyAgent, instrument
from services.turns import turn

from constants import BOTH_AGENTS, CUSTOM_AGENT, DATASET_NAME, READY_AGENT

AGENTS: dict[str, type[Agent] | type[ReadyAgent]] = {CUSTOM_AGENT: Agent, READY_AGENT: ReadyAgent}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=[*AGENTS, BOTH_AGENTS], default=BOTH_AGENTS)
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--index', type=Path)
    parser.add_argument('--publish-dataset', action='store_true',
                        help='Связать прогон с датасетом для оценки в Langfuse')
    parser.add_argument('--tasks', nargs='+', help='Необязательный список ID для пробного запуска')
    args = parser.parse_args()
    lf = get_client()
    require_project()
    instrument()
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid4().hex[:6]
    try:
        with closing(BookTools(args.index)) as tools:
            run_batch(args, run_id, tools)
    finally:
        lf.flush()


def run_batch(args: argparse.Namespace, run_id: str, tools: BookTools) -> None:
    if args.chat:
        agent_name = CUSTOM_AGENT if args.agent == BOTH_AGENTS else args.agent
        with closing(AGENTS[agent_name](tools)) as agent:
            while question := input('Вы (пустая строка — выход): ').strip():
                reply = turn(agent, tools, question, run_id, agent_name, run_id)
                print(reply.output, '\n', reply.trace_url)
        return
    tasks = json.loads((ROOT / 'tasks.json').read_text(encoding='utf-8'))
    if args.tasks:
        tasks = [t for t in tasks if t['id'] in args.tasks]
        if {t['id'] for t in tasks} != set(args.tasks):
            raise ValueError('Неизвестный ID задания')
    items = publish_dataset(tasks) if args.publish_dataset else {}
    names = list(AGENTS) if args.agent == BOTH_AGENTS else [args.agent]
    folder = ROOT / 'results' / run_id
    folder.mkdir(parents=True)
    (folder / 'metadata.json').write_text(json.dumps({'run_id': run_id, 'config': CONFIG,
        'tasks': tasks, 'agents': names, 'rag_manifest': tools.rag.manifest}, ensure_ascii=False, indent=2), encoding='utf-8')
    # Эталонные чанки вычисляются по тем же PDF-страницам, что в Лабе 2.
    points = json.loads((tools.rag.folder / 'points.json').read_text(encoding='utf-8'))
    chunks = [{key: point['payload'][key] for key in ('id', 'source_id', 'start_page', 'end_page')}
              for point in points]
    (folder / 'chunks.json').write_text(json.dumps(chunks), encoding='utf-8')
    links: list[CreateDatasetRunItemRequest] = []
    with (folder / 'runs.jsonl').open('w', encoding='utf-8') as output, \
            (folder / 'review.csv').open('w', encoding='utf-8-sig', newline='') as review:
        writer = csv.DictWriter(review, fieldnames=['agent', 'task_id', 'turn', 'input',
            'expected_output', 'output', 'success', 'error_category', 'comment', 'trace_url'])
        writer.writeheader()
        for name in names:
            for task in tasks:
                session = f'{run_id}-{name}-{task["id"]}'
                with closing(AGENTS[name](tools)) as agent:
                    for i, question in enumerate(task['input']):
                        row = turn(agent, tools, question, session, name, run_id).model_dump(exclude_unset=True)
                        row['quizzes'] = [get_quiz(quiz['quiz_id'], session) for quiz in list_quizes(session)]
                        row.update(agent=name, task_id=task['id'], turn=i, input=question,
                            expected_output=task['expected_output'][i], metadata=task['metadata'])
                        if items:
                            links.append(CreateDatasetRunItemRequest(
                                run_name=f'{run_id}-{name}', dataset_item_id=items[task['id'], i],
                                trace_id=row['trace_id'], observation_id=row['observation_id'],
                            ))
                        output.write(json.dumps(row, ensure_ascii=False) + '\n')
                        output.flush()
                        writer.writerow({k: row[k] for k in writer.fieldnames if k in row})
                        review.flush()
                        print(name, task['id'], i, row['status'], flush=True)
    print('Результаты:', folder)
    if links:
        lf = get_client()
        lf.flush()
        for link in links:
            lf.api.dataset_run_items.create(request=link)
        print('Прогоны опубликованы в датасете Langfuse:', DATASET_NAME)


def publish_dataset(tasks: list[dict]) -> dict[tuple[str, int], str]:
    lf = get_client()
    dataset = lf.create_dataset(name=DATASET_NAME, description='Проверяемые реплики Лабы 3')
    items = {}
    for task in tasks:
        for i, question in enumerate(task['input']):
            expected = task['expected_output'][i]
            item_input = {'question': question, 'previous_user_turns': task['input'][:i]}
            identity = json.dumps([project_id(), dataset.id, task['id'], i, item_input, expected],
                                  ensure_ascii=False, sort_keys=True)
            item = lf.create_dataset_item(
                dataset_name=DATASET_NAME, id=str(uuid5(NAMESPACE_URL, identity)),
                input=item_input, expected_output=expected,
                metadata={'task_id': task['id'], 'turn': i, 'type': task['metadata']['type'],
                          'difficulty': task['metadata']['difficulty'],
                          'history_dependent': i in task['metadata']['history_dependent']},
            )
            items[task['id'], i] = item.id
    return items


if __name__ == '__main__':
    main()
