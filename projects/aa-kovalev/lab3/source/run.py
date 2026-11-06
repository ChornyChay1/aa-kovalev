"""Прогон custom, smolagents ToolCallingAgent или CodeAgent."""
import argparse
import json
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from langfuse import get_client
from langfuse.api.resources.dataset_run_items.types.create_dataset_run_item_request import (
    CreateDatasetRunItemRequest,
)
from services.observability import require_project
from load_dataset import publish_dataset
from settings import ROOT, CONFIG
from tools.book import BookTools
from agents.custom import Agent
from services.quizzes import get_quiz, list_quizes
from agents.ready import CodingAgent, ReadyAgent, instrument
from services.turns import turn
from task_data import load_tasks

from constants import ALL_AGENTS, BOTH_AGENTS, CODING_AGENT, CUSTOM_AGENT, DATASET_NAME, READY_AGENT

AGENTS: dict[str, type[Agent] | type[ReadyAgent]] = {
    CUSTOM_AGENT: Agent, READY_AGENT: ReadyAgent, CODING_AGENT: CodingAgent,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=[*AGENTS, BOTH_AGENTS, ALL_AGENTS], default=BOTH_AGENTS)
    parser.add_argument('--chat', action='store_true')
    parser.add_argument('--index', type=Path)
    parser.add_argument(
        '--publish-dataset', action='store_true', help='Связать прогон с датасетом для оценки в Langfuse'
    )
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
        agent_name = CUSTOM_AGENT if args.agent in (BOTH_AGENTS, ALL_AGENTS) else args.agent
        with closing(AGENTS[agent_name](tools)) as agent:
            while question := input('Вы (пустая строка — выход): ').strip():
                reply = turn(agent, tools, question, run_id, agent_name, run_id)
                print(reply.output, '\n', reply.trace_url)
        return
    
    tasks = load_tasks()
    if args.tasks:
        tasks = [t for t in tasks if t['id'] in args.tasks]
        if {t['id'] for t in tasks} != set(args.tasks):
            raise ValueError('Неизвестный ID задания')
    
    items = publish_dataset(tasks) if args.publish_dataset else {}
    names = ([CUSTOM_AGENT, READY_AGENT] if args.agent == BOTH_AGENTS
             else list(AGENTS) if args.agent == ALL_AGENTS else [args.agent])
    
    folder = ROOT / 'results' / run_id
    folder.mkdir(parents=True)
    (folder / 'metadata.json').write_text(
        json.dumps(
            {
                'run_id': run_id,
                'config': CONFIG,
                'tasks': tasks,
                'agents': names,
                'rag_manifest': tools.rag.manifest},
            ensure_ascii=False, indent=2
        ),
        encoding='utf-8'
    )
    # Эталонные чанки вычисляются по тем же PDF-страницам, что в Лабе 2.
    points = json.loads((tools.rag.folder / 'points.json').read_text(encoding='utf-8'))
    chunks = [
        {key: point['payload'][key] for key in ('id', 'source_id', 'start_page', 'end_page')}for point in points
    ]
    (folder / 'chunks.json').write_text(json.dumps(chunks), encoding='utf-8')
    
    links: list[CreateDatasetRunItemRequest] = []
    with (folder / 'runs.jsonl').open('w', encoding='utf-8') as output:
        for name in names:
            for task in tasks:
                session = f'{run_id}-{name}-{task["id"]}'
                
                with closing(AGENTS[name](tools)) as agent:
                    for i, question in enumerate(task['input']):
                        #вот тут вызывается агент
                        row = turn(agent, tools, question, session, name, run_id).model_dump(exclude_unset=True)
                        
                        row['quizzes'] = [get_quiz(quiz['quiz_id'], session) for quiz in list_quizes(session)]
                        row.update(
                            agent=name,
                            task_id=task['id'],
                            turn=i,
                            input=question,
                            expected_output=task['expected_output'][i],
                            metadata=task['metadata']
                        )
                        if items:
                            links.append(
                                CreateDatasetRunItemRequest(
                                    run_name=f'{run_id}-{name}',
                                    dataset_item_id=items[task['id'], i],
                                    trace_id=row['trace_id'], observation_id=row['observation_id'],
                                    metadata={
                                        'agent': name,
                                        'config': CONFIG,
                                        'rubric_version': task['metadata']['rubric_version']},
                                    )
                            )
                        output.write(json.dumps(row, ensure_ascii=False) + '\n')
                        output.flush()
                        print(name, task['id'], i, row['status'], flush=True)

    print('Результаты:', folder)
    if links:
        lf = get_client()
        lf.flush()
        for link in links:
            lf.api.dataset_run_items.create(request=link)
        
        print('Прогоны опубликованы в датасете Langfuse:', DATASET_NAME)


if __name__ == '__main__':
    main()
