"""Загрузка полного набора реплик в Langfuse без запуска агентов."""
import json
from uuid import NAMESPACE_URL, uuid5

from langfuse import get_client

from constants import DATASET_NAME
from services.observability import project_id
from settings import ROOT


def publish_dataset(tasks: list[dict]) -> dict[tuple[str, int], str]:
    lf = get_client()
    dataset = lf.create_dataset(name=DATASET_NAME, description='Проверяемые реплики Лабы 3')
    items = {}
    for task in tasks:
        for i, question in enumerate(task['input']):
            expected = task['expected_output'][i]
            item_input = {'question': question, 'previous_user_turns': task['input'][:i]}
            task_metadata = task['metadata']
            metadata = {
                'task_id': task['id'], 'turn': i, 'type': task_metadata['type'],
                'difficulty': task_metadata['difficulty'],
                'history_dependent': i in task_metadata['history_dependent'],
                'expected_tools': task_metadata['expected_tools'][i],
                'relevant_pages': task_metadata['relevant_pages'][i],
                'source_id': task_metadata['source_id'],
            }
            identity = json.dumps([project_id(), dataset.id, item_input, expected, metadata],
                                  ensure_ascii=False, sort_keys=True)
            item = lf.create_dataset_item(
                dataset_name=DATASET_NAME, id=str(uuid5(NAMESPACE_URL, identity)),
                input=item_input, expected_output=expected,
                metadata=metadata,
            )
            items[task['id'], i] = item.id
    return items


if __name__ == '__main__':
    items = publish_dataset(json.loads((ROOT / 'tasks.json').read_text(encoding='utf-8')))
    print(f'{DATASET_NAME}: загружено {len(items)} реплик.')
