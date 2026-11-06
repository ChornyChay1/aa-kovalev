"""Загрузка и проверка разметки задач до запуска моделей."""
import json
from typing import get_args

from constants import MANAGE_QUIZ, TOOL_NAMES, QuizAction
from settings import ROOT


def load_tasks() -> list[dict]:
    tasks = json.loads((ROOT / 'tasks.json').read_text(encoding='utf-8'))
    ids = set()
    for task in tasks:
        label = task['id']
        if label in ids:
            raise ValueError(f'Повторяется ID задания: {label}')
        ids.add(label)
        size = len(task['input'])
        meta = task['metadata']
        if not size or any(len(values) != size for values in (
            task['expected_output'], meta['expected_tools'], meta['expected_actions'], meta['relevant_pages'],
        )):
            raise ValueError(f'{label}: разметка должна соответствовать числу реплик')
        if not set(meta['history_dependent']) <= set(range(1, size)):
            raise ValueError(f'{label}: неверные номера контекстных реплик')
        for tools, actions in zip(meta['expected_tools'], meta['expected_actions']):
            if not set(tools) <= TOOL_NAMES or not set(actions) <= set(get_args(QuizAction)):
                raise ValueError(f'{label}: неизвестный инструмент или действие')
            if actions and MANAGE_QUIZ not in tools:
                raise ValueError(f'{label}: действия требуют manage_quiz')
    return tasks
