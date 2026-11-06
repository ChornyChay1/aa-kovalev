"""Проверка обязательных инструментов и действий без привязки к агенту."""
from constants import MANAGE_QUIZ


def check_tools(row: dict) -> tuple[int | None, list[str]]:
    turn = row['turn']
    meta = row['metadata']
    expected = set(meta['expected_tools'][turn])
    # Старые прогоны считаются по сохранённой в них разметке.
    actions = set(meta['expected_actions'][turn]) if 'expected_actions' in meta else set()
    calls = row['calls']
    called = {call['name'] for call in calls}
    called_actions = {call['arguments'].get('action') for call in calls if call['name'] == MANAGE_QUIZ}
    successful = [call for call in calls if call.get('error') is None
                  and call.get('result') is not None
                  and not (isinstance(call['result'], dict) and 'error' in call['result'])]
    completed = {call['name'] for call in successful}
    completed_actions = {call['arguments'].get('action') for call in successful if call['name'] == MANAGE_QUIZ}
    missing = sorted(expected - completed)
    missing += [f'{MANAGE_QUIZ}:{action}' for action in sorted(actions - completed_actions)]
    correctness = int(expected <= called and actions <= called_actions) if expected or actions else None
    return correctness, missing
