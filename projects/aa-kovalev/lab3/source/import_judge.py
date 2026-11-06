"""Импорт оценок внешнего судьи в существующие трейсы Langfuse."""
import argparse
import json
import os
from pathlib import Path

import httpx

from services.observability import base_url
from settings import ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scores', type=Path, help='Файл langfuse_external_scores.jsonl')
    args = parser.parse_args()

    try:
        rows = [json.loads(line) for line in args.scores.read_text(encoding='utf-8-sig').splitlines()
                if line.strip()]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not rows:
        parser.error('Файл оценок пуст.')

    trace_ids = set()
    score_ids = set()
    for index, row in enumerate(rows, 1):
        if (
            not isinstance(row, dict) or row.get('name') != 'external_judge'
            or row.get('dataType') != 'NUMERIC' or type(row.get('value')) is not int
            or row['value'] not in (0, 1)
            or not all(isinstance(row.get(key), str) and row[key] for key in ('id', 'traceId', 'comment'))
        ):
            parser.error(f'Запись {index}: неверный формат оценки.')
        if row['id'] in score_ids or row['traceId'] in trace_ids:
            parser.error(f'Запись {index}: повторный идентификатор оценки или трейса.')
        score_ids.add(row['id'])
        trace_ids.add(row['traceId'])

    public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    secret_key = os.getenv('LANGFUSE_SECRET_KEY')
    if not public_key or not secret_key:
        parser.error(f'Задайте ключи Langfuse в {ROOT / ".env"}.')

    with httpx.Client(base_url=base_url(), auth=(public_key, secret_key), timeout=20) as client:
        for row in rows:
            trace_id = row['traceId']
            try:
                client.get(f'/api/public/traces/{trace_id}').raise_for_status()
            except httpx.HTTPError as error:
                raise SystemExit(f'Не удалось проверить трейс {trace_id}. Импорт не начат. {error}') from error
        for index, row in enumerate(rows, 1):
            try:
                client.post('/api/public/scores', json=row).raise_for_status()
            except httpx.HTTPError as error:
                raise SystemExit(
                    f'Ошибка импорта записи {index}/{len(rows)} (трейс {row["traceId"]}). '
                    f'После устранения ошибки повторите команду с тем же файлом. {error}'
                ) from error
    print(f'Импортировано оценок external_judge: {len(rows)}.')


if __name__ == '__main__':
    main()
