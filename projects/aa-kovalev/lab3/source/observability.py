"""Проект и ссылки Langfuse через публичный Projects API."""
import logging
import os
from functools import lru_cache

import httpx

logger = logging.getLogger(__name__)


def base_url() -> str:
    return (os.getenv('LANGFUSE_BASE_URL') or os.getenv('LANGFUSE_HOST')
            or 'https://cloud.langfuse.com').rstrip('/')


@lru_cache(maxsize=1)
def project_id() -> str:
    public_key = os.getenv('LANGFUSE_PUBLIC_KEY')
    secret_key = os.getenv('LANGFUSE_SECRET_KEY')
    if not public_key or not secret_key:
        raise ValueError('Задайте ключи Langfuse в .env.')
    response = httpx.get(
        f'{base_url()}/api/public/projects', auth=(public_key, secret_key), timeout=5,
    )
    response.raise_for_status()
    projects = response.json().get('data', [])
    if not projects or not projects[0].get('id'):
        raise ValueError('Для этих ключей Langfuse не найден проект.')
    return str(projects[0]['id'])


def trace_url(trace_id: str) -> str | None:
    if not trace_id or not trace_id.strip('0'):
        return None
    try:
        return f'{base_url()}/project/{project_id()}/traces/{trace_id}'
    except (httpx.HTTPError, ValueError) as error:
        logger.warning('Не удалось получить ссылку Langfuse: %s', error)
        return None


def require_project() -> None:
    try:
        project_id()
    except (httpx.HTTPError, ValueError) as error:
        raise SystemExit(f'Langfuse недоступен: {error}') from error
