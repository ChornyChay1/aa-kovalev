"""Загрузка промптов из Langfuse с локальными резервными копиями."""
import logging
from functools import lru_cache

from langfuse import get_client

from settings import LANGFUSE_PROMPT_NAME, ROOT, SYSTEM

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def system_prompt() -> tuple[str, str]:
    if not LANGFUSE_PROMPT_NAME:
        return SYSTEM, 'local'
    prompt = get_client().get_prompt(
        LANGFUSE_PROMPT_NAME, type='text', fallback=SYSTEM, max_retries=0,
    )
    if prompt.is_fallback:
        return SYSTEM, 'local'
    return prompt.prompt, f'{prompt.name}@{prompt.version}'


@lru_cache(maxsize=1)
def summary_focus() -> str:
    """Langfuse summary_focus в приоритете, локальный файл — резерв."""
    path = ROOT / 'prompts/summary_focus.txt'
    local = path.read_text(encoding='utf-8') if path.is_file() else None
    try:
        prompt = get_client().get_prompt(
            'summary_focus', type='text', fallback=local, max_retries=0,
        )
    except Exception as error:
        if local is None:
            raise
        logger.warning('Не удалось загрузить summary_focus из Langfuse: %s', error)
        return local
    if prompt.is_fallback:
        if local is None:
            raise ValueError('summary_focus отсутствует в Langfuse и локально')
        return local
    return prompt.prompt
