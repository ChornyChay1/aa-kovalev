"""Системный промпт из репозитория или Langfuse Prompt Management."""
from functools import lru_cache

from langfuse import get_client

from settings import LANGFUSE_PROMPT_NAME, SYSTEM


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
