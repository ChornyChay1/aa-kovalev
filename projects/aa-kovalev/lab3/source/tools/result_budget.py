"""Ограничение ответов инструментов в токенах с сохранением JSON и идентификаторов."""
import json
from copy import deepcopy
from functools import lru_cache

from transformers import AutoTokenizer

from settings import CONFIG

TEXT_KEYS = frozenset({'text', 'message', 'hint', 'description', 'error'})


@lru_cache(maxsize=1)
def tokenizer():
    return AutoTokenizer.from_pretrained(CONFIG['tokenizer'])


def count_text(text: str) -> int:
    return len(tokenizer().encode(text, add_special_tokens=False))


def count_result(value) -> int:
    return count_text(json.dumps(value, ensure_ascii=False, separators=(',', ':')))


def _text_prefix(text: str, token_limit: int) -> str:
    """Границы Unicode, но бюджет измеряется исключительно в токенах."""
    if count_text(text) <= token_limit:
        return text
    low, high = 0, len(text)
    while low < high:
        mid = (low + high + 1) // 2
        if count_text(text[:mid]) <= token_limit:
            low = mid
        else:
            high = mid - 1
    return text[:low]


def _cap_texts(value, token_limit: int, key: str = ''):
    if isinstance(value, str) and key in TEXT_KEYS:
        return _text_prefix(value, token_limit)
    if isinstance(value, dict):
        return {k: _cap_texts(v, token_limit, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_cap_texts(item, token_limit, key) for item in value]
    return value


def _text_sizes(value, key: str = '') -> list[int]:
    if isinstance(value, str) and key in TEXT_KEYS:
        return [count_text(value)]
    if isinstance(value, dict):
        return [n for k, v in value.items() for n in _text_sizes(v, k)]
    if isinstance(value, list):
        return [n for item in value for n in _text_sizes(item, key)]
    return []


def _mark(value, omitted: int = 0):
    if isinstance(value, dict):
        value['_truncated'] = True
        if omitted:
            value['_omitted_items'] = omitted
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                item['truncated'] = True
        if value and omitted and isinstance(value[-1], dict):
            value[-1]['omitted_items'] = omitted
    return value


def _best_text_fit(value, budget: int):
    sizes = _text_sizes(value)
    if not sizes:
        return value if count_result(value) <= budget else None
    low, high = 0, max(sizes)
    best = None
    while low <= high:
        mid = (low + high) // 2
        candidate = _cap_texts(value, mid)
        if count_result(candidate) <= budget:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def limit_result(result, budget: int):
    """Возвращает (структурированный результат, усечён ли он).

    Не меняет входной объект. При сокращении сохраняет ID/URL, помечает
    результат и выбрасывает явную ошибку, если даже обязательные поля
    не помещаются. Никогда не отдаёт модели половину JSON-строки.
    """
    if budget < 1:
        raise ValueError('Лимит результата инструмента исчерпан')
    if count_result(result) <= budget:
        return result, False

    if isinstance(result, list):
        field, total = None, len(result)
    elif isinstance(result, dict):
        field = next((k for k in ('questions', 'quizzes') if isinstance(result.get(k), list)), None)
        total = len(result[field]) if field else 0
    else:
        raise TypeError('Результат инструмента должен быть объектом или списком')

    def candidate(keep: int):
        value = deepcopy(result)
        omitted = total - keep
        if field is not None:
            value[field] = value[field][:keep]
        elif isinstance(value, list):
            value = value[:keep]
        return _mark(value, omitted)

    # Сначала сохраняем все элементы и сокращаем только свободный текст.
    full = _best_text_fit(candidate(total), budget)
    if full is not None:
        return full, True

    # Если служебные поля и элементы не помещаются — убираем хвост списка,
    # затем используем освобождённое место под текст оставшихся элементов.
    if total:
        low, high = 0, total - 1
        best = None
        while low <= high:
            mid = (low + high) // 2
            fitted = _best_text_fit(candidate(mid), budget)
            if fitted is not None:
                best = fitted
                low = mid + 1
            else:
                high = mid - 1
        if best is not None:
            return best, True

    # Критичные идентификаторы операций записи не удаляем молча.
    if isinstance(result, dict):
        minimal = {'_truncated': True}
        for k in ('quiz_id', 'url', 'title', 'deleted', 'updated_question', 'question_count'):
            if k in result:
                minimal[k] = result[k]
        minimal['message'] = 'Ответ инструмента превышает токенный лимит.'
        fitted = _best_text_fit(minimal, budget)
        if fitted is not None:
            return fitted, True
    else:
        minimal = [{'truncated': True, 'omitted_items': len(result)}]
        if count_result(minimal) <= budget:
            return minimal, True
    raise ValueError('Метаданные результата инструмента не помещаются в токенный бюджет')
