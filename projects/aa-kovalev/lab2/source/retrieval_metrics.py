"""Общие метрики поиска по эталонным страницам учебника."""
from collections.abc import Mapping, Sequence
from typing import Any


def relevant_chunk_ids(
    chunks: Sequence[Mapping[str, Any]], source_id: str, pages: Sequence[int],
) -> set[int]:
    return {
        chunk['id'] for chunk in chunks
        if chunk['source_id'] == source_id
        and any(chunk['start_page'] <= page <= chunk['end_page'] for page in pages)
    }


def metrics(hits: Sequence[Mapping[str, Any]], relevant: set[int], k: int) -> dict[str, float]:
    found = [hit['id'] in relevant for hit in hits[:k]]
    return {'recall': sum(found) / len(relevant), 'precision': sum(found) / k,
            'mrr': next((1 / (i + 1) for i, yes in enumerate(found) if yes), 0)}
