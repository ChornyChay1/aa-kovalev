"""Сравнить конфигурации по Recall/Precision/MRR и сохранить ответы для проверки QA."""
import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from time import perf_counter
from uuid import uuid4
from common import ROOT, ARTIFACTS, read, write, settings
from build_index import build
from load_to_vector_store import load
from rag import RAG


def metrics(pages, relevant, k):
    found = [page in relevant for page in pages[:k]]
    return {'recall': sum(found) / len(relevant), 'precision': sum(found) / k,
            'mrr': next((1 / (i + 1) for i, yes in enumerate(found) if yes), 0)}


def evaluate(args):
    questions = read(ROOT / 'questions.json')
    output = ROOT / 'results' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:6])
    output.mkdir(parents=True)
    summaries, reviews = [], []
    meta = {'status': 'running', 'variants': args.variants, 'k': [5, 10], 'questions': questions,
            'qa_enabled': not args.retrieval_only, 'configs': {}}
    write(output / 'metadata.json', meta)
    try:
        with (output / 'runs.jsonl').open('w', encoding='utf-8') as log:
            for variant in args.variants:
                started = perf_counter()
                folder = build(settings(variant, args.config), args.rebuild)
                manifest = load(folder, args.reset)
                meta['configs'][variant] = manifest
                write(output / 'metadata.json', meta)
                rag = RAG(folder)
                records = []
                try:
                    # Retrieve all questions before QA to avoid repeated model swapping.
                    retrieved = [(q, *rag.retrieve(q['question'], 10)) for q in questions]
                    for q, hits, elapsed in retrieved:
                        row = {'variant': variant, 'id': q['id'], 'question': q['question'],
                            'expected_answer': q['answer'], 'relevant_pages': q['pages'],
                            'hits': hits, 'retrieval_s': elapsed,
                            'metrics': {str(k): metrics([h['page'] for h in hits], set(q['pages']), k) for k in [5, 10]}}
                        if not args.retrieval_only:
                            row.update(rag.answer(q['question'], hits))
                            reviews.append({'variant': variant, 'id': q['id'], 'question': q['question'],
                                'expected_answer': q['answer'], 'answer': row['answer'],
                                'answer_correct_0_or_1': '', 'citations_support_answer_0_or_1': '', 'note': ''})
                        log.write(json.dumps(row, ensure_ascii=False) + '\n'); log.flush()
                        records.append(row)
                        print(variant, q['id'], f'{elapsed:.3f} с', flush=True)
                finally:
                    rag.close()
                summary = {'variant': variant, 'index': str(folder.relative_to(ROOT)), 'chunks': manifest['chunks'],
                    'index_build_s': manifest['build_s'], 'load_s': read(folder / 'load.json')['load_s'],
                    'index_bytes': (folder / 'points.json').stat().st_size,
                    'retrieval_s_mean': mean(r['retrieval_s'] for r in records),
                    'qa_s_mean': mean(r.get('generation_s', 0) for r in records),
                    'run_s': perf_counter() - started}
                for k in [5, 10]:
                    for metric in ['recall', 'precision', 'mrr']:
                        summary[f'{metric}@{k}'] = mean(r['metrics'][str(k)][metric] for r in records)
                summaries.append(summary)
        best = max(summaries, key=lambda s: (s['mrr@10'], s['recall@5'], -s['retrieval_s_mean']))
        write(ARTIFACTS / 'active.json', {'index': best['index']})
        meta.update(status='completed', best_retrieval_variant=best['variant'],
            selection='Максимум MRR@10, затем Recall@5, затем минимум retrieval_s; QA проверяется вручную')
    except BaseException as error:
        meta.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
        raise
    finally:
        for name, rows in [('summary.csv', summaries), ('qa_review.csv', reviews)]:
            if rows:
                with (output / name).open('w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        write(output / 'metadata.json', meta)
        print('Результаты:', output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--variants', nargs='+', default=['dense_small', 'dense_large', 'hybrid_large'])
    parser.add_argument('--retrieval-only', action='store_true')
    parser.add_argument('--rebuild', action='store_true')
    parser.add_argument('--reset', action='store_true')
    evaluate(parser.parse_args())
