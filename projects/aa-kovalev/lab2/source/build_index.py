"""Пересобрать чанки и dense/sparse-векторы, сохранив версию конфигурации."""
import argparse
from pathlib import Path
import json
import re
from time import perf_counter
import tiktoken
from common import ROOT, ARTIFACTS, API, BM25, settings, digest, read, write
from prepare import prepare


def split_pages(pages, config):
    tokenizer = tiktoken.get_encoding(config['tokenizer'])
    chunks = []
    for page in pages:
        sections = re.split(r'(?m)(?=^#{1,3} )', page['text']) if config['splitter'] == 'markdown' else [page['text']]
        for section in sections:
            if not config['include_headings']:
                section = re.sub(r'(?m)^#{1,6} .*\n?', '', section)
            if not section.strip():
                continue
            prefix = (page['section'] + '\n\n') if config['include_headings'] else ''
            # Use Unicode character boundaries, measuring every candidate in tokens.
            # This avoids corrupting Cyrillic characters at token-byte boundaries.
            budget = config['chunk_size'] - len(tokenizer.encode(prefix))
            if budget < 32:
                raise ValueError('Заголовок слишком длинный для chunk_size')
            start = 0
            while start < len(section):
                low, high = start + 1, len(section)
                while low < high:
                    middle = (low + high + 1) // 2
                    if len(tokenizer.encode(prefix + section[start:middle])) <= config['chunk_size']:
                        low = middle
                    else:
                        high = middle - 1
                end = low
                text = prefix + section[start:end]
                tokens = len(tokenizer.encode(text))
                assert tokens <= config['chunk_size']
                chunks.append({**{k: v for k, v in page.items() if k != 'text'},
                    'id': len(chunks), 'text': text, 'tokens': tokens})
                if end == len(section):
                    break
                low, high = start + 1, end
                while low < high:
                    middle = (low + high) // 2
                    if len(tokenizer.encode(section[middle:end])) <= config['overlap']:
                        high = middle
                    else:
                        low = middle + 1
                start = low
    return chunks


def build(config, rebuild=False):
    corpus = prepare(config)
    api = API(config)
    try:
        model = None
        if config['vectorization'] != 'sparse':
            models = api.request('ollama', 'GET', '/api/tags')['models']
            model = next((m for m in models if m['name'] == config['embedding_model']), None)
            if model is None:
                raise ValueError('Сначала ollama pull ' + config['embedding_model'])
        identity = {'config': config, 'corpus': read(corpus / 'manifest.json')['pages_sha256'],
            'embedding_digest': model['digest'] if model else None,
            'code_sha256': digest(b''.join((ROOT / 'source' / name).read_bytes()
                for name in ['common.py', 'prepare.py', 'build_index.py'])),
            'lock_sha256': digest((ROOT / 'uv.lock').read_bytes())}
        version = digest(json.dumps(identity, sort_keys=True).encode())[:16]
        folder = ARTIFACTS / 'indexes' / version
        if (folder / 'manifest.json').exists() and not rebuild:
            previous = read(folder / 'manifest.json')
            if ((folder / 'points.json').exists() and (folder / 'bm25.json').exists()
                    and digest((folder / 'points.json').read_bytes()) == previous['points_sha256']):
                return folder
        started = perf_counter()
        chunks = split_pages(read(corpus / 'pages.json'), config)
        texts = [c['text'] for c in chunks]
        bm25 = BM25.fit(texts)
        points = []
        for offset in range(0, len(chunks), 8):
            batch = chunks[offset:offset + 8]
            dense = api.embed([c['text'] for c in batch]) if config['vectorization'] != 'sparse' else None
            for i, chunk in enumerate(batch):
                vector = {}
                if dense is not None:
                    vector['dense'] = dense[i]
                if config['vectorization'] != 'dense':
                    vector['sparse'] = bm25.vector(chunk['text'])
                points.append({'id': chunk['id'], 'payload': chunk, 'vector': vector})
            print(f'Векторы: {min(offset + 8, len(chunks))}/{len(chunks)}', flush=True)
        write(folder / 'points.json', points)
        write(folder / 'bm25.json', bm25.state)
        write(folder / 'manifest.json', {**identity, 'version': version, 'collection': 'lab2_' + version,
            'chunks': len(chunks), 'corpus_path': str(corpus.relative_to(ROOT)),
            'build_s': perf_counter() - started, 'points_sha256': digest((folder / 'points.json').read_bytes())})
        return folder
    finally:
        api.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', default='dense_small')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    print(build(settings(args.variant, args.config), args.rebuild))
