"""Пересобрать чанки и dense-векторы, сохранив версию конфигурации."""
import argparse
from pathlib import Path
import json
from time import perf_counter
from langchain_text_splitters import MarkdownTextSplitter
from common import ROOT, ARTIFACTS, API, settings, digest, read, write, get_tokenizer
from prepare import prepare


def split_pages(pages, config):
    tokenizer = get_tokenizer(config)
    chunks = []
    for page in pages:
        text = page['text']
        prefix = page['section'] + '\n\n'
        # Count the complete input, including the heading and BGE special tokens.
        # The splitter cuts strings, so Cyrillic characters remain intact.
        def token_count(body):
            return len(tokenizer.encode(prefix + body, add_special_tokens=True))

        prefix_tokens = token_count('')
        budget = config['chunk_size'] - prefix_tokens
        if budget <= config['overlap']:
            raise ValueError('Заголовок слишком длинный для chunk_size')
        splitter = MarkdownTextSplitter(
            chunk_size=budget, chunk_overlap=config['overlap'],
            length_function=lambda body: len(tokenizer.encode(body, add_special_tokens=True)),
            is_separator_regex=True,
        )
        for body in splitter.split_text(text):
            tokens = token_count(body)
            if tokens > config['chunk_size']:
                raise ValueError('Фрагмент превышает chunk_size с учётом заголовка')
            chunks.append({**{k: v for k, v in page.items() if k != 'text'},
                'id': len(chunks), 'text': prefix + body, 'tokens': tokens})
    return chunks


def build(config, rebuild=False):
    corpus = prepare(config)
    api = API(config)
    try:
        identity = {'config': config, 'corpus': read(corpus / 'manifest.json')['pages_sha256']}
        version = digest(json.dumps(identity, sort_keys=True).encode())[:16]
        folder = ARTIFACTS / 'indexes' / version
        if (folder / 'manifest.json').exists() and not rebuild:
            previous = read(folder / 'manifest.json')
            if ((folder / 'points.json').exists()
                    and digest((folder / 'points.json').read_bytes()) == previous['points_sha256']):
                return folder
        started = perf_counter()
        chunks = split_pages(read(corpus / 'pages.json'), config)
        points = []
        for offset in range(0, len(chunks), 8):
            batch = chunks[offset:offset + 8]
            dense = api.embed([c['text'] for c in batch])
            for i, chunk in enumerate(batch):
                vector = {}
                vector['dense'] = dense[i]
                points.append({'id': chunk['id'], 'payload': chunk, 'vector': vector})
            print(f'Векторы: {min(offset + 8, len(chunks))}/{len(chunks)}', flush=True)
        write(folder / 'points.json', points)
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
