"""Пересобрать чанки и dense-векторы, сохранив версию конфигурации."""
import argparse
from pathlib import Path
import json
from time import perf_counter
from langchain_text_splitters import MarkdownHeaderTextSplitter, MarkdownTextSplitter, RecursiveCharacterTextSplitter
from common import ROOT, ARTIFACTS, API, settings, digest, read, write, get_tokenizer
from prepare import prepare


def split_pages(pages, config):
    chunk_size, overlap = config['chunk_size'], config['overlap']
    include_headings, splitter_type = config['include_headings'], config['splitter']
    tokenizer = get_tokenizer(config)
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=True))
    headers = [('#' * level, '#' * level) for level in range(1, 7)]
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers, strip_headers=True)
    splitter_class = {'markdown': MarkdownTextSplitter, 'recursive': RecursiveCharacterTextSplitter}[splitter_type]
    chunks = []
    for page in pages:
        for section in header_splitter.split_text(page['text']):
            headings = section.metadata
            heading_text = '\n'.join(f'{mark} {headings[mark]}' for mark, _ in headers if mark in headings)
            prefix = (heading_text or page['section']) + '\n\n' if include_headings else ''
            budget = chunk_size - count_tokens(prefix)
            if budget <= overlap:
                raise ValueError('Заголовки слишком длинные для chunk_size и overlap')
            splitter = splitter_class(
                chunk_size=budget, chunk_overlap=overlap,
                length_function=count_tokens, is_separator_regex=True)
            for body in splitter.split_text(section.page_content):
                text = prefix + body
                tokens = count_tokens(text)
                if tokens > chunk_size:
                    raise ValueError('Фрагмент превышает chunk_size с учётом заголовков')
                chunks.append({**{k: v for k, v in page.items() if k != 'text'},
                    'id': len(chunks), 'text': text, 'tokens': tokens, 'headings': headings})
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
        write(folder / 'chunk_example.json', chunks[0])
        print('Пример чанка:', folder / 'chunk_example.json', flush=True)
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
