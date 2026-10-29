"""Пересобрать чанки и dense-векторы, сохранив версию конфигурации."""
import argparse
from pathlib import Path
import json
import re
from bisect import bisect_right
from time import perf_counter
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from common import CONFIG, EMBEDDING_MODEL, ROOT, ARTIFACTS, API, settings, digest, read, write, get_tokenizer
from prepare import prepare

CHUNK_SIZE = CONFIG['chunk_size']
OVERLAP = CONFIG['overlap']
INCLUDE_HEADINGS = CONFIG['include_headings']
PAGE_MARKER = re.compile(r'(?m)^<!-- PAGE:(\d+) -->[ \t]*$')


def split_markdown(markdown, manifest, *, chunk_size=CHUNK_SIZE, overlap=OVERLAP,
                   include_headings=INCLUDE_HEADINGS, embedding_model=EMBEDDING_MODEL):
    tokenizer = get_tokenizer(embedding_model)
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=True))
    headers = [('#' * level, '#' * level) for level in range(1, 7)]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=not include_headings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap,
        length_function=count_tokens, add_start_index=True)

    page_number = manifest['selected_pages'][0]
    chunks = []
    for section in header_splitter.split_text(markdown):
        positions, numbers = [0], [page_number]
        removed = 0
        for marker in PAGE_MARKER.finditer(section.page_content):
            positions.append(marker.start() - removed)
            numbers.append(int(marker.group(1)))
            removed += len(marker.group())
        page_number = numbers[-1]
        text = PAGE_MARKER.sub('', section.page_content)
        for chunk in text_splitter.split_documents([Document(page_content=text, metadata=section.metadata)]):
            start = chunk.metadata.pop('start_index')
            if start < 0:
                raise ValueError('Не удалось определить позицию чанка')
            start_page = numbers[bisect_right(positions, start) - 1]
            end_page = numbers[bisect_right(positions, start + len(chunk.page_content) - 1) - 1]
            tokens = count_tokens(chunk.page_content)
            if tokens > chunk_size:
                raise ValueError('Фрагмент превышает chunk_size')
            source = manifest['source']
            chunks.append({'id': len(chunks), 'source_id': 'byte_python_ru',
                'title': source['title'], 'source_url': source['url'], 'source_date': source['date'],
                'path': 'book.md', 'page': start_page, 'section': manifest['sections'][str(start_page)],
                'text': chunk.page_content, 'tokens': tokens, 'headings': chunk.metadata,
                'start_page': start_page, 'end_page': end_page})
    return chunks

def build(config, rebuild=False):
    # Explicit overrides keep experiment variants and --config independent of module defaults.
    split_options = {key: config[key] for key in
                     ('chunk_size', 'overlap', 'include_headings', 'embedding_model')}
    corpus = prepare(config)
    api = API(config)
    try:
        manifest = read(corpus / 'manifest.json')
        identity = {'config': config, 'corpus': digest((corpus / 'manifest.json').read_bytes()),
                    'splitter_sha256': digest(Path(__file__).read_bytes())}
        version = digest(json.dumps(identity, sort_keys=True).encode())[:16]
        folder = ARTIFACTS / 'indexes' / version
        if (folder / 'manifest.json').exists() and not rebuild:
            previous = read(folder / 'manifest.json')
            if ((folder / 'points.json').exists()
                    and digest((folder / 'points.json').read_bytes()) == previous['points_sha256']):
                return folder
        started = perf_counter()
        chunks = split_markdown((corpus / 'book.md').read_text(encoding='utf-8'), manifest, **split_options)
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
