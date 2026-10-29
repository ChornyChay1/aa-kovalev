"""Пересобрать чанки и dense-векторы, сохранив версию конфигурации."""
import argparse
from pathlib import Path
import json
from time import perf_counter
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from common import CONFIG, EMBEDDING_MODEL, ROOT, ARTIFACTS, API, settings, digest, read, write, get_tokenizer
from prepare import prepare

CHUNK_SIZE = CONFIG['chunk_size']
OVERLAP = CONFIG['overlap']
INCLUDE_HEADINGS = CONFIG['include_headings']


def split_pages(pages, *, chunk_size=CHUNK_SIZE, overlap=OVERLAP,
                include_headings=INCLUDE_HEADINGS, embedding_model=EMBEDDING_MODEL):
    tokenizer = get_tokenizer(embedding_model)
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=True))
    headers = [('#' * level, '#' * level) for level in range(1, 7)]
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers, strip_headers=not include_headings)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=overlap, length_function=count_tokens)

    chunks = []
    for page in pages:
        sections = header_splitter.split_text(page['text'])
        for section in text_splitter.split_documents(sections):
            text = section.page_content
            tokens = count_tokens(text)
            if tokens > chunk_size:
                raise ValueError('Фрагмент превышает chunk_size')
            chunks.append({**{k: v for k, v in page.items() if k != 'text'},
                'id': len(chunks), 'text': text, 'tokens': tokens, 'headings': section.metadata})
    return chunks

def build(config, rebuild=False):
    # Explicit overrides keep experiment variants and --config independent of module defaults.
    split_options = {key: config[key] for key in
                     ('chunk_size', 'overlap', 'include_headings', 'embedding_model')}
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
        chunks = split_pages(read(corpus / 'pages.json'), **split_options)
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
