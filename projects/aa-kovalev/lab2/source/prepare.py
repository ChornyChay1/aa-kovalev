"""Скачать проверенный PDF и сохранить Markdown с метаданными по страницам."""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
import httpx
import pymupdf
import pymupdf4llm
from common import ARTIFACTS, settings, digest, read, write


def prepare(config, pdf=None, rebuild=False):
    source = config['source']
    folder = ARTIFACTS / 'corpus' / source['sha256'][:12]
    manifest_path = folder / 'manifest.json'
    parser_hash = digest(Path(__file__).read_bytes())
    if manifest_path.exists() and not rebuild:
        manifest = read(manifest_path)
        if (manifest.get('source') == source and manifest.get('parser_sha256') == parser_hash
                and (folder / 'book.pdf').exists() and (folder / 'pages.json').exists()
                and digest((folder / 'pages.json').read_bytes()) == manifest['pages_sha256']):
            return folder
    folder.mkdir(parents=True, exist_ok=True)
    if pdf:
        content = Path(pdf).read_bytes()
    elif (folder / 'book.pdf').exists():
        content = (folder / 'book.pdf').read_bytes()
    else:
        response = httpx.get(source['url'], follow_redirects=True, timeout=120)
        response.raise_for_status()
        content = response.content
    if digest(content) != source['sha256']:
        raise ValueError('SHA256 PDF не совпал с конфигурацией; нужна исходная версия 2.02')
    (folder / 'book.pdf').write_bytes(content)
    doc = pymupdf.open(folder / 'book.pdf')
    start, end = source['first_page'], source['last_page']
    if not 1 <= start <= end <= len(doc):
        raise ValueError('Неверный диапазон PDF')
    toc = doc.get_toc()
    pages = pymupdf4llm.to_markdown(doc, pages=list(range(start - 1, end)),
        page_chunks=True, show_progress=False, margins=(0, 40, 0, 40))
    records = []
    for page_number, page in zip(range(start, end + 1), pages, strict=True):
        sections = []
        for level, title, number in toc:
            if number <= page_number:
                sections = sections[:level-1] + [title]
        text = page['text'].replace('\r\n', '\n')
        text = re.sub(r'(?m)^A Byte of Python \(Russian\), Версия 2\.02\s*$', '', text)
        text = re.sub(r'(?m)^#{4,6} ', '### ', text)
        text = text.replace('\ufffd', '').replace('\uffff', '')  # PDF continuation marker artifacts
        text = re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'
        record = {'source_id': 'byte_python_ru', 'title': source['title'],
            'page': page_number, 'section': ' / '.join(sections), 'source_url': source['url'],
            'source_date': source['date'], 'path': f'pages/page_{page_number:03}.md', 'text': text}
        target = folder / record['path']; target.parent.mkdir(exist_ok=True)
        target.write_text('---\n' + '\n'.join(f'{key}: {json.dumps(value, ensure_ascii=False)}'
            for key, value in record.items() if key != 'text') + '\n---\n\n' + text, encoding='utf-8')
        records.append(record)
    nonempty = sum(len(p['text'].strip()) >= 100 for p in records)
    if nonempty < 50:
        raise ValueError(f'Недостаточно страниц текста: {nonempty}')
    write(folder / 'pages.json', records)
    write(manifest_path, {'source': source, 'parser_sha256': parser_hash, 'pdf_pages': len(doc), 'text_pages': nonempty,
        'selected_pages': [start, end], 'prepared_at': datetime.now(timezone.utc).isoformat(),
        'pages_sha256': digest((folder / 'pages.json').read_bytes())})
    doc.close()
    return folder

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--pdf', type=Path)
    parser.add_argument('--rebuild', action='store_true')
    args = parser.parse_args()
    print(prepare(settings(path=args.config), args.pdf, args.rebuild))
