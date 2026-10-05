"""Идемпотентно загрузить версию индекса; --reset очищает только её коллекцию."""
import argparse
from pathlib import Path
from time import perf_counter
from common import API, ARTIFACTS, ROOT, read, write, digest


def load(folder, reset=False):
    folder = Path(folder)
    manifest = read(folder / 'manifest.json')
    config = manifest['config']
    if digest((folder / 'points.json').read_bytes()) != manifest['points_sha256']:
        raise ValueError('Файл векторов изменён; пересоберите индекс')
    points = read(folder / 'points.json')
    api = API(config)
    started = perf_counter()
    path = '/collections/' + manifest['collection']
    try:
        names = {c['name'] for c in api.request('qdrant', 'GET', '/collections')['result']['collections']}
        exists = manifest['collection'] in names
        if exists and reset:
            api.request('qdrant', 'DELETE', path); exists = False
        if not exists:
            dense = points[0]['vector'].get('dense')
            body = {'vectors': {'dense': {'size': len(dense), 'distance': 'Cosine'}} if dense else {},
                'hnsw_config': {'m': config['hnsw']['m'], 'ef_construct': config['hnsw']['ef_construct'],
                                'full_scan_threshold': 10},
                'optimizers_config': {'indexing_threshold': 1}}
            if config['vectorization'] != 'dense':
                body['sparse_vectors'] = {'sparse': {}}
            api.request('qdrant', 'PUT', path, body)
        for offset in range(0, len(points), 64):
            api.request('qdrant', 'PUT', path + '/points?wait=true', {'points': points[offset:offset+64]})
        count = api.request('qdrant', 'POST', path + '/points/count', {'exact': True})['result']['count']
        if count != len(points):
            raise ValueError('Число точек не совпало; повторите с --reset')
        info = api.request('qdrant', 'GET', path)['result']
        write(folder / 'load.json', {'load_s': perf_counter() - started, 'points': count,
            'collection_info': info, 'qdrant_version': api.request('qdrant', 'GET', '/')})
        write(ARTIFACTS / 'active.json', {'index': str(folder.resolve().relative_to(ROOT))})
        return manifest
    finally:
        api.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path, help='Путь artifacts/indexes/<версия> из build_index.py')
    parser.add_argument('--reset', action='store_true')
    args = parser.parse_args()
    print(load(args.index, args.reset)['collection'])
