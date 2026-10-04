"""Общие настройки, REST-клиенты и разреженные BM25-векторы."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import re
import httpx

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / 'artifacts'

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def digest(value):
    return hashlib.sha256(value).hexdigest()

def settings(variant='dense_small', path=None):
    config = read(path or ROOT / 'config.json')
    overrides = config.pop('variants')[variant]
    config.update(overrides)
    if config['splitter'] not in ('markdown', 'window'):
        raise ValueError('splitter: markdown или window')
    if not 100 <= config['chunk_size'] <= 1000 or not 0 <= config['overlap'] < config['chunk_size'] // 2:
        raise ValueError('chunk_size: 100–1000; overlap: от 0 до половины размера')
    if config['vectorization'] not in ('dense', 'sparse', 'hybrid'):
        raise ValueError('vectorization: dense, sparse или hybrid')
    return config

class API:
    def __init__(self, config):
        self.config = config
        self.client = httpx.Client(timeout=300, trust_env=False)

    def close(self):
        self.client.close()

    def request(self, service, method, path, body=None):
        url = self.config[service + '_url'].rstrip('/') + path
        response = self.client.request(method, url, json=body)
        response.raise_for_status()
        value = response.json()
        if value.get('error'):
            raise ValueError(str(value['error']))
        return value

    def embed(self, texts):
        response = self.request('ollama', 'POST', '/api/embed', {
            'model': self.config['embedding_model'], 'input': texts, 'truncate': False})
        vectors = response['embeddings']
        if len(vectors) != len(texts) or not vectors or not vectors[0]:
            raise ValueError('Получены неполные эмбеддинги')
        return vectors

def terms(text):
    return re.findall(r'\w+', text.casefold().replace('ё', 'е'))

class BM25:
    """BM25: k1=1.2, b=0.75; IDF включён в вектор документа один раз."""
    def __init__(self, state):
        self.state = state

    @classmethod
    def fit(cls, texts):
        docs = [terms(text) for text in texts]
        frequency = Counter(term for doc in docs for term in set(doc))
        vocabulary = {term: i for i, term in enumerate(sorted(frequency))}
        return cls({'vocabulary': vocabulary, 'idf': {
            term: math.log(1 + (len(docs) - count + 0.5) / (count + 0.5))
            for term, count in frequency.items()},
            'avg_length': sum(map(len, docs)) / len(docs), 'k1': 1.2, 'b': 0.75})

    def vector(self, text, query=False):
        counts = Counter(terms(text))
        vocab, state = self.state['vocabulary'], self.state
        indices, values = [], []
        for term in sorted(counts, key=lambda x: vocab.get(x, -1)):
            if term not in vocab:
                continue
            tf = counts[term]
            value = 1.0 if query else state['idf'][term] * tf * (state['k1'] + 1) / (
                tf + state['k1'] * (1 - state['b'] + state['b'] * sum(counts.values()) / state['avg_length']))
            indices.append(vocab[term]); values.append(value)
        return {'indices': indices, 'values': values}
