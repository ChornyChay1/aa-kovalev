"""Общие настройки, REST-клиенты."""
from functools import lru_cache
from transformers import AutoTokenizer
from pathlib import Path
import hashlib
import json
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
    if not 100 <= config['chunk_size'] <= 1000 or not 0 <= config['overlap'] < config['chunk_size'] // 2:
        raise ValueError('chunk_size: 100–1000; overlap: от 0 до половины размера')
    if config['vectorization'] not in ('dense',):
        raise ValueError('vectorization: dense')
    return config

# Ollama's BGE-M3 and the chunk splitter use the same model vocabulary.
EMBEDDING_TOKENIZERS = {'bge-m3': 'BAAI/bge-m3', 'bge-m3:567m': 'BAAI/bge-m3',
                        'bge-m3:latest': 'BAAI/bge-m3'}

@lru_cache(maxsize=None)
def _load_tokenizer(name):
    return AutoTokenizer.from_pretrained(name)

def get_tokenizer(config):
    expected = EMBEDDING_TOKENIZERS.get(config['embedding_model'])
    if expected is None:
        raise ValueError('Неизвестный токенизатор для модели ' + config['embedding_model'])
    return _load_tokenizer(expected)

class API:
    def __init__(self, config):
        self.config = config
        self.client = httpx.Client(timeout=300, trust_env=False)

    def close(self):
        self.client.close()

    def request(self, service, method, path, body=None):
        url = self.config[service + '_url'].rstrip('/') + path
        response = self.client.request(method, url, json=body)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            raise httpx.HTTPStatusError(
                f'{error}\nОтвет {service}: {response.text}',
                request=error.request, response=error.response) from error
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
