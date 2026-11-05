"""Общие настройки; пути не зависят от рабочей директории."""
import json
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LAB2 = ROOT.parent / 'lab2'
load_dotenv(ROOT / '.env')
CONFIG = json.loads((ROOT / 'config.json').read_text(encoding='utf-8'))
SYSTEM = (ROOT / 'prompts/system.txt').read_text(encoding='utf-8')
SCHEMAS = json.loads((ROOT / 'prompts/tools.json').read_text(encoding='utf-8'))
BASE_URL = os.getenv('LLM_BASE_URL', 'http://localhost:11434/v1')
API_KEY = os.getenv('LLM_API_KEY', 'ollama')


def cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * CONFIG['input_usd_per_million']
            + output_tokens * CONFIG['output_usd_per_million']) / 1_000_000
