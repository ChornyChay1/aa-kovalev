"""Проверка Ollama через OpenAI-совместимый REST; не часть эксперимента."""

import argparse
import sys
from time import perf_counter

import httpx

from config import BASE_URL, MODELS


class OllamaClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def check(self, model: str) -> None:
        with httpx.Client(timeout=180, trust_env=False) as client:
            response = client.get(f"{self.base_url}/models")
            response.raise_for_status()
            available = {item["id"] for item in response.json()["data"]}
            if model not in available:
                raise ValueError(f"Модель {model} не установлена. Выполните: ollama pull {model}")

            started = perf_counter()
            response = client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "Поздоровайся одним предложением на русском."}],
                },
            )
            response.raise_for_status()
            elapsed = perf_counter() - started
            result = response.json()

        print(f"Модель: {model}")
        print(result["choices"][0]["message"]["content"])
        print(f"Полное время запроса (включая возможную загрузку): {elapsed:.2f} с")
        print(f"Токены: {result.get('usage', {})}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, default=MODELS[0])
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    try:
        OllamaClient(args.base_url).check(args.model)
    except (httpx.HTTPError, ValueError) as error:
        print(f"Ошибка проверки: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
