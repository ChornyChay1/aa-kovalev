"""Запуск эксперимента через OpenAI-совместимый REST API Ollama."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
from time import perf_counter
from uuid import uuid4
from urllib.parse import urlparse

import httpx
import psutil

from config import BASE_URL, MODELS

ROOT = Path(__file__).resolve().parents[1]
MODES = {"A": {}, "B": {"temperature": 0.2, "top_p": 0.8, "max_tokens": 512}}


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def command_output(command: list[str]) -> dict:
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10, encoding="utf-8", errors="replace")
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


class Experiment:
    def __init__(self, base_url: str, timeout: float):
        self.base_url = base_url.rstrip("/")
        self.native_url = self.base_url.removesuffix("/v1")
        self.client = httpx.Client(timeout=timeout, trust_env=False)

    def diagnostic(self, path: str, payload: dict | None = None) -> dict:
        try:
            url = f"{self.native_url}{path}"
            response = self.client.get(url) if payload is None else self.client.post(url, json=payload)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as error:
            return {"error": str(error)}

    def measure_tokenization(self, text: str, model_details: dict) -> dict:
        """Измерить локальный /tokenize активного runner, включая HTTP-задержки."""
        result = {"status": "error", "text": text, "scope": "raw_prompt_without_chat_template", "timing_scope": "runner_http_roundtrip", "repeats": 20}
        try:
            if urlparse(self.base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
                raise ValueError("Поиск runner доступен только для локальной Ollama")
            model_file = next(line[5:].strip().strip('"') for line in model_details.get("modelfile", "").splitlines() if line.startswith("FROM "))
            model_basename = model_file.replace("\\", "/").split("/")[-1]
            runner_url = None
            for process in psutil.process_iter(["name", "cmdline"]):
                try:
                    args = process.info["cmdline"] or []
                    name = (process.info["name"] or "").lower()
                    if not any(part in name for part in ("ollama", "llama")):
                        continue
                    if "--model" not in args or "--port" not in args:
                        continue
                    loaded_file = args[args.index("--model") + 1].replace("\\", "/").split("/")[-1]
                    if loaded_file == model_basename:
                        port = int(args[args.index("--port") + 1])
                        runner_url = f"http://127.0.0.1:{port}/tokenize"
                        break
                except (psutil.Error, ValueError, IndexError):
                    continue
            if runner_url is None:
                raise ValueError("Не найден локальный runner выбранной модели с --model и --port")
            timings = []
            count = None
            for index in range(21):
                started = perf_counter()
                response = self.client.post(runner_url, json={"content": text, "add_special": False})
                response.raise_for_status()
                tokens = response.json()["tokens"]
                elapsed = perf_counter() - started
                if not isinstance(tokens, list):
                    raise ValueError("Runner вернул некорректный список токенов")
                if count is not None and count != len(tokens):
                    raise ValueError("Число токенов меняется между повторами")
                count = len(tokens)
                if index:
                    timings.append(elapsed)
            result.update(status="ok", runner_url=runner_url, token_count=count, samples_s=timings,
                          tokenization_s=statistics.median(timings),
                          tokenization_tokens_per_s=count / statistics.median(timings))
        except (httpx.HTTPError, psutil.Error, ValueError, KeyError, StopIteration) as error:
            result["error"] = str(error) or "В /api/show отсутствует путь FROM к модели"
        return result

    def measure_native(self, model: str, text: str, parameters: dict) -> dict:
        """Отдельная генерация для серверных длительностей, не метрики ответа /v1."""
        options = {("num_predict" if key == "max_tokens" else key): value for key, value in parameters.items()}
        payload = {"model": model, "messages": [{"role": "user", "content": text}], "stream": False}
        if options:
            payload["options"] = options
        started = perf_counter()
        result = {"status": "error", "endpoint": "/api/chat", "request": payload}
        try:
            response = self.client.post(f"{self.native_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            result["response"] = data
            if not data.get("done") or data.get("error"):
                raise ValueError(str(data.get("error", "Генерация не завершена")))
            def seconds(key):
                value = data.get(key)
                return value / 1e9 if isinstance(value, (int, float)) else None
            generation_s = seconds("eval_duration")
            prompt_s = seconds("prompt_eval_duration")
            generated = data.get("eval_count")
            prompt = data.get("prompt_eval_count")
            cached = data.get("prompt_eval_cached_count")
            uncached = max(0, prompt - cached) if prompt is not None and cached is not None else None
            result.update(status="ok", metrics={
                "server_total_s": seconds("total_duration"),
                "server_load_s": seconds("load_duration"),
                "server_prompt_eval_s": prompt_s,
                "server_generation_s": generation_s,
                "prompt_tokens": prompt,
                "cached_prompt_tokens": cached,
                "uncached_prompt_tokens": uncached,
                "completion_tokens": generated,
                "server_generation_tokens_per_s": generated / generation_s if generated is not None and generation_s else None,
                "server_uncached_prompt_tokens_per_s": uncached / prompt_s if uncached is not None and prompt_s else None,
            })
        except (httpx.HTTPError, ValueError) as error:
            result["error"] = str(error)
        result["elapsed_s"] = perf_counter() - started
        return result

    def generate(self, model: str, text: str, parameters: dict) -> dict:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": text}],
            "stream": True,
            "stream_options": {"include_usage": True},
            **parameters,
        }
        record = {"started_at": datetime.now(timezone.utc).isoformat(), "request": payload, "status": "ok"}
        chunks, parts = [], []
        usage, finish_reason = {}, None
        first_text, last_text, headers_time = None, None, None
        done = False
        started = perf_counter()
        try:
            with self.client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as response:
                headers_time = perf_counter() - started
                record["http_status"] = response.status_code
                if response.is_error:
                    response.read()
                    record["error_body"] = response.text
                    response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(data)
                    if chunk.get("error"):
                        raise ValueError(str(chunk["error"]))
                    chunks.append(chunk)
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for choice in chunk.get("choices", []):
                        content = choice.get("delta", {}).get("content")
                        if content:
                            now = perf_counter() - started
                            first_text = now if first_text is None else first_text
                            last_text = now
                            parts.append(content)
                        if choice.get("finish_reason") is not None:
                            finish_reason = choice["finish_reason"]
                if not done or finish_reason is None:
                    raise ValueError("Поток завершился без [DONE] или finish_reason")
        except (httpx.HTTPError, ValueError) as error:
            record.update(status="error", error=str(error))
        elapsed = perf_counter() - started
        text = "".join(parts)
        tokens = usage.get("completion_tokens")
        words = re.findall(r"\w+", text.lower())
        trigrams = list(zip(words, words[1:], words[2:]))
        record.update(
            answer=text,
            finish_reason=finish_reason,
            usage=usage,
            raw_chunks=chunks,
            metrics={
                "elapsed_s": elapsed,
                "response_headers_s": headers_time,
                "first_text_s": first_text,
                "last_text_s": last_text,
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": tokens,
                "total_tokens": usage.get("total_tokens"),
                "cached_prompt_tokens": usage.get("prompt_tokens_details", {}).get("cached_tokens"),
                "output_tokens_per_request_second": tokens / elapsed if tokens is not None and record["status"] == "ok" else None,
                "tokenization_s": None,
                "tokenization_tokens_per_s": None,
                "server_prompt_eval_s": None,
                "server_generation_s": None,
                "server_generation_tokens_per_s": None,
                "answer_chars": len(text),
                "answer_words": len(words),
                "repeated_word_trigram_fraction": (len(trigrams) - len(set(trigrams))) / len(trigrams) if trigrams else 0,
            },
        )
        return record


def save_summary(path: Path, records: list[dict], tokenization_benchmarks: dict) -> None:
    groups = defaultdict(list)
    for record in records:
        groups[(record["model"], record["prompt_id"], record["mode"])].append(record)
    rows = []
    for (model, prompt_id, mode), group in groups.items():
        successful = [record for record in group if record["status"] == "ok"]
        row = {"model": model, "prompt_id": prompt_id, "mode": mode, "runs": len(group), "successful": len(successful)}
        row["unique_answers"] = len({record["answer"] for record in successful})
        row["most_common_answer_fraction"] = max(Counter(record["answer"] for record in successful).values()) / len(successful) if successful else None
        row["truncated_answers"] = sum(record["finish_reason"] == "length" for record in successful)
        tokenizer = tokenization_benchmarks.get(model, {}).get(prompt_id, {})
        row["raw_prompt_tokens"] = tokenizer.get("token_count")
        row["tokenization_http_s_median"] = tokenizer.get("tokenization_s")
        row["tokenization_http_tokens_per_s"] = tokenizer.get("tokenization_tokens_per_s")
        for metric in ("elapsed_s", "first_text_s", "prompt_tokens", "completion_tokens", "output_tokens_per_request_second", "repeated_word_trigram_fraction"):
            values = [record["metrics"][metric] for record in successful if record["metrics"][metric] is not None]
            row[f"{metric}_mean"] = statistics.mean(values) if values else None
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        native = [record["native_measurement"]["metrics"] for record in group if record["native_measurement"]["status"] == "ok"]
        row["native_successful"] = len(native)
        for metric in ("server_generation_tokens_per_s", "server_generation_s", "server_prompt_eval_s", "server_uncached_prompt_tokens_per_s"):
            values = [item[metric] for item in native if item[metric] is not None]
            row[f"native_{metric}_mean"] = statistics.mean(values) if values else None
            row[f"native_{metric}_std"] = statistics.stdev(values) if len(values) > 1 else None
        rows.append(row)
    if rows:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("--repeats и --timeout должны быть положительными")

    prompts = json.loads((ROOT / "prompts.json").read_text(encoding="utf-8"))
    experiment = Experiment(args.base_url, args.timeout)
    records = []
    folder = ROOT / "results" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    folder.mkdir(parents=True)
    metadata = {
        "schema_version": 2,
        "base_url": args.base_url,
        "models": args.models,
        "repeats": args.repeats,
        "timeout_s": args.timeout,
        "modes": MODES,
        "prompts": prompts,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ollama_environment": {name: os.environ.get(name) for name in ("OLLAMA_CONTEXT_LENGTH", "OLLAMA_MAX_LOADED_MODELS", "OLLAMA_NUM_PARALLEL", "OLLAMA_FLASH_ATTENTION", "OLLAMA_KV_CACHE_TYPE")},
        "ollama_version": experiment.diagnostic("/api/version"),
        "ollama_models": experiment.diagnostic("/api/tags"),
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,power.limit", "--format=csv"]),
        "git_revision": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "-C", str(ROOT), "status", "--porcelain"]),
        "model_details": {},
        "tokenization_benchmarks": {},
        "warmups": [],
        "notes": [
            "Основные генерации идут через /v1/chat/completions. Каждый ответ имеет отдельный native_measurement через /api/chat для серверных длительностей.",
            "A не задаёт параметры генерации; B меняет temperature, top_p, max_tokens. Seed и JSON mode не задаются.",
            "first_text_s — клиентское время до первого непустого текстового фрагмента, не точное серверное время до первого токена.",
            "output_tokens_per_request_second включает обработку входа, сеть и служебные задержки; это не чистая скорость генерации.",
            "OpenAI endpoint не отдаёт отдельное время токенизации и серверные длительности; недоступные показатели равны null.",
            "Серверные длительности native_measurement относятся только к его собственному response, а не к основному answer.",
            "Токенизация сырого промпта измеряется через внутренний /tokenize локального runner: медиана 20 запросов после прогрева, включая HTTP и разбор JSON. Chat template и специальные токены не добавляются.",
            "Внутренний API runner может различаться между версиями Ollama; ошибки токенизации сохраняются явно.",
            "Кэш Ollama не очищается. История в каждом запросе новая. Порядок A/B чередуется между повторами.",
            "Переменные окружения относятся к процессу скрипта; настройки сервера проверяются по /api/ps.",
        ],
        "status": "running",
    }
    write_json(folder / "metadata.json", metadata)
    print(f"Результаты: {folder}", flush=True)
    total = len(args.models) * len(prompts) * len(MODES) * args.repeats
    try:
        response = experiment.client.get(f"{experiment.base_url}/models")
        response.raise_for_status()
        available = {item["id"] for item in response.json()["data"]}
        missing = set(args.models) - available
        if missing:
            raise ValueError("Не установлены модели: " + ", ".join(sorted(missing)))
        with (folder / "runs.jsonl").open("w", encoding="utf-8") as output:
            for model in args.models:
                metadata["model_details"][model] = experiment.diagnostic("/api/show", {"model": model})
                warmup = experiment.generate(model, "Ответь одним словом: готов.", {"max_tokens": 16})
                metadata["warmups"].append({"model": model, **warmup})
                write_json(folder / "metadata.json", metadata)
                if warmup["status"] != "ok":
                    raise ValueError(f"Прогрев {model} не удался: {warmup.get('error')}")
                metadata["tokenization_benchmarks"][model] = {}
                for prompt in prompts:
                    tokenization = experiment.measure_tokenization(prompt["text"], metadata["model_details"][model])
                    metadata["tokenization_benchmarks"][model][prompt["id"]] = tokenization
                    write_json(folder / "metadata.json", metadata)
                    if tokenization["status"] != "ok":
                        print(f"Токенизация {model} {prompt['id']}: {tokenization.get('error')}", file=sys.stderr)
                    for repeat in range(1, args.repeats + 1):
                        modes = ("A", "B") if repeat % 2 else ("B", "A")
                        for mode in modes:
                            record = experiment.generate(model, prompt["text"], MODES[mode])
                            record.update(model=model, prompt_id=prompt["id"], mode=mode, repeat=repeat)
                            # Сначала сохранить основной ответ: дополнительное измерение может быть прервано.
                            record["native_measurement"] = {"status": "pending"}
                            line_start = output.tell()
                            output.write(json.dumps(record, ensure_ascii=False) + "\n")
                            output.flush()
                            records.append(record)
                            record["native_measurement"] = experiment.measure_native(model, prompt["text"], MODES[mode])
                            record["ollama_ps_after"] = experiment.diagnostic("/api/ps")
                            # Обновить последнюю строку, сохранив один объект на основной прогон.
                            output.seek(line_start)
                            output.write(json.dumps(record, ensure_ascii=False) + "\n")
                            output.truncate()
                            output.flush()
                            print(f"[{len(records)}/{total}] {model} {prompt['id']} {mode} #{repeat}: {record['status']}, {record['metrics']['elapsed_s']:.2f} с, {record['metrics']['completion_tokens']} токенов", flush=True)
                            if record["status"] != "ok":
                                print(record["error"], file=sys.stderr)
                            if record["native_measurement"]["status"] != "ok":
                                print(f"Серверные измерения: {record['native_measurement'].get('error')}", file=sys.stderr)
                write_json(folder / "metadata.json", metadata)
        complete = all(record["status"] == "ok" and record["native_measurement"]["status"] == "ok" for record in records)
        tokenization_complete = all(item["status"] == "ok" for group in metadata["tokenization_benchmarks"].values() for item in group.values())
        metadata["status"] = "completed" if complete and tokenization_complete else "completed_with_errors"
    except KeyboardInterrupt:
        metadata["status"] = "interrupted"
        print("Остановлено. Завершённые прогоны сохранены.", file=sys.stderr)
    except (httpx.HTTPError, ValueError) as error:
        metadata.update(status="failed", error=str(error))
        print(f"Ошибка: {error}", file=sys.stderr)
    finally:
        experiment.client.close()
        metadata["finished_at"] = datetime.now(timezone.utc).isoformat()
        metadata["saved_runs"] = len(records)
        write_json(folder / "metadata.json", metadata)
        save_summary(folder / "summary.csv", records, metadata["tokenization_benchmarks"])
    return 0 if metadata["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
