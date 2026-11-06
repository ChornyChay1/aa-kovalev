# Лаба 3 — агенты для квизов по Python

Три варианта на **Qwen 2.5 7B / Ollama** и общих инструментах:

| Режим `--agent` | Реализация | История |
|---|---|---|
| `custom` | PydanticAI | `ObservedCompaction(TieredCompaction)` |
| `smolagents` | `ToolCallingAgent` | Штатная `AgentMemory` |
| `coding` | `CodeAgent`, выполнение Python | Штатная `AgentMemory` |

Полное сравнение, таблицы и ошибки: **[REPORT.md](REPORT.md)**. Оценённые полные прогоны есть для исходного `custom`, нового `custom` и `smolagents`; для `coding` полной оценки пока нет.

## Запуск

Из каталога `projects/aa-kovalev/lab3`:

```powershell
uv sync --locked
if (!(Test-Path .env)) { Copy-Item .env.example .env }
ollama pull qwen2.5:7b
docker compose up -d
uv run --locked uvicorn app:app --app-dir source --host 127.0.0.1 --port 8000
```

Нужны работающая Ollama, индекс RAG из [Лабы 2](../lab2/README.md), Qdrant и Langfuse. Веб-чат: http://127.0.0.1:8000; Langfuse: http://localhost:3000. Квизы — `quizzes.sqlite3`, диалоги — `dialogues.sqlite3`. Настройки подключений — `.env`; модель и лимиты — `config.json`. Для сопоставимых экспериментов выставляйте одинаковый `num_ctx` Ollama.

## Архитектура

| Инструмент | Назначение |
|---|---|
| `search_book` | RAG-поиск по учебнику с номером PDF-страницы |
| `list_quizes` | Названия, ссылки и ID квизов |
| `get_quiz_info` | Чтение квиза / вопроса без ключа ответа |
| `manage_quiz` | Создание, удаление, добавление и изменение вопросов |

Сигнатуры и описания: `source/tools/definitions.py`; Pydantic-модель вопроса: `source/models/schemas.py`; системный промпт: `prompts/system.txt`. PydanticAI формирует схемы из типов; для `smolagents` вложенный `Question` адаптируется и валидируется в `source/tools/smolagents_tools.py`. Для `CodeAgent` дополнительно настроены Markdown-блоки Python и пример вложенного вопроса в промпте. Поиск возвращает суммарно до 4 000 символов; результаты и ошибки инструментов трассируются в Langfuse.

### Контекст `custom`

В `source/agents/custom.py` работает `ObservedCompaction` — собственная обёртка над библиотечным `TieredCompaction`:

1. Отделяет старые ходы от текущего; **текущий ход не сокращает**.
2. Применяет `ClearToolResults → SummarizingCompaction → SlidingWindowCompaction` к старой истории по необходимости. Фокус сводки задан в `prompts/summary_focus.txt`.
3. Если бюджет всё ещё превышен, удаляет старые завершённые ходы; пишет `context-trim` и `context-check` в Langfuse.
4. Сохраняет историю и после неуспешных вызовов через `capture_run_messages` и `repair_messages`.

Параметры текущей версии: **17 288** токенов контекста, **13 960** — порог сокращения, **1024** — максимум ответа; до **8** запросов/шагов, **4** вызовов инструментов и **120 с** на реплику. Суммаризация использует ту же LLM и не гарантирует сохранность ID; при потере ID следует пользоваться `list_quizes`. `smolagents` и `coding` не используют автоматическое сокращение истории.

## Эксперимент и оценка

`tasks.json`: **18 задач / 39 реплик** на агента, включая зависимые от истории. Состояние каждого квиза проверяется по сохранённым снимкам. Трейсы имеют теги агента и идентификатор сессии.

```powershell
# Небольшой пробный прогон
uv run --locked python source/run.py --agent custom --tasks S01 S07 D01

# Полный запуск и публикация в Langfuse Datasets
uv run --locked python source/run.py --agent custom --publish-dataset
uv run --locked python source/run.py --agent smolagents --publish-dataset
uv run --locked python source/run.py --agent coding --publish-dataset

# Варианты: both = custom + smolagents; all = все три
uv run --locked python source/run.py --agent all --tasks S01 D01 L01
```

Результаты сохраняются в `results/<run_id>/runs.jsonl`. Список задач и эталонов публикуется в `lab3-agent-turns` через `--publish-dataset` или отдельно `source/load_dataset.py`.

Для внешней оценки: `source/export_judge.py` создаёт кейсы; файл оценок `external_judge` импортируется через `source/import_judge.py`; `source/evaluate.py` считает метрики и формирует `summary.csv` + `report.md`. Нужные поля оценки: `id`, `traceId`, `name="external_judge"`, `value` (0/1), `dataType="NUMERIC"`, `comment`. **Успех задания** означает выполнение всех его реплик; `Tool correctness` не проверяет правильность содержимого квиза.

```powershell
uv run --locked python source/export_judge.py ".\results\RUN_ID\runs.jsonl" -o judge_cases.jsonl
uv run --locked python source/import_judge.py .\langfuse_external_scores.jsonl
uv run --locked python source/evaluate.py ".\results\20261106T115324Z-41891f" --judge-score external_judge
```

Для пересчёта двух прежних прогонов используйте также `20261106T091347Z-f13a68` (`custom`) и `20261106T092905Z-775cec` (`smolagents`). Импорт оценки в трейс **не** публикует автоматически прогон в Dataset.

## Результаты для защиты

| Метрика | Старый `custom` | Новый `custom` | `smolagents` |
|---|---:|---:|---:|
| **Успех реплик** | 11/39 (28,2%) | **18/39 (46,2%)** | — |
| **Успех целых задач** | 4/18 (22,2%) | **4/18 (22,2%)** | 0/18 |
| Успех контекстных реплик | 20% | **40%** | 0% |
| Tool correctness | 61,5% | **79,5%** | 74,4% |
| Recall@5 / Precision@5 / MRR | 0,500 / 0,350 / 0,820 | **0,632 / 0,417 / 1,000** | 0,620 / 0,410 / 0,960 |
| Среднее время задачи | 49,61 с | **82,73 с** | 72,85 с |

Новая конфигурация повысила успешность отдельных реплик на **18 п.п.**, но не улучшила долю полностью решённых задач и увеличила среднее время на **66,8%**. Среди 14 проваленных заданий нового прогона: 8 содержательных отклонений, 5 пропущенных обязательных инструментов, 1 таймаут. Подробности и ссылки на трейсы — в [REPORT.md](REPORT.md).

**Оговорка:** одновременно с контекстом менялись бюджет токенов и другие части кода; нельзя доказать причинный вклад только суммаризации без отдельной абляции.

## Проверка перед защитой

```powershell
$env:PYTHONPATH = 'source'
uv run --locked python -m unittest discover -s tests -v
```

Для демонстрации: создайте и измените квиз в веб-чате, откройте его трейс; покажите `search_book`/`manage_quiz`, затем `context-trim`/`context-check` в длинной сессии. Сопоставьте `REPORT.md` с `run-summary` в Langfuse. На коротком диалоге сокращение контекста может не запускаться.
