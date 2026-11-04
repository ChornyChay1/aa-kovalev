"""Собственный цикл native tool calling с ограниченной историей."""
import copy
import json
import httpx
from time import monotonic
from openai import OpenAI
from langfuse import get_client
from transformers import AutoTokenizer
from settings import CONFIG, SYSTEM, SCHEMAS, BASE_URL, API_KEY, cost

MIN_TOOL_TOKENS = 32
TRUNCATED = '\n[Результат сокращён из-за лимита контекста]'


class Agent:
    def __init__(self, tools):
        self.tools = tools
        self.client = OpenAI(base_url=BASE_URL, api_key=API_KEY, max_retries=0, http_client=httpx.Client(trust_env=False))
        self.tokenizer = AutoTokenizer.from_pretrained(CONFIG['tokenizer'])
        self.history = [{'role': 'system', 'content': SYSTEM}]
        self.trimmed = False

    def count(self):
        messages = copy.deepcopy(self.history)
        # HF-шаблон Qwen принимает arguments как объект, OpenAI API — как JSON-строку.
        for message in messages:
            for call in message.get('tool_calls', []):
                try:
                    call['function']['arguments'] = json.loads(call['function']['arguments'])
                except (ValueError, TypeError):
                    pass
        return len(self.tokenizer.apply_chat_template(messages, tools=SCHEMAS,
                   tokenize=True, add_generation_prompt=True))

    def fit_context(self, stats):
        limit = CONFIG['context_tokens'] - CONFIG['max_output_tokens'] - CONFIG['context_reserve_tokens']
        before = self.count()
        stats['max_context_before'] = max(stats['max_context_before'], before)
        if before <= limit:
            return before
        with get_client().start_as_current_observation(name='context-trim',
                input={'tokens': before, 'budget': limit}) as span:
            removed = []
            # Удаляем целиком ход: user → assistant/tool → окончательный ответ.
            while self.count() > limit:
                users = [i for i, m in enumerate(self.history) if m['role'] == 'user']
                if len(users) < 2:
                    break
                removed.extend(self.history[users[0]:users[1]])
                del self.history[users[0]:users[1]]
            # Текущую задачу и пары tool_call/tool_result сохраняем.
            for message in self.history:
                if self.count() <= limit:
                    break
                if message['role'] != 'tool':
                    continue
                tokens = self.tokenizer.encode(message['content'], add_special_tokens=False)
                while len(tokens) > MIN_TOOL_TOKENS and self.count() > limit:
                    original = message['content']
                    tokens = tokens[:len(tokens) // 2]
                    message['content'] = self.tokenizer.decode(tokens) + TRUNCATED
                    removed.append({'tool_call_id': message['tool_call_id'], 'previous_content': original})
            after = self.count()
            stats['trimmed'] = True
            self.trimmed = True
            span.update(output={'tokens': after, 'removed': removed},
                        level='WARNING', status_message='Сокращение истории')
            if after > limit:
                raise ValueError('Текущий вопрос с описаниями инструментов превышает лимит; сократите вопрос.')
            return after

    def answer(self, question):
        lf = get_client()
        self.history.append({'role': 'user', 'content': question})
        stats = dict(input_tokens=0, output_tokens=0, steps=0, trimmed=False,
                     max_context=0, max_context_before=0, status='success')
        answer = ''
        try:
            for step in range(1, CONFIG['max_steps'] + 1):
                stats['steps'] = step
                with lf.start_as_current_observation(name=f'iteration-{step}'):
                    tokens = self.fit_context(stats)
                    stats['max_context'] = max(stats['max_context'], tokens)
                    lf.update_current_span(metadata={'context_tokens': tokens,
                        'context_limit': CONFIG['context_tokens']})
                    remaining = self.tools.deadline - monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Исчерпан таймаут реплики')
                    with lf.start_as_current_observation(name='llm', as_type='generation',
                            model=CONFIG['model'], input={'messages': self.history, 'tools': SCHEMAS},
                            model_parameters={'temperature': CONFIG['temperature'],
                                              'max_tokens': CONFIG['max_output_tokens']}) as generation:
                        response = self.client.chat.completions.create(model=CONFIG['model'],
                            messages=self.history, tools=SCHEMAS, temperature=CONFIG['temperature'],
                            max_tokens=CONFIG['max_output_tokens'], timeout=remaining)
                        usage = response.usage
                        stats['input_tokens'] += usage.prompt_tokens
                        stats['output_tokens'] += usage.completion_tokens
                        message = response.choices[0].message
                        generation.update(output=message.model_dump(exclude_none=True),
                            usage_details={'input': usage.prompt_tokens, 'output': usage.completion_tokens},
                            cost_details={'total': cost(usage.prompt_tokens, usage.completion_tokens)})
                    if monotonic() > self.tools.deadline:
                        raise TimeoutError('Исчерпан таймаут реплики')
                    self.history.append(message.model_dump(include={'role', 'content', 'tool_calls'}, exclude_none=True))
                    if not message.tool_calls:
                        answer = message.content or ''
                        if response.choices[0].finish_reason == 'length':
                            stats['status'] = 'output_limit'
                        break
                    for call in message.tool_calls:
                        try:
                            arguments = json.loads(call.function.arguments)
                        except ValueError:
                            arguments = {'invalid_json': call.function.arguments}
                        result = self.tools.call(call.function.name, arguments)
                        self.history.append({'role': 'tool', 'tool_call_id': call.id,
                            'content': json.dumps(result, ensure_ascii=False)})
            else:
                stats['status'] = 'max_steps'
                answer = 'Не удалось завершить задачу за отведённое число шагов.'
        except Exception as error:
            stats['status'] = 'timeout' if isinstance(error, TimeoutError) or 'timeout' in type(error).__name__.lower() else 'error'
            answer = str(error)
        if not self.history or self.history[-1].get('content') != answer:
            self.history.append({'role': 'assistant', 'content': answer})
        stats['context_tokens'] = self.count()
        stats['trimmed_before_answer'] = self.trimmed
        if stats['status'] != 'success':
            lf.update_current_span(level='WARNING' if stats['status'] in ('max_steps', 'timeout', 'output_limit') else 'ERROR',
                                   status_message=stats['status'])
        return answer, stats

    def close(self):
        self.client.close()
