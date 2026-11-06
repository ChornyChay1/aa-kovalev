"""Регрессии без Ollama, Qdrant и Langfuse: unittest discover -s tests."""
import os
import unittest
from dataclasses import dataclass
from contextlib import nullcontext
from time import monotonic
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ['LANGFUSE_TRACING_ENABLED'] = 'false'

from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RunUsage, RequestUsage

from agents.custom import Agent
from models.schemas import AgentStats
from agents.ready import CodingAgent, OllamaToolModel, ReadyAgent
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole
from task_data import load_tasks
from tool_metrics import check_tools
from tools.book import BookTools
from services.turns import turn
from constants import MAX_USER_INPUT_CONTEXT_RATIO
from settings import CONFIG
from tools.result_budget import count_result, limit_result
from tools.pydantic_tools import build_tools as custom_tools
from tools.smolagents_tools import build_tools as ready_tools


class ToolMetricsTests(unittest.TestCase):
    def row(self, action, error=None):
        return {'turn': 0, 'metadata': {'expected_tools': [['manage_quiz']],
                'expected_actions': [['delete_question']]},
                'calls': [{'name': 'manage_quiz', 'arguments': {'action': action},
                           'result': {'quiz_id': 'quiz'}, 'error': error}]}

    def test_deleting_quiz_does_not_pass_question_deletion(self):
        correctness, missing = check_tools(self.row('delete'))
        self.assertEqual(correctness, 0)
        self.assertIn('manage_quiz:delete_question', missing)

    def test_failed_call_is_attempted_but_not_completed(self):
        correctness, missing = check_tools(self.row('delete_question', 'Not found'))
        self.assertEqual(correctness, 1)
        self.assertTrue(missing)
        self.assertEqual(check_tools(self.row('delete_question')), (1, []))

    def test_dataset_requires_search_only_when_requested(self):
        tasks = load_tasks()
        self.assertEqual(len(tasks), 18)
        self.assertEqual(sum(len(t['input']) for t in tasks), 39)
        for task in tasks:
            if task['id'].startswith('L'):
                self.assertEqual(task['metadata']['expected_tools'][0], ['manage_quiz'])
                self.assertEqual(task['metadata']['expected_tools'][2], ['manage_quiz'])
                self.assertTrue(task['metadata']['requires_context_trim'])
        s07 = next(t for t in tasks if t['id'] == 'S07')
        self.assertIn('get_quiz_info', s07['metadata']['expected_tools'][0])


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tokens = patch('agents.custom.count_text', side_effect=lambda text: len(text) // 4)
        self.tokens.start()
        self.addCleanup(self.tokens.stop)
        self.tool_tokens = patch('tools.result_budget.count_text',
                                 side_effect=lambda text: max(1, len(text) // 4))
        self.tool_tokens.start()
        self.addCleanup(self.tool_tokens.stop)
        self.prompts = patch('agents.custom.system_prompt', return_value=('Учебный помощник.', 'test'))
        self.prompts.start()
        self.addCleanup(self.prompts.stop)
        self.tools = BookTools.__new__(BookTools)
        self.tools.reset()
        self.tools.session = 'test'
        self.agent = Agent(self.tools)
        self.addCleanup(self.agent.close)

    def test_failure_keeps_completed_tool_result_and_usage(self):
        count = 0

        def respond(messages, info):
            nonlocal count
            count += 1
            if count == 1:
                return ModelResponse(parts=[ToolCallPart('list_quizes', {}, 'call-1')],
                                     usage=RequestUsage(input_tokens=100, output_tokens=10))
            raise RuntimeError('Model unavailable')

        with patch.object(self.tools, 'dispatch', return_value={'quizzes': [], 'count': 0}):
            with self.agent.agent.override(model=FunctionModel(respond)):
                _, stats = self.agent.answer('Какие есть квизы?')
        self.assertEqual(stats.status, 'error')
        self.assertEqual(stats.input_tokens, 100)
        self.assertTrue(any(isinstance(p, ToolReturnPart) and p.tool_call_id == 'call-1'
                            for m in self.agent.history for p in m.parts))
        # Следующая реплика получает пригодную для модели историю.
        with self.agent.agent.override(model=FunctionModel(lambda m, i: ModelResponse(parts=[TextPart('Нет квизов.')]))):
            answer, stats = self.agent.answer('Напомни результат')
        self.assertEqual(stats.status, 'success')
        self.assertEqual(answer, 'Нет квизов.')

    def test_tools_stop_after_four_calls_with_room_for_final_answer(self):
        def respond(messages, info):
            if info.function_tools:
                return ModelResponse(parts=[ToolCallPart('list_quizes', {}, f'call-{len(messages)}')])
            return ModelResponse(parts=[TextPart('Выполнены четыре вызова.')])

        with patch.object(self.tools, 'dispatch', return_value={'quizzes': [], 'count': 0}):
            with self.agent.agent.override(model=FunctionModel(respond)):
                output, stats = self.agent.answer('Проверь квизы')
        self.assertEqual(stats.status, 'success')
        self.assertEqual(len(self.tools.calls), 4)
        self.assertEqual(output, 'Выполнены четыре вызова.')

    def test_finalize_retries_unconfirmed_write_and_uses_tool_receipt(self):
        responses = iter([
            ModelResponse(parts=[TextPart('Вопрос успешно добавлен в квиз.')]),
            ModelResponse(parts=[ToolCallPart('manage_quiz', {
                'action': 'append_questions', 'quiz_id': 'quiz-123', 'questions': [
                    {'text': 'Вопрос?', 'options': ['Да', 'Нет'], 'correct': 1},
                ],
            }, 'append')]),
            ModelResponse(parts=[TextPart('Добавил десять вопросов. Правильный ответ: Да.')]),
        ])
        receipt = {'quiz_id': 'quiz-123', 'title': 'Циклы', 'url': '/quiz/quiz-123',
                   'added_question_numbers': [5]}
        with patch.object(self.tools, 'dispatch', return_value=receipt) as dispatch:
            with self.agent.agent.override(model=FunctionModel(lambda m, i: next(responses))):
                answer, stats = self.agent.answer('Добавь ещё 1 вопрос')
        self.assertEqual(stats.status, 'success')
        dispatch.assert_called_once()
        self.assertIn('Номера: 5.', answer)
        self.assertIn('/quiz/quiz-123', answer)
        self.assertNotIn('Правильный ответ', answer)

    def test_summary_requests_do_not_hide_tools(self):
        ctx = SimpleNamespace(usage=RunUsage(requests=7), run_step=1)
        definitions = [SimpleNamespace(name='manage_quiz')]
        self.assertEqual(self.agent.loop.run(self.agent.prepare_tools(ctx, definitions)), definitions)

    def test_compaction_keeps_tools_on_next_model_request(self):
        self.agent.compaction.target_tokens = 1500
        self.agent.compaction.tiers = [self.agent.compaction.tiers[-1]]
        self.agent.compaction.tiers[0].keep_tokens = 100
        self.agent.history = [ModelRequest(parts=[UserPromptPart('старый контекст ' * 3000)], run_id='old-run'),
                              ModelResponse(parts=[TextPart('старый ответ')], run_id='old-run'),
                              ModelRequest(parts=[UserPromptPart('недавняя тема ' * 100)], run_id='recent'),
                              ModelResponse(parts=[TextPart('ответ')], run_id='recent')]
        def respond(messages, info):
            self.assertIn('manage_quiz', [t.name for t in info.function_tools])
            self.assertTrue(any(isinstance(p, UserPromptPart) and p.content == 'Создай квиз'
                                for m in messages for p in m.parts))
            return ModelResponse(parts=[TextPart('Ответ')])
        with self.agent.agent.override(model=FunctionModel(respond)):
            _, stats = self.agent.answer('Создай квиз')
        self.assertEqual(stats.status, 'success')
        self.assertTrue(stats.trimmed)
        self.assertGreater(stats.max_context, 0)

    def test_large_completed_history_is_summarized_before_new_tool_call(self):
        self.agent.history = [
            ModelRequest(parts=[UserPromptPart('Квиз quiz-123. ' * 6000)], run_id='old'),
            ModelResponse(parts=[TextPart('Создан квиз quiz-123')], run_id='old'),
        ]
        summaries = []
        def respond(messages, info):
            if not info.function_tools:
                summaries.append(True)
                return ModelResponse(parts=[TextPart('Создан квиз quiz-123, требуется добавить вопрос.')])
            self.assertTrue(any(isinstance(p, UserPromptPart) and p.content == 'Добавь пятый вопрос'
                                for m in messages for p in m.parts))
            if any(isinstance(p, ToolReturnPart) for m in messages for p in m.parts):
                return ModelResponse(parts=[TextPart('Проверено')])
            return ModelResponse(parts=[ToolCallPart('list_quizes', {}, 'new-call')])
        with patch.object(self.tools, 'dispatch', return_value={'quizzes': [], 'count': 0}):
            with self.agent.agent.override(model=FunctionModel(respond)):
                _, stats = self.agent.answer('Добавь пятый вопрос')
        self.assertEqual(stats.status, 'success')
        self.assertTrue(summaries)
        self.assertTrue(stats.trimmed)
        self.assertEqual(len(self.tools.calls), 1)
        self.assertTrue(any(isinstance(p, ToolReturnPart) and p.tool_call_id == 'new-call'
                            for m in self.agent.history for p in m.parts))

    def test_both_adapters_return_the_same_search_evidence(self):
        hits = [{'id': 'chunk', 'page': 47, 'text': 'Имена переменных ' * 1000, 'score': 0.9}]
        self.tools.rag = SimpleNamespace(
            api=SimpleNamespace(client=SimpleNamespace()), config={'source': 'book'},
            manifest={'version': 'test'}, retrieve=lambda query, k: (hits, None))
        custom = next(t for t in custom_tools(self.tools) if t.name == 'search_book')
        ready = next(t for t in ready_tools(self.tools) if t.name == 'search_book')
        result = custom.function('Имена')
        self.assertEqual(result, ready.forward('Имена'))
        self.assertEqual(result[0]['page'], 47)
        self.assertFalse(result[0]['truncated'])
        self.assertEqual(result[0]['text'], hits[0]['text'])
        self.assertEqual(self.tools.retrievals[0], hits)

    def test_large_search_result_is_clipped_by_tokens_and_marked(self):
        hits = [{'id': 'chunk-1', 'page': 47, 'text': 'Длинный фрагмент ' * 5000}]
        self.tools.rag = SimpleNamespace(
            api=SimpleNamespace(client=SimpleNamespace()), config={'source': 'book'},
            manifest={'version': 'test'}, retrieve=lambda query, k: (hits, None))
        result = self.tools.call('search_book', {'query': 'фрагмент'})
        budget = (17288 - 1024 - 256)
        self.assertTrue(result[0]['truncated'])
        self.assertLessEqual(count_result(result), budget)
        self.assertEqual(self.tools.retrievals[0], hits)
        self.assertEqual(self.tools.calls[0].result[0]['text'], hits[0]['text'])
        self.assertTrue(self.tools.calls[0].truncated)
        self.assertEqual(self.tools.calls[0].returned_tokens, count_result(result))

    def test_large_list_quizes_keeps_ids_and_marks_omissions(self):
        quizzes = [
            {'quiz_id': str(i), 'title': 'Квиз ' + 'Название ' * 12,
             'url': '/quiz/' + str(i)}
            for i in range(1000)
        ]
        with patch.object(self.tools, 'dispatch',
                          return_value={'quizzes': quizzes, 'count': len(quizzes)}):
            result = self.tools.call('list_quizes', {})
        self.assertTrue(result['_truncated'])
        self.assertLessEqual(count_result(result), 17288 - 1024 - 256)
        self.assertEqual(result['count'], 1000)
        self.assertEqual(result['quizzes'][0]['quiz_id'], '0')
        self.assertEqual(len(self.tools.calls[0].result['quizzes']), 1000)

    def test_after_clean_current_tool_output_is_trimmed_to_fit(self):
        @dataclass
        class Context:
            run_id: str
            messages: list

        class ClearOld:
            async def compact(self, messages, ctx):
                cleared.append(True)
                return []

        cleared = []
        big = [{'id': 'chunk-1', 'page': 47, 'text': 'Большой текст ' * 2000}]
        messages = [
            ModelRequest(parts=[UserPromptPart('старый запрос')], run_id='old'),
            ModelResponse(parts=[TextPart('старый ответ')], run_id='old'),
            ModelRequest(parts=[UserPromptPart('новый запрос')], run_id='new'),
            ModelResponse(parts=[ToolCallPart('search_book', {'query': 'x'}, 'call-1')], run_id='new'),
            ModelRequest(parts=[ToolReturnPart(
                tool_name='search_book', content=big, tool_call_id='call-1')], run_id='new'),
        ]
        compaction = self.agent.compaction
        compaction.tiers = [ClearOld()]
        compaction.target_tokens = 400
        compaction.measure = lambda msgs: max(1, len(
            ModelMessagesTypeAdapter.dump_json(msgs, exclude_defaults=True,
                                                exclude_none=True)) // 4)
        compaction.observe = lambda ctx, changed: None
        ctx = Context('new', messages)
        with patch('agents.custom.INPUT_BUDGET', 650):
            output = self.agent.loop.run(
                compaction.before_model_request(ctx, Context('new', messages)))
        self.assertTrue(cleared)
        self.assertLessEqual(compaction.measure(output.messages), 650)
        self.assertEqual(output.messages[-3].parts[0].content, 'новый запрос')
        returned = output.messages[-1].parts[0].content
        self.assertTrue(returned[0]['truncated'])
        if 'text' in returned[0]:
            self.assertLess(len(returned[0]['text']), len(big[0]['text']))
        else:
            self.assertEqual(returned[0]['omitted_items'], 1)

    def test_oversized_list_has_explicit_marker_even_when_no_hit_fits(self):
        oversized = [{'id': 'huge-id', 'page': 47, 'text': 'x' * 1000}]
        marker = [{'truncated': True, 'omitted_items': 1}]
        result, clipped = limit_result(oversized, count_result(marker))
        self.assertTrue(clipped)
        self.assertEqual(result, marker)

    def test_coding_agent_executes_shared_tool(self):
        with patch('agents.ready.system_prompt', return_value=('Учебный помощник.', 'test')):
            ready = CodingAgent(self.tools)
        self.addCleanup(ready.close)
        message = ChatMessage(role=MessageRole.ASSISTANT,
            content="```python\nresult = list_quizes()\nfinal_answer(result)\n```", tool_calls=[])
        with patch.object(ready.model, 'generate', return_value=message):
            with patch.object(self.tools, 'dispatch', return_value={'quizzes': [], 'count': 0}):
                _, stats = ready.answer('Покажи квизы')
        self.assertEqual(stats.status, 'success')
        self.assertEqual(self.tools.calls[-1].name, 'list_quizes')

    def test_ready_agent_constructs_and_refuses_expired_request(self):
        with patch('agents.ready.system_prompt', return_value=('Учебный помощник.', 'test')):
            ready = ReadyAgent(self.tools)
        self.addCleanup(ready.close)
        self.tools.deadline = monotonic() - 1
        with self.assertRaises(TimeoutError):
            ready.model.generate([])



class OllamaAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tools = SimpleNamespace(deadline=monotonic() + 60)
        self.model = OllamaToolModel(
            self.tools, model_id='qwen2.5:7b', api_base='http://localhost:11434/v1',
            api_key='ollama')
        self.addCleanup(self.model.client.close)
        self.available = [SimpleNamespace(name='manage_quiz'), SimpleNamespace(name='final_answer')]

    @staticmethod
    def call(name, arguments):
        return ChatMessageToolCall(
            id=name, type='function',
            function=ChatMessageToolCallFunction(name=name, arguments=arguments))

    def generate(self, message):
        with patch('agents.ready.OpenAIServerModel.generate', return_value=message):
            return self.model.generate([], tools_to_call_from=self.available)

    def test_mixed_final_answer_does_not_discard_action(self):
        action = self.call('manage_quiz', {'action': 'create', 'title': 'Списки'})
        final = self.call('final_answer', {'answer': 'Готово'})
        message = ChatMessage(role=MessageRole.ASSISTANT, tool_calls=[action, final])
        self.assertEqual(self.generate(message).tool_calls, [action])

    def test_multiple_final_answers_keep_one(self):
        first = self.call('final_answer', {'answer': 'Ответ'})
        message = ChatMessage(role=MessageRole.ASSISTANT, tool_calls=[
            first, self.call('final_answer', {'answer': 'Другой'})])
        self.assertEqual(self.generate(message).tool_calls, [first])

    def test_completed_plain_text_becomes_final_answer(self):
        message = ChatMessage(role=MessageRole.ASSISTANT, content='Объяснение',
            raw=SimpleNamespace(choices=[SimpleNamespace(finish_reason='stop')]))
        call = self.generate(message).tool_calls[0]
        self.assertEqual(call.function.name, 'final_answer')
        self.assertEqual(call.function.arguments, {'answer': 'Объяснение'})

    def test_truncated_output_is_not_misreported_as_success(self):
        message = ChatMessage(role=MessageRole.ASSISTANT, content='{"name": "manage_quiz"',
            raw=SimpleNamespace(choices=[SimpleNamespace(finish_reason='length')]))
        self.assertFalse(self.generate(message).tool_calls)

    def test_unknown_title_is_ignored_when_required_arguments_exist(self):
        self.available.insert(0, SimpleNamespace(
            name='get_quiz_info', inputs={'quiz_id': {'type': 'string'},
                                          'question_number': {'type': 'integer', 'nullable': True}}))
        action = self.call('get_quiz_info', {'quiz_id': 'abc', 'title': 'Списки'})
        message = ChatMessage(role=MessageRole.ASSISTANT, tool_calls=[action])
        self.generate(message)
        self.assertEqual(action.function.arguments, {'quiz_id': 'abc'})

    def test_unknown_title_does_not_hide_missing_required_argument(self):
        self.available.insert(0, SimpleNamespace(
            name='get_quiz_info', inputs={'quiz_id': {'type': 'string'}}))
        action = self.call('get_quiz_info', {'title': 'Списки'})
        message = ChatMessage(role=MessageRole.ASSISTANT, tool_calls=[action])
        self.generate(message)
        self.assertEqual(action.function.arguments, {'title': 'Списки'})

    def test_existing_tool_call_is_preserved(self):
        action = self.call('manage_quiz', {'action': 'create', 'title': 'Списки'})
        message = ChatMessage(role=MessageRole.ASSISTANT, tool_calls=[action])
        self.assertEqual(self.generate(message).tool_calls, [action])


class UserInputLimitTests(unittest.TestCase):
    def test_oversized_prompt_rejected_without_model_call(self):
        limit = int(CONFIG['context_tokens'] * MAX_USER_INPUT_CONTEXT_RATIO)
        span = SimpleNamespace(
            trace_id='a' * 32, id='b' * 16,
            update=MagicMock(), set_trace_io=MagicMock(),
        )
        lf = SimpleNamespace(
            start_as_current_observation=lambda **kwargs: nullcontext(span))
        tools = BookTools.__new__(BookTools)
        agent = SimpleNamespace(
            prompt_version='test',
            answer=MagicMock(return_value=('Готово', AgentStats(
                status='success', input_tokens=2, output_tokens=2)))),
        )

        with patch('services.turns.get_client', return_value=lf), \
                patch('services.turns.propagate_attributes', return_value=nullcontext()), \
                patch('services.turns.trace_url', return_value=None):
            with patch('services.turns.count_text', return_value=limit + 1):
                result = turn(agent, tools, 'Слишком длинный запрос',
                              'session', 'custom', 'run')
            self.assertEqual(result.status, 'error')
            self.assertIn('Сократите сообщение', result.output)
            self.assertEqual(result.steps, 0)
            self.assertEqual(result.tool_calls, 0)
            agent.answer.assert_not_called()
            self.assertTrue(any(call.kwargs.get('metadata', {}).get('user_input_rejected')
                                for call in span.update.call_args_list))

            with patch('services.turns.count_text', return_value=limit):
                accepted = turn(agent, tools, 'Допустимый запрос',
                                'session', 'custom', 'run')
            self.assertEqual(accepted.status, 'success')
            agent.answer.assert_called_once_with('Допустимый запрос')


if __name__ == '__main__':
    unittest.main()
