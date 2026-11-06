"""Регрессии без Ollama, Qdrant и Langfuse: unittest discover -s tests."""
import os
import unittest
from time import monotonic
from types import SimpleNamespace
from unittest.mock import patch

os.environ['LANGFUSE_TRACING_ENABLED'] = 'false'

from pydantic_ai.messages import ModelRequest, UserPromptPart, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.usage import RunUsage, RequestUsage

from agents.custom import Agent
from agents.ready import CodingAgent, OllamaToolModel, ReadyAgent
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole
from task_data import load_tasks
from tool_metrics import check_tools
from tools.book import BookTools
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
        self.assertTrue(result[0]['truncated'])
        self.assertEqual(self.tools.retrievals[0], hits)

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


if __name__ == '__main__':
    unittest.main()
