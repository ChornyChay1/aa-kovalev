"""Имена действий, инструментов и стабильные настройки Lab 3."""
import re
from typing import Literal, get_args

QuizAction = Literal['create', 'update', 'delete', 'append_questions', 'delete_question']
CREATE, UPDATE, DELETE, APPEND_QUESTIONS, DELETE_QUESTION = get_args(QuizAction)

SEARCH_BOOK = 'search_book'
MANAGE_QUIZ = 'manage_quiz'
LIST_QUIZES = 'list_quizes'
GET_QUIZ_INFO = 'get_quiz_info'

# Explicit constant set for tool name validation (task_data, etc.).
TOOL_NAMES = frozenset({SEARCH_BOOK, MANAGE_QUIZ, LIST_QUIZES, GET_QUIZ_INFO})

CUSTOM_AGENT = 'custom'
READY_AGENT = 'smolagents'
CODING_AGENT = 'coding'
BOTH_AGENTS = 'both'
ALL_AGENTS = 'all'
WEB_RUN_ID = 'web'
WEB_QUIZ_COLLECTION = WEB_RUN_ID
DATASET_NAME = 'lab3-agent-turns'

MAX_QUIZ_TITLE = 150
MAX_QUESTIONS = 10
MAX_OPTIONS = 5
TURN_RESERVE_TOKENS = 2048
MAX_TOOL_CALLS_PER_TURN = 4
MAX_USER_INPUT_CONTEXT_RATIO = 0.7

QUIZ_LINK = re.compile(
    r'(?P<label>\[[^\]\n]+\])\((?:https?://[^/\s)]+)?(?P<path>/quiz/[^\s)]+)\)'
)
