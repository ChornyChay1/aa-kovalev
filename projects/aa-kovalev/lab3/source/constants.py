"""Имена действий, инструментов и стабильные настройки Lab 3."""
import re
from typing import Literal, get_args

QuizAction = Literal['create', 'update', 'delete']
CREATE, UPDATE, DELETE = get_args(QuizAction)

SEARCH_BOOK = 'search_book'
MANAGE_QUIZ = 'manage_quiz'
LIST_QUIZES = 'list_quizes'
GET_QUIZ_INFO = 'get_quiz_info'

CUSTOM_AGENT = 'custom'
READY_AGENT = 'smolagents'
BOTH_AGENTS = 'both'
WEB_RUN_ID = 'web'
WEB_QUIZ_COLLECTION = WEB_RUN_ID
DATASET_NAME = 'lab3-agent-turns'

MAX_QUIZ_TITLE = 150
MAX_QUESTIONS = 10
MAX_OPTIONS = 5
RECENT_HISTORY_TOKENS = 600
TOOL_OUTPUT_CHARS = 1400
QUIZ_LINK = re.compile(
    r'(?P<label>\[[^\]\n]+\])\((?:https?://[^/\s)]+)?(?P<path>/quiz/[^\s)]+)\)'
)
