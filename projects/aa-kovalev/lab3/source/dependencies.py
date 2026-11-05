"""Composition root for the web chat and CLI."""
from collections.abc import Iterator
from pathlib import Path

from dishka import Provider, Scope, provide

from agent import Agent
from book_tools import BookTools
from ready_agent import ReadyAgent


class LabProvider(Provider):
    def __init__(self, index: Path | None = None) -> None:
        super().__init__()
        self.index = index

    @provide(scope=Scope.SESSION)
    def book_tools(self) -> Iterator[BookTools]:
        tools = BookTools(self.index)
        try:
            yield tools
        finally:
            tools.close()

    @provide(scope=Scope.REQUEST)
    def custom_agent(self, tools: BookTools) -> Iterator[Agent]:
        agent = Agent(tools)
        try:
            yield agent
        finally:
            agent.close()

    @provide(scope=Scope.REQUEST)
    def ready_agent(self, tools: BookTools) -> Iterator[ReadyAgent]:
        agent = ReadyAgent(tools)
        try:
            yield agent
        finally:
            agent.close()
