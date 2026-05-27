"""Tests for Agent long-term memory integration.

Covers `_build_long_term_memory_tools` (the per-run tool-construction
helper) and the snapshot/restore behavior of `self.tools` across an
`execute()` call.

The execute-level tests mock `_run_agent` so we don't need a real LLM
backend response — we only verify the agent-loop bookkeeping.
"""
import hashlib
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

import pytest

from dynamiq.connections import BaseConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.memory.long_term import LongTermMemory, LongTermMemoryConfig
from dynamiq.memory.long_term.backends.in_memory import InMemoryLongTermMemoryBackend
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.embedders.base import TextEmbedder, TextEmbedderInputSchema
from dynamiq.nodes.llms import OpenAI


class _StubConnection(BaseConnection):
    def connect(self) -> None:
        return None


class _FakeEmbedder(TextEmbedder):
    name: str = "fake-text-embedder"
    connection: BaseConnection = _StubConnection()
    DIM: ClassVar[int] = 16

    def execute(self, input_data: TextEmbedderInputSchema, config=None, **kwargs):
        text = input_data.query if hasattr(input_data, "query") else input_data["query"]
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        raw = [(b / 127.5) - 1.0 for b in digest[: self.DIM]]
        norm = sum(x * x for x in raw) ** 0.5 or 1.0
        return {"query": text, "embedding": [x / norm for x in raw]}


@pytest.fixture
def ltm():
    return LongTermMemory(backend=InMemoryLongTermMemoryBackend(), embedder=_FakeEmbedder())


@pytest.fixture
def llm():
    """Real OpenAI LLM object — never executed in these tests. Constructed
    only to satisfy Agent's pydantic validation."""
    return OpenAI(
        connection=OpenAIConnection(api_key="test-key"),
        model="gpt-4o",
    )


def _make_agent(llm, *, ltm=None, ltm_config=None) -> Agent:
    kwargs = {"name": "test", "llm": llm, "tools": []}
    if ltm is not None:
        kwargs["long_term_memory"] = ltm
    if ltm_config is not None:
        kwargs["long_term_memory_config"] = ltm_config
    return Agent(**kwargs)


def _input(user_id=None, session_id=None):
    return SimpleNamespace(user_id=user_id, session_id=session_id, input="hi")


# --- LongTermMemoryConfig ---


def test_config_default_includes_all_three_tools():
    assert LongTermMemoryConfig().tools == ("remember", "recall", "forget")


def test_config_can_restrict_to_read_only():
    assert LongTermMemoryConfig(tools=("recall",)).tools == ("recall",)


# --- Agent field declarations ---


def test_agent_has_long_term_memory_fields():
    fields = Agent.model_fields
    assert "long_term_memory" in fields
    assert "long_term_memory_config" in fields
    assert fields["long_term_memory"].default is None


def test_agent_long_term_memory_defaults_to_none(llm):
    agent = _make_agent(llm)
    assert agent.long_term_memory is None
    assert agent.long_term_memory_config.tools == ("remember", "recall", "forget")


# --- _build_long_term_memory_tools ---


def test_build_returns_three_tools_when_ltm_and_user_id_present(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    tools = agent._build_long_term_memory_tools(_input(user_id="u1"))
    assert {t.name for t in tools} == {"remember_fact", "recall_facts", "forget_fact"}


def test_build_returns_empty_when_no_user_id(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    assert agent._build_long_term_memory_tools(_input(session_id="s1")) == []


def test_build_returns_empty_when_no_long_term_memory(llm):
    agent = _make_agent(llm)
    assert agent._build_long_term_memory_tools(_input(user_id="u1")) == []


def test_build_respects_config_include(llm, ltm):
    agent = _make_agent(llm, ltm=ltm, ltm_config=LongTermMemoryConfig(tools=("recall",)))
    tools = agent._build_long_term_memory_tools(_input(user_id="u1"))
    assert [t.name for t in tools] == ["recall_facts"]


def test_build_bakes_user_id_into_each_tool(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    tools = agent._build_long_term_memory_tools(_input(user_id="u1"))
    for tool in tools:
        assert tool.user_id == "u1"


# --- execute() splice: snapshot/restore self.tools ---


def _patch_run_agent_capture_tools(agent, captured):
    def fake_run(*args, **kwargs):
        captured.extend(agent.tools)
        return "ok"

    return patch.object(agent, "_run_agent", side_effect=fake_run)


def test_execute_attaches_ltm_tools_during_run_and_restores_after(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)
    captured: list = []

    with _patch_run_agent_capture_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert {"remember_fact", "recall_facts", "forget_fact"} <= {t.name for t in captured}
    assert agent.tools == original_tools


def test_execute_restores_tools_even_when_run_raises(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)

    with patch.object(agent, "_run_agent", side_effect=RuntimeError("boom")):
        # run_sync wraps exceptions in a failed RunnableResult; check tools after.
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert agent.tools == original_tools


def test_execute_does_not_mutate_tools_when_no_user_id(llm, ltm):
    agent = _make_agent(llm, ltm=ltm)
    original_tools = list(agent.tools)
    captured: list = []

    with _patch_run_agent_capture_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi"})

    assert {t.name for t in captured} == {t.name for t in original_tools}
    assert agent.tools == original_tools


def test_execute_does_not_mutate_tools_when_no_long_term_memory(llm):
    agent = _make_agent(llm)
    original_tools = list(agent.tools)
    captured: list = []

    with _patch_run_agent_capture_tools(agent, captured):
        agent.run_sync(input_data={"input": "hi", "user_id": "u1"})

    assert {t.name for t in captured} == {t.name for t in original_tools}
    assert agent.tools == original_tools
