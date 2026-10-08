"""An agent behind a selector still scopes memory to the run's user_id / session_id.

A selector replaces the agent's input, so ids the run was started with are gone unless the
selector maps them. The flow records them as the run's identity, and the agent falls back to it.
"""

import asyncio
import uuid

import pytest

from dynamiq import Workflow, connections
from dynamiq.flows import Flow
from dynamiq.memory import Memory
from dynamiq.memory.backends import InMemory
from dynamiq.nodes import InputTransformer
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents import base as agent_base
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.types import InferenceMode
from dynamiq.nodes.utils import Input
from dynamiq.runnables import RunnableStatus
from dynamiq.utils.run_context import current_run_identity
from tests.unit.nodes.agents.test_long_term_memory_integration import _ltm_config
from tests.unit.storages.memory.conftest import FakeMemoryStore

USER_ID = "user-1"
SESSION_ID = "session-1"


@pytest.fixture
def mock_llm_response_text():
    return "Thought: I can answer directly.\nAnswer: mocked_response"


@pytest.fixture
def llm():
    return OpenAI(
        name="OpenAI",
        model="gpt-4o-mini",
        connection=connections.OpenAI(id=str(uuid.uuid4()), api_key="api-key"),
    )


def _workflow(llm, selector: dict[str, str], **agent_kwargs) -> tuple[Workflow, Agent]:
    start = Input(id="start", name="start")
    agent = Agent(
        id="assistant",
        name="assistant",
        llm=llm,
        tools=[],
        role="You are a helpful assistant.",
        inference_mode=InferenceMode.DEFAULT,
        depends=[NodeDependency(node=start)],
        input_transformer=InputTransformer(selector=selector),
        **agent_kwargs,
    )
    return Workflow(flow=Flow(nodes=[start, agent])), agent


def _scopes(memory: Memory) -> set[tuple[str | None, str | None]]:
    return {(m.metadata.get("user_id"), m.metadata.get("session_id")) for m in memory.backend.messages}


def test_selector_without_ids_uses_run_identity(llm, mock_llm_executor):
    memory = Memory(backend=InMemory())
    workflow, _ = _workflow(llm, {"input": "$.start.output.query"}, memory=memory)

    result = workflow.run(input_data={"query": "Hello", "user_id": USER_ID, "session_id": SESSION_ID})

    assert result.status == RunnableStatus.SUCCESS
    assert memory.backend.messages, "memory should be engaged with the run's identity"
    assert _scopes(memory) == {(USER_ID, SESSION_ID)}


def test_history_is_read_back_on_the_next_run(llm, mock_llm_executor):
    memory = Memory(backend=InMemory())
    workflow, _ = _workflow(llm, {"input": "$.start.output.query"}, memory=memory)
    run_input = {"user_id": USER_ID, "session_id": SESSION_ID}

    workflow.run(input_data={"query": "My name is Ada", **run_input})
    workflow.run(input_data={"query": "What is my name?", **run_input})

    second_call_messages = mock_llm_executor.call_args_list[-1].kwargs["messages"]
    contents = [str(message.get("content")) for message in second_call_messages]
    assert any("My name is Ada" in content for content in contents)


def test_async_run_uses_run_identity(llm, mock_llm_executor):
    memory = Memory(backend=InMemory())
    workflow, _ = _workflow(llm, {"input": "$.start.output.query"}, memory=memory)

    result = asyncio.run(
        workflow.run_async(input_data={"query": "Hello", "user_id": USER_ID, "session_id": SESSION_ID})
    )

    assert result.status == RunnableStatus.SUCCESS
    assert _scopes(memory) == {(USER_ID, SESSION_ID)}


def test_ids_on_the_agent_input_win_and_are_not_mixed(llm, mock_llm_executor):
    memory = Memory(backend=InMemory())
    workflow, _ = _workflow(
        llm,
        {"input": "$.start.output.query", "user_id": "$.start.output.account"},
        memory=memory,
    )

    workflow.run(input_data={"query": "Hello", "account": "acct-7", "user_id": USER_ID, "session_id": SESSION_ID})

    assert _scopes(memory) == {("acct-7", None)}


def test_no_identity_warns_and_skips_memory(llm, mock_llm_executor, mocker):
    memory = Memory(backend=InMemory())
    workflow, _ = _workflow(llm, {"input": "$.start.output.query"}, memory=memory)
    warning = mocker.spy(agent_base.logger, "warning")

    result = workflow.run(input_data={"query": "Hello"})

    assert result.status == RunnableStatus.SUCCESS
    assert memory.backend.messages == []
    assert any("no user_id or session_id" in str(call.args[0]) for call in warning.call_args_list)


def test_identity_does_not_outlive_the_run(llm, mock_llm_executor):
    workflow, _ = _workflow(llm, {"input": "$.start.output.query"}, memory=Memory(backend=InMemory()))

    workflow.run(input_data={"query": "Hello", "user_id": USER_ID, "session_id": SESSION_ID})

    assert current_run_identity() is None


def test_long_term_memory_and_memory_store_get_run_user_id(llm, mock_llm_executor, mocker):
    workflow, _ = _workflow(
        llm,
        {"input": "$.start.output.query"},
        long_term_memory=_ltm_config(),
        memory_store={"enabled": True, "backend": FakeMemoryStore()},
    )
    ltm_build = mocker.spy(Agent, "_build_long_term_memory_tools")
    store_build = mocker.spy(Agent, "_build_memory_store_tool")

    workflow.run(input_data={"query": "Hello", "user_id": USER_ID, "session_id": SESSION_ID})

    assert ltm_build.call_count == 1
    assert ltm_build.call_args.args[1].user_id == USER_ID
    assert not isinstance(ltm_build.spy_exception, ValueError)
    assert store_build.call_args.args[1].user_id == USER_ID
