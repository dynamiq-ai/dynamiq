import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python_code_executor import PythonCodeExecutor
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore


@pytest.fixture
def test_llm():
    return OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", max_tokens=100, temperature=0)


def _agent_with_store(llm, tool, store):
    return Agent(
        name="Agent",
        llm=llm,
        role="r",
        tools=[tool],
        file_store=(
            FileStoreConfig(enabled=True, backend=store)
            if store
            else FileStoreConfig(enabled=False, backend=InMemoryFileStore())
        ),
    )


def test_file_store_injected_when_store_is_empty(test_llm):
    """The first run of any agent: the store is enabled but nothing was uploaded.

    PythonCodeExecutor refuses to run without a file store, so an empty store is
    the case that has to work, not the exception.
    """
    tool = PythonCodeExecutor(name="code-interpreter")
    store = InMemoryFileStore()
    agent = _agent_with_store(test_llm, tool, store)

    agent._inject_files_into_tool(tool, {})

    assert tool.file_store is store


def test_file_store_injected_when_store_has_files(test_llm):
    tool = PythonCodeExecutor(name="code-interpreter")
    store = InMemoryFileStore()
    store.store("notes.txt", b"hello")
    agent = _agent_with_store(test_llm, tool, store)

    agent._inject_files_into_tool(tool, {})

    assert tool.file_store is store


def test_tool_keeps_its_own_file_store(test_llm):
    own_store = InMemoryFileStore()
    tool = PythonCodeExecutor(name="code-interpreter", file_store=own_store)
    agent = _agent_with_store(test_llm, tool, InMemoryFileStore())

    agent._inject_files_into_tool(tool, {})

    assert tool.file_store is own_store


def test_nothing_injected_without_an_agent_file_store(test_llm):
    tool = PythonCodeExecutor(name="code-interpreter")
    agent = _agent_with_store(test_llm, tool, None)

    agent._inject_files_into_tool(tool, {})

    assert tool.file_store is None
