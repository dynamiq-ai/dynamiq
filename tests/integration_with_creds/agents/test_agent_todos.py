import json

import pytest

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.file import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore
from dynamiq.utils.logger import logger


@pytest.fixture(scope="module")
def openai_connection():
    return OpenAIConnection()


@pytest.fixture(scope="module")
def openai_llm(openai_connection):
    return OpenAI(
        connection=openai_connection,
        model="gpt-4o-mini",
        max_tokens=3000,
        temperature=0.1,
    )


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig(request_timeout=120)


@pytest.mark.integration
def test_agent_todo_state_updates(openai_llm, run_config, monkeypatch):
    """Verify the agent writes todos mid-run and that end-of-run cleanup wipes them.

    The todos file and ``agent.state.todos`` only exist during the run — ``Agent.execute()``
    unconditionally deletes them in its ``finally`` block. To observe that the file
    really did exist, we intercept ``_clear_todos_file`` and snapshot the pre-cleanup
    state, then check post-run that cleanup actually ran.
    """
    from dynamiq.nodes.agents.agent import TodoItem
    from dynamiq.nodes.agents.base import Agent as AgentBase
    from dynamiq.nodes.tools.todo_tools import TODOS_FILE_PATH, TodoWriteTool

    file_store_backend = InMemoryFileStore()
    file_store_config = FileStoreConfig(
        enabled=True,
        backend=file_store_backend,
        todo_enabled=True,
    )

    agent = Agent(
        name="TodoAgent",
        id="todo_agent",
        llm=openai_llm,
        role="A helpful assistant that creates todo lists for tasks.",
        inference_mode=InferenceMode.XML,
        file_store=file_store_config,
        tools=[],
        max_loops=5,
        verbose=True,
    )

    # Verify TodoWriteTool was added automatically
    todo_tools = [t for t in agent.tools if isinstance(t, TodoWriteTool)]
    assert len(todo_tools) == 1, "TodoWriteTool should be automatically added"

    # Snapshot file + state at the instant before cleanup deletes them.
    snapshot: dict = {}
    original_clear = AgentBase._clear_todos_file

    def _snapshot_then_clear(self):
        snapshot["file_existed"] = file_store_backend.exists(TODOS_FILE_PATH)
        if snapshot["file_existed"]:
            snapshot["file_content"] = file_store_backend.retrieve(TODOS_FILE_PATH)
        snapshot["state_todos"] = list(self.state.todos) if self.state is not None else []
        return original_clear(self)

    monkeypatch.setattr(AgentBase, "_clear_todos_file", _snapshot_then_clear)

    workflow = Workflow(flow=Flow(nodes=[agent]))
    tracing = TracingCallbackHandler()
    config = run_config.model_copy(update={"callbacks": [tracing]})

    input_data = {
        "input": "Create a simple todo list with 3 tasks: 1) Review code, 2) Write tests, 3) Update docs. "
        "Then mark the first task as completed and answer with what you did."
    }

    result = workflow.run(input_data=input_data, config=config)

    assert result.status == RunnableStatus.SUCCESS, f"Agent run failed: {result.output}"

    agent_output = result.output[agent.id]["output"]["content"]
    logger.info(f"Agent output: {agent_output}")

    # The todos file MUST have existed at the moment cleanup was about to run,
    # proving TodoWriteTool wrote it during the loop.
    assert snapshot.get("file_existed"), "Todos file should have existed before cleanup fired"
    todos_data = json.loads(snapshot["file_content"].decode("utf-8"))
    assert "todos" in todos_data, "Todos data should have 'todos' key"
    assert (
        len(todos_data["todos"]) == 3
    ), f"Pre-cleanup todos file should contain 3 items, got {len(todos_data['todos'])}"

    # In-memory state must also have carried 3 TodoItem objects at that moment.
    pre_cleanup_state = snapshot["state_todos"]
    assert len(pre_cleanup_state) == 3, f"Pre-cleanup state should have 3 todos, got {len(pre_cleanup_state)}"
    assert all(isinstance(t, TodoItem) for t in pre_cleanup_state), "All pre-cleanup todos should be TodoItem instances"

    # After execute() returns, the cleanup must have completed: file gone, state empty.
    assert not file_store_backend.exists(
        TODOS_FILE_PATH
    ), "Todos file should be deleted by Agent cleanup in execute()'s finally block"
    assert agent.state.todos == [], f"AgentState.todos should be empty after cleanup; got {agent.state.todos}"

    logger.info("--- Test Passed for Todo State Updates ---")
