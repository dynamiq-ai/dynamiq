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
from dynamiq.storages.file import InMemorySandbox, SandboxConfig
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
def test_agent_todo_state_updates(openai_llm, run_config):
    """Test that Agent todo state is updated when using TodoWriteTool."""
    from dynamiq.nodes.tools.todo_tools import TODOS_FILE_PATH, TodoWriteTool

    sandbox_backend = InMemorySandbox()
    sandbox_config = SandboxConfig(
        enabled=True,
        backend=sandbox_backend,
        todo_enabled=True,
    )

    agent = Agent(
        name="TodoAgent",
        id="todo_agent",
        llm=openai_llm,
        role="A helpful assistant that creates todo lists for tasks.",
        inference_mode=InferenceMode.XML,
        sandbox=sandbox_config,
        tools=[],
        max_loops=5,
        verbose=True,
    )

    # Verify TodoWriteTool was added automatically
    todo_tools = [t for t in agent.tools if isinstance(t, TodoWriteTool)]
    assert len(todo_tools) == 1, "TodoWriteTool should be automatically added"

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

    # Check that todos were created in the sandbox
    assert sandbox_backend.exists(TODOS_FILE_PATH), "Todos file should exist in sandbox"

    # Verify the agent state has exactly 3 todos
    todos_content = sandbox_backend.retrieve(TODOS_FILE_PATH).decode("utf-8")
    todos_data = json.loads(todos_content)
    assert "todos" in todos_data, "Todos data should have 'todos' key"
    assert len(todos_data["todos"]) == 3, f"Should have exactly 3 todos, got {len(todos_data['todos'])}"

    # Verify AgentState also has the todos
    from dynamiq.nodes.agents.agent import TodoItem

    assert len(agent.state.todos) == 3, f"AgentState should have 3 todos, got {len(agent.state.todos)}"
    assert all(
        isinstance(t, TodoItem) for t in agent.state.todos
    ), "All todos in AgentState should be TodoItem instances"

    logger.info("--- Test Passed for Todo State Updates ---")
