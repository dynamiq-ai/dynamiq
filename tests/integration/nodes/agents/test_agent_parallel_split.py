"""Tests for the parallel/sequential tool split in _execute_tools."""

import uuid

import pytest

from dynamiq import connections, prompts
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig


@pytest.fixture
def openai_node():
    conn = connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key")
    return OpenAI(
        name="OpenAI",
        model="gpt-4o-mini",
        connection=conn,
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )


def _make_tool(name: str, *, is_parallel_execution_allowed: bool = True) -> Python:
    return Python(
        name=name,
        description=f"Tool {name}",
        code="",
        is_parallel_execution_allowed=is_parallel_execution_allowed,
    )


@pytest.fixture
def parallel_tool_a():
    return _make_tool("ToolA", is_parallel_execution_allowed=True)


@pytest.fixture
def parallel_tool_b():
    return _make_tool("ToolB", is_parallel_execution_allowed=True)


@pytest.fixture
def sequential_tool_x():
    return _make_tool("ToolX", is_parallel_execution_allowed=False)


@pytest.fixture
def sequential_tool_y():
    return _make_tool("ToolY", is_parallel_execution_allowed=False)


def _build_agent(openai_node, mock_llm_executor, tools):
    return Agent(
        name="SplitTestAgent",
        llm=openai_node,
        tools=tools,
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=True,
    )


SINGLE_TOOL_RESULT = ("result", [], False, True, {"node": "dep"})


class TestIsToolParallelEligible:

    def test_eligible_tool_returns_true(self, openai_node, mock_llm_executor, parallel_tool_a):
        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a])
        assert agent._is_tool_parallel_eligible("ToolA") is True

    def test_non_eligible_tool_returns_false(self, openai_node, mock_llm_executor, sequential_tool_x):
        agent = _build_agent(openai_node, mock_llm_executor, [sequential_tool_x])
        assert agent._is_tool_parallel_eligible("ToolX") is False


class TestExecuteToolsSplit:
    """Test that _execute_tools correctly splits tools into parallel and sequential groups."""

    def test_all_parallel_tools_run_concurrently(
        self, openai_node, mock_llm_executor, parallel_tool_a, parallel_tool_b, mocker
    ):
        """When all tools are is_parallel_execution_allowed=True, they should all run via ThreadPoolExecutor."""
        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a, parallel_tool_b])
        mock_single = mocker.patch.object(agent, "_execute_single_tool", return_value=SINGLE_TOOL_RESULT)

        tools_data = [
            {"name": "ToolA", "input": {"x": 1}},
            {"name": "ToolB", "input": {"x": 2}},
        ]
        agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        assert mock_single.call_count == 2
        for c in mock_single.call_args_list:
            assert c.kwargs.get("is_parallel") is True, "Parallel tools must run with is_parallel=True"

    def test_all_sequential_tools_run_one_by_one(
        self, openai_node, mock_llm_executor, sequential_tool_x, sequential_tool_y, mocker
    ):
        """When all tools are is_parallel_execution_allowed=False, they should all run sequentially."""
        agent = _build_agent(openai_node, mock_llm_executor, [sequential_tool_x, sequential_tool_y])
        mock_single = mocker.patch.object(agent, "_execute_single_tool", return_value=SINGLE_TOOL_RESULT)

        tools_data = [
            {"name": "ToolX", "input": {"x": 1}},
            {"name": "ToolY", "input": {"x": 2}},
        ]
        agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        assert mock_single.call_count == 2
        for c in mock_single.call_args_list:
            assert c.kwargs.get("is_parallel", False) is False, "Sequential tools must NOT run with is_parallel=True"

    def test_mixed_tools_split_correctly(
        self,
        openai_node,
        mock_llm_executor,
        parallel_tool_a,
        parallel_tool_b,
        sequential_tool_x,
        sequential_tool_y,
        mocker,
    ):
        """Mixed batch: parallel tools run with is_parallel=True, sequential tools without."""
        agent = _build_agent(
            openai_node,
            mock_llm_executor,
            [parallel_tool_a, parallel_tool_b, sequential_tool_x, sequential_tool_y],
        )
        mock_single = mocker.patch.object(agent, "_execute_single_tool", return_value=SINGLE_TOOL_RESULT)

        tools_data = [
            {"name": "ToolA", "input": {"x": 1}},
            {"name": "ToolB", "input": {"x": 2}},
            {"name": "ToolX", "input": {"x": 3}},
            {"name": "ToolY", "input": {"x": 4}},
        ]
        agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        assert mock_single.call_count == 4

        parallel_calls = [c for c in mock_single.call_args_list if c.kwargs.get("is_parallel") is True]
        sequential_calls = [c for c in mock_single.call_args_list if c.kwargs.get("is_parallel", False) is False]

        assert len(parallel_calls) == 2, "Two parallel-eligible tools should run with is_parallel=True"
        assert len(sequential_calls) == 2, "Two sequential-only tools should run without is_parallel"

        parallel_names = {c.args[0] for c in parallel_calls}
        sequential_names = {c.args[0] for c in sequential_calls}
        assert parallel_names == {"ToolA", "ToolB"}
        assert sequential_names == {"ToolX", "ToolY"}

    def test_single_parallel_tool_runs_without_threadpool(
        self, openai_node, mock_llm_executor, parallel_tool_a, mocker
    ):
        """A single tool in the batch should bypass ThreadPoolExecutor entirely."""
        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a])
        mock_single = mocker.patch.object(agent, "_execute_single_tool", return_value=SINGLE_TOOL_RESULT)

        tools_data = [{"name": "ToolA", "input": {"x": 1}}]
        agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        mock_single.assert_called_once()
        assert mock_single.call_args.kwargs.get("is_parallel", False) is False

    def test_results_preserve_original_order(
        self, openai_node, mock_llm_executor, parallel_tool_a, sequential_tool_x, mocker
    ):
        """Results should be ordered by the original tool order, not execution order."""
        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a, sequential_tool_x])

        call_count = 0

        def mock_execute(name, inp, thought, loop, config, **kwargs):
            nonlocal call_count
            call_count += 1
            return f"result-{name}", [], False, True, {"node": f"dep-{name}"}

        mocker.patch.object(agent, "_execute_single_tool", side_effect=mock_execute)

        tools_data = [
            {"name": "ToolX", "input": {"x": 1}},
            {"name": "ToolA", "input": {"x": 2}},
        ]
        observation, files = agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        lines = observation.strip().split("\n\n")
        assert "ToolX" in lines[0], "ToolX (order=0) should appear first in output"
        assert "ToolA" in lines[1], "ToolA (order=1) should appear second in output"

    def test_one_parallel_one_sequential_split(
        self, openai_node, mock_llm_executor, parallel_tool_a, sequential_tool_x, mocker
    ):
        """With one parallel + one sequential: parallel runs as single (no threadpool), sequential runs after."""
        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a, sequential_tool_x])
        mock_single = mocker.patch.object(agent, "_execute_single_tool", return_value=SINGLE_TOOL_RESULT)

        tools_data = [
            {"name": "ToolA", "input": {"x": 1}},
            {"name": "ToolX", "input": {"x": 2}},
        ]
        agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        assert mock_single.call_count == 2

    def test_empty_tools_data_returns_empty(self, openai_node, mock_llm_executor):
        agent = _build_agent(openai_node, mock_llm_executor, [])
        observation, files = agent._execute_tools([], "thinking", 1, RunnableConfig())
        assert observation == ""
        assert files == {}

    def test_invalid_tool_payload_produces_error_result(self, openai_node, mock_llm_executor, parallel_tool_a, mocker):
        """Missing 'name' or 'input' in a payload should produce an error entry, not crash."""
        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a])
        mock_single = mocker.patch.object(agent, "_execute_single_tool", return_value=SINGLE_TOOL_RESULT)

        tools_data = [
            {"name": "ToolA"},  # missing 'input'
            {"name": "ToolA", "input": {"x": 1}},
        ]
        observation, _ = agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        assert "ERROR" in observation, "Invalid payload should appear as ERROR in observation"
        mock_single.assert_called_once()

    def test_logging_for_sequential_tools(
        self, openai_node, mock_llm_executor, parallel_tool_a, sequential_tool_x, mocker, caplog
    ):
        """Agent should log which tools are excluded from parallel execution."""
        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a, sequential_tool_x])
        mocker.patch.object(agent, "_execute_single_tool", return_value=SINGLE_TOOL_RESULT)

        tools_data = [
            {"name": "ToolA", "input": {"x": 1}},
            {"name": "ToolX", "input": {"x": 2}},
        ]

        import logging

        with caplog.at_level(logging.INFO):
            agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        assert any(
            "is_parallel_execution_allowed=False" in record.message for record in caplog.records
        ), "Should log which tools are excluded from parallel execution"
