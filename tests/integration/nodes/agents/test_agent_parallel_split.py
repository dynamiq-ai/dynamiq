"""Tests for the parallel/sequential tool split in _execute_tools."""

import uuid

import pytest

from dynamiq import connections, prompts
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus


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


class TestBatchStreamEvents:
    """Test that multi-tool batches stream batch reasoning and completion events."""

    def test_multi_tool_batch_streams_batch_and_individual_reasoning(
        self, openai_node, mock_llm_executor, parallel_tool_a, parallel_tool_b, mocker
    ):
        """A multi-tool batch should emit a batch run_parallel reasoning event
        as an extra message, while each tool also emits its own reasoning event."""
        from dynamiq.nodes.tools.parallel_tool_calls import PARALLEL_TOOL_NAME

        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a, parallel_tool_b])
        mock_single = mocker.patch.object(agent, "_execute_single_tool", return_value=SINGLE_TOOL_RESULT)
        mock_stream = mocker.patch.object(agent, "_stream_agent_event")

        tools_data = [
            {"name": "ToolA", "input": {"x": 1}},
            {"name": "ToolB", "input": {"x": 2}},
        ]
        agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        # Two _stream_agent_event calls: batch reasoning + batch completion
        assert mock_stream.call_count == 2

        reasoning_event = mock_stream.call_args_list[0][0][0]
        assert reasoning_event.action == PARALLEL_TOOL_NAME
        assert isinstance(reasoning_event.action_input, list)
        assert len(reasoning_event.action_input) == 2

        # Individual per-tool reasoning events are emitted inside _execute_single_tool
        assert mock_single.call_count == 2
        for c in mock_single.call_args_list:
            assert c.kwargs.get("tool_run_id") is not None

    def test_batch_completion_event_emitted_after_tools_finish(
        self, openai_node, mock_llm_executor, parallel_tool_a, parallel_tool_b, mocker
    ):
        """After all parallel tools finish, a run_parallel tool-result event should be emitted
        with per-tool status summaries and no actual results."""
        from dynamiq.nodes.tools.parallel_tool_calls import PARALLEL_TOOL_NAME
        from dynamiq.types.streaming import AgentToolResultEventMessageData

        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a, parallel_tool_b])
        mocker.patch.object(agent, "_execute_single_tool", return_value=SINGLE_TOOL_RESULT)
        mock_stream = mocker.patch.object(agent, "_stream_agent_event")

        tools_data = [
            {"name": "ToolA", "input": {"x": 1}},
            {"name": "ToolB", "input": {"x": 2}},
        ]
        agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        # Second call is the completion event
        completion_event = mock_stream.call_args_list[1][0][0]
        completion_step = mock_stream.call_args_list[1][0][1]

        assert isinstance(completion_event, AgentToolResultEventMessageData)
        assert completion_step == "tool"
        assert completion_event.name == PARALLEL_TOOL_NAME
        assert completion_event.tool.name == PARALLEL_TOOL_NAME
        assert completion_event.tool.action_type == "parallel_execution"
        assert completion_event.status == RunnableStatus.SUCCESS

        # result is a list of per-tool summaries
        assert isinstance(completion_event.result, list)
        assert len(completion_event.result) == 2
        tool_names = {entry["name"] for entry in completion_event.result}
        assert tool_names == {"ToolA", "ToolB"}
        for entry in completion_event.result:
            assert entry["result"] is None
            assert entry["status"] == RunnableStatus.SUCCESS
            assert "tool_run_id" in entry

    def test_batch_completion_reports_failure(
        self, openai_node, mock_llm_executor, parallel_tool_a, parallel_tool_b, mocker
    ):
        """If one tool fails, the batch completion status should be failure."""
        from dynamiq.types.streaming import AgentToolResultEventMessageData

        agent = _build_agent(openai_node, mock_llm_executor, [parallel_tool_a, parallel_tool_b])
        success_result = ("ok", [], False, True, {"node": "dep"})
        failure_result = ("error msg", [], False, False, None)

        call_count = 0

        def mock_execute(name, inp, thought, loop, config, **kwargs):
            nonlocal call_count
            call_count += 1
            if name == "ToolA":
                return success_result
            return failure_result

        mocker.patch.object(agent, "_execute_single_tool", side_effect=mock_execute)
        mock_stream = mocker.patch.object(agent, "_stream_agent_event")

        tools_data = [
            {"name": "ToolA", "input": {"x": 1}},
            {"name": "ToolB", "input": {"x": 2}},
        ]
        agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        completion_event = mock_stream.call_args_list[1][0][0]
        assert isinstance(completion_event, AgentToolResultEventMessageData)
        assert completion_event.status == RunnableStatus.FAILURE

        statuses = {entry["name"]: entry["status"] for entry in completion_event.result}
        assert statuses["ToolA"] == RunnableStatus.SUCCESS
        assert statuses["ToolB"] == RunnableStatus.FAILURE
