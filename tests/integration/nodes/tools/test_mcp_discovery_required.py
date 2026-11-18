import asyncio
import contextlib
import json
import uuid
from unittest.mock import patch

import pytest
from litellm import ModelResponse
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from dynamiq import Workflow, connections
from dynamiq.connections import MCPStreamableHTTP
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.mcp import MCPServer
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus

GOOD = {"Authorization": "Bearer live-token"}
EXPIRED = {"Authorization": "Bearer expired-token"}


class FlakyMCPServer:
    """An MCP server that refuses a session without the live token, and every session while it is down."""

    def __init__(self):
        self.down = False
        self.listing_fails = False
        self.hangs: asyncio.Event | None = None
        self.sessions: list[dict | None] = []

    @contextlib.contextmanager
    def serving(self):
        server = self

        @contextlib.asynccontextmanager
        async def connect(_connection, headers=None):
            server.sessions.append(headers)
            if server.down:
                raise ConnectionError("All connection attempts failed")
            sent = {**(_connection.headers or {}), **(headers or {})}
            if sent.get("Authorization") != GOOD["Authorization"]:
                raise PermissionError("Client error '401 Unauthorized' for url 'https://tracker.example.com/mcp'")
            yield (object(), object())

        class Session:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def initialize(self):
                if server.hangs:
                    server.hangs.set()
                    await asyncio.Event().wait()

            async def list_tools(self, params=None):
                if server.listing_fails:
                    raise RuntimeError("Connection closed")
                schema = {"type": "object", "properties": {"task": {"type": "string"}}}
                return ListToolsResult(tools=[Tool(name="find_task", description="Finds a task.", inputSchema=schema)])

            async def call_tool(self, name, arguments):
                return CallToolResult(content=[TextContent(type="text", text="status of task 7: in review")])

        with (
            patch.object(MCPStreamableHTTP, "connect", new=connect),
            patch("mcp.ClientSession", new=Session),
        ):
            yield self


@pytest.fixture
def llm():
    return OpenAI(model="gpt-4o-mini", connection=connections.OpenAI(id=str(uuid.uuid4()), api_key="key"))


@pytest.fixture
def tracker():
    return FlakyMCPServer()


@pytest.fixture
def calculator():
    code = "def run(params: dict):\n    return {'sum': 'five (from the add tool)'}\n"
    return Python(name="add", description="Adds two numbers.", code=code)


def tracker_server(**kwargs):
    return MCPServer(name="tracker", connection=MCPStreamableHTTP(url="https://tracker.example.com/mcp"), **kwargs)


def script_llm(mocker, replies):
    """Answer the agent's LLM calls with ``replies`` in order, and return the prompts it sent."""
    prompts = []

    def respond(*args, **kwargs):
        prompts.append(json.dumps(kwargs.get("messages"), default=str))
        response = ModelResponse()
        response["choices"][0]["message"]["content"] = replies.pop(0)
        return response

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=respond)
    return prompts


def run_with(agent, headers):
    return agent.run(
        {"input": "What is 2 + 3?", "tool_params": {"by_name": {"tracker": {"mcp_http_headers": headers}}}}
    )


def test_a_failed_discovery_fails_the_build_by_default(tracker, llm, calculator):
    tracker.down = True
    with tracker.serving(), pytest.raises(ToolExecutionException, match="failed to initialize session"):
        Agent(name="assistant", llm=llm, tools=[calculator, tracker_server()])


def test_a_failed_per_run_discovery_fails_the_run_by_default(tracker, llm, calculator, mocker):
    script_llm(mocker, ["Thought: I know this.\nAnswer: 5"])
    with tracker.serving():
        agent = Agent(
            name="assistant",
            llm=llm,
            tools=[calculator, tracker_server(discover_per_run=True)],
            inference_mode=InferenceMode.DEFAULT,
        )
        result = run_with(agent, EXPIRED)

    assert result.status == RunnableStatus.FAILURE
    assert "failed to initialize session" in result.error.message


def test_a_server_that_is_not_required_is_left_out_of_the_build_when_discovery_fails(tracker, llm, calculator):
    tracker.down = True
    server = tracker_server(discovery_required=False)
    with tracker.serving(), patch("dynamiq.nodes.tools.mcp.logger.warning") as warning:
        agent = Agent(name="assistant", llm=llm, tools=[calculator, server])

    assert agent.tool_names == "add"
    assert len(tracker.sessions) == 1
    messages = [str(call.args[0]) for call in warning.call_args_list]
    assert any(
        "tracker" in message and server.id in message and "All connection attempts failed" in message
        for message in messages
    )


def test_a_failed_build_time_discovery_is_tried_again_by_the_next_agent(tracker, llm, calculator):
    """The failure is not cached on the server, so an agent built once the outage is over gets its tools."""
    server = MCPServer(
        name="tracker",
        connection=MCPStreamableHTTP(url="https://tracker.example.com/mcp", headers=GOOD),
        discovery_required=False,
    )
    with tracker.serving():
        tracker.down = True
        during_outage = Agent(name="assistant", llm=llm, tools=[calculator, server])
        tracker.down = False
        after = Agent(name="assistant", llm=llm, tools=[calculator, server])

    assert during_outage.tool_names == "add"
    assert after.tool_names == "add,find_task"
    assert len(tracker.sessions) == 2


@pytest.mark.parametrize(
    ("headers", "down"),
    [
        pytest.param(EXPIRED, False, id="an expired token"),
        pytest.param(None, False, id="no token"),
        pytest.param(GOOD, True, id="the server is down"),
    ],
)
def test_a_run_goes_on_with_its_other_tools_when_a_server_that_is_not_required_fails(
    tracker, llm, calculator, mocker, headers, down
):
    prompts = script_llm(
        mocker,
        [
            "Thought: Add them.\nAction: add\nAction Input: " + json.dumps({"a": 2, "b": 3}),
            "Thought: Done.\nAnswer: 5",
        ],
    )
    tracker.down = down
    with tracker.serving():
        agent = Agent(
            name="assistant",
            llm=llm,
            tools=[calculator, tracker_server(discover_per_run=True, discovery_required=False)],
            inference_mode=InferenceMode.DEFAULT,
        )
        result = run_with(agent, headers)

    assert result.status == RunnableStatus.SUCCESS
    assert tracker.sessions == [headers]
    assert "find_task" not in prompts[0]
    # The agent's other tool was offered and ran: its result is what the second LLM call was given.
    assert "five (from the add tool)" in prompts[1]


def test_the_run_after_a_failed_discovery_lists_the_tools_again(tracker, llm, calculator, mocker):
    """One caller's expired token, or an outage that ended, costs the next run nothing."""
    prompts = script_llm(
        mocker,
        [
            "Thought: I know this.\nAnswer: 5",
            "Thought: Look it up.\nAction: find_task\nAction Input: " + json.dumps({"task": "7"}),
            "Thought: Done.\nAnswer: It is in review.",
        ],
    )
    with tracker.serving():
        agent = Agent(
            name="assistant",
            llm=llm,
            tools=[calculator, tracker_server(discover_per_run=True, discovery_required=False)],
            inference_mode=InferenceMode.DEFAULT,
        )
        first = run_with(agent, EXPIRED)
        second = run_with(agent, GOOD)

    assert first.status == RunnableStatus.SUCCESS
    assert "find_task" not in prompts[0]
    assert second.status == RunnableStatus.SUCCESS
    assert "find_task" in prompts[1]
    assert "status of task 7: in review" in prompts[2]


@pytest.mark.asyncio
async def test_a_failure_after_the_session_opened_is_forgiven_too(tracker):
    tracker.listing_fails = True
    server = tracker_server(discover_per_run=True, discovery_required=False)
    with tracker.serving():
        tools = await server.get_run_tools_async(GOOD)

    assert tools == []
    assert tracker.sessions == [GOOD]


@pytest.mark.asyncio
async def test_cancelling_a_discovery_that_is_not_required_still_cancels_it(tracker):
    """Only a failure is forgiven: a run that is cancelled while it lists the tools must stop."""
    tracker.hangs = asyncio.Event()
    server = tracker_server(discover_per_run=True, discovery_required=False)
    with tracker.serving():
        listing = asyncio.create_task(server.get_run_tools_async(GOOD))
        await asyncio.wait_for(tracker.hangs.wait(), timeout=5)
        listing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await listing


def test_discovery_required_survives_a_yaml_round_trip(tracker, llm, calculator, tmp_path):
    tracker.down = True
    with tracker.serving():
        agent = Agent(name="assistant", llm=llm, tools=[calculator, tracker_server(discovery_required=False)])
        path = tmp_path / "workflow.yaml"
        Workflow(flow=Flow(nodes=[agent])).to_yaml_file(path)
        loaded = Workflow.from_yaml_file(str(path), init_components=True)

    assert [server.discovery_required for server in loaded.flow.nodes[0]._mcp_servers] == [False]
    assert "discovery_required: false" in path.read_text()
