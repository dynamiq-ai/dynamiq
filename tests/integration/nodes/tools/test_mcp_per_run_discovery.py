import contextlib
import json
import uuid
from unittest.mock import patch

import pytest
from litellm import ModelResponse
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from dynamiq import connections
from dynamiq.connections import MCPStreamableHTTP
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.base import ToolParams
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.mcp import MCPServer
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus

RUN_HEADERS = {"Authorization": "Bearer agent-sa-id-token", "X-User-Id": "signed-user-1"}


class FakeMCPServer:
    """An MCP server that lists tools per caller, recording the headers and requests of every session."""

    def __init__(self, tools_by_authorization: dict[str | None, list[str]]):
        self.tools_by_authorization = tools_by_authorization
        self.sessions: list[dict] = []

    @contextlib.contextmanager
    def serving(self):
        server = self

        @contextlib.asynccontextmanager
        async def connect(_connection, headers=None):
            server.sessions.append({"headers": headers, "requests": []})
            yield (object(), object())

        class Session:
            def __init__(self, *args, **kwargs):
                self.record = server.sessions[-1]

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                return False

            async def initialize(self):
                self.record["requests"].append("initialize")

            async def list_tools(self, params=None):
                self.record["requests"].append("tools/list")
                authorization = (self.record["headers"] or {}).get("Authorization")
                names = server.tools_by_authorization.get(authorization, [])
                schema = {"type": "object", "properties": {"employee": {"type": "string"}}}
                return ListToolsResult(tools=[Tool(name=name, description=name, inputSchema=schema) for name in names])

            async def call_tool(self, name, arguments):
                self.record["requests"].append("tools/call")
                return CallToolResult(content=[TextContent(type="text", text=f"{name}: Alex Kim")])

        with (
            patch.object(MCPStreamableHTTP, "connect", new=connect),
            patch("mcp.ClientSession", new=Session),
        ):
            yield self


@pytest.fixture
def llm():
    return OpenAI(model="gpt-4o-mini", connection=connections.OpenAI(id=str(uuid.uuid4()), api_key="key"))


@pytest.fixture
def teammate():
    return FakeMCPServer({"Bearer agent-sa-id-token": ["get_manager"], None: ["list_everything"]})


def per_run_server(**kwargs):
    return MCPServer(
        name="mcp-server",
        connection=MCPStreamableHTTP(url="https://teammate.example.com/mcp"),
        discover_per_run=True,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_initialize_tools_sends_the_given_headers(teammate):
    server = MCPServer(connection=MCPStreamableHTTP(url="https://teammate.example.com/mcp"))
    with teammate.serving():
        await server.initialize_tools(headers=RUN_HEADERS)

    assert teammate.sessions == [{"headers": RUN_HEADERS, "requests": ["initialize", "tools/list"]}]
    assert set(server._mcp_tools) == {"get_manager"}


def test_building_the_agent_does_not_list_a_per_run_servers_tools(teammate, llm):
    with teammate.serving():
        agent = Agent(name="hr", llm=llm, tools=[per_run_server()])

    assert teammate.sessions == []
    assert agent.tool_names == ""


def test_a_build_time_server_is_still_listed_once_without_run_headers(teammate, llm):
    server = MCPServer(name="mcp-server", connection=MCPStreamableHTTP(url="https://teammate.example.com/mcp"))
    with teammate.serving():
        agent = Agent(name="hr", llm=llm, tools=[server])
        per_run = agent._list_per_run_mcp_tools({"by_name": {"mcp-server": {"mcp_http_headers": RUN_HEADERS}}})

    assert teammate.sessions == [{"headers": None, "requests": ["initialize", "tools/list"]}]
    assert agent.tool_names == "list_everything"
    assert per_run == []


def test_per_run_tools_are_listed_with_the_runs_headers_and_not_cached(teammate, llm):
    server = per_run_server()
    with teammate.serving():
        agent = Agent(name="hr", llm=llm, tools=[server])
        tools = agent._list_per_run_mcp_tools(
            ToolParams.model_validate({"by_name": {"mcp-server": {"mcp_http_headers": RUN_HEADERS}}})
        )
        anonymous = agent._list_per_run_mcp_tools(None)

    assert [tool.name for tool in tools] == ["get_manager"]
    assert tools[0]._owner_server is server
    assert [tool.name for tool in anonymous] == ["list_everything"]
    assert [session["headers"] for session in teammate.sessions] == [RUN_HEADERS, None]
    assert server._mcp_tools == {}


def test_per_run_tools_respect_include_tools(llm):
    teammate = FakeMCPServer({"Bearer agent-sa-id-token": ["get_manager", "delete_employee"]})
    with teammate.serving():
        agent = Agent(name="hr", llm=llm, tools=[per_run_server(include_tools=["get_manager"])])
        tools = agent._list_per_run_mcp_tools({"by_name": {"mcp-server": {"mcp_http_headers": RUN_HEADERS}}})

    assert [tool.name for tool in tools] == ["get_manager"]


def test_per_run_headers_layer_global_then_name_then_id(llm):
    server = per_run_server()
    agent = Agent(name="hr", llm=llm, tools=[server])
    tool_params = ToolParams.model_validate(
        {
            "global": {"mcp_http_headers": {"X-A": "global", "X-B": "global"}},
            "by_name": {"mcp-server": {"mcp_http_headers": {"X-B": "name", "X-C": "name"}}},
            "by_id": {server.id: {"mcp_http_headers": {"X-C": "id"}}},
        }
    )

    assert agent._mcp_run_headers(server, tool_params) == {"X-A": "global", "X-B": "name", "X-C": "id"}


def test_per_run_headers_skip_by_name_when_two_servers_share_the_name(llm):
    """A credential keyed by a shared name could be meant for either host, so only the id selects."""
    first, second = per_run_server(), per_run_server()
    agent = Agent(name="hr", llm=llm, tools=[first, second])
    tool_params = ToolParams.model_validate(
        {
            "by_name": {"mcp-server": {"mcp_http_headers": {"Authorization": "Bearer meant-for-one"}}},
            "by_id": {second.id: {"mcp_http_headers": {"Authorization": "Bearer second"}}},
        }
    )

    with patch("dynamiq.nodes.agents.base.logger.warning") as warning:
        assert agent._mcp_run_headers(first, tool_params) is None
    assert agent._mcp_run_headers(second, tool_params) == {"Authorization": "Bearer second"}
    assert any("ambiguous" in str(call.args[0]) for call in warning.call_args_list if call.args)


def test_discover_per_run_survives_serialization(llm):
    agent = Agent(name="hr", llm=llm, tools=[per_run_server()])

    servers = [tool for tool in agent.to_dict()["tools"] if tool["type"] == "dynamiq.nodes.tools.MCPServer"]

    assert len(servers) == 1
    assert servers[0]["discover_per_run"] is True


def test_a_run_sends_the_token_when_listing_tools_and_when_calling_one(teammate, llm, mocker):
    replies = [
        "Thought: Look up the manager.\nAction: get_manager\nAction Input: " + json.dumps({"employee": "me"}),
        "Thought: Done.\nAnswer: Your manager is Alex Kim.",
    ]
    prompts = []

    def respond(*args, **kwargs):
        prompts.append(json.dumps(kwargs.get("messages"), default=str))
        response = ModelResponse()
        response["choices"][0]["message"]["content"] = replies.pop(0)
        return response

    mocker.patch("dynamiq.nodes.llms.base.BaseLLM._completion", side_effect=respond)

    with teammate.serving():
        agent = Agent(name="hr", llm=llm, tools=[per_run_server()], inference_mode=InferenceMode.DEFAULT)
        result = agent.run(
            {
                "input": "Who is my manager?",
                "tool_params": {"by_name": {"mcp-server": {"mcp_http_headers": RUN_HEADERS}}},
            }
        )

    assert result.status == RunnableStatus.SUCCESS
    assert teammate.sessions == [
        {"headers": RUN_HEADERS, "requests": ["initialize", "tools/list"]},
        {"headers": RUN_HEADERS, "requests": ["initialize", "tools/call"]},
    ]
    assert "get_manager" in prompts[0]
    assert "list_everything" not in prompts[0]
    assert agent.tool_names == ""
