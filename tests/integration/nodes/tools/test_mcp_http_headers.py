"""HTTP headers an MCP call sends."""

import contextlib
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.connections import MCPSse, merge_mcp_http_headers
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.base import ToolParams
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.mcp import MCPServer
from dynamiq.nodes.types import InferenceMode
from tests.helpers.agent_hooks import final_step, make_llm, run_scripted, tool_step

CONNECTION = MCPSse(url="https://mcp.example.com/sse")
SERVER_ID = "github-server"


def test_a_per_run_header_replaces_the_connection_header_whatever_its_case():
    # HTTP header names are case-insensitive: sending both, the server may authenticate with the static one.
    connection_headers = {"Authorization": "Bearer service", "Accept": "*/*"}
    merged = merge_mcp_http_headers(connection_headers, {"authorization": "Bearer user"})

    assert merged == {"Accept": "*/*", "authorization": "Bearer user"}


def test_headers_from_several_tool_params_layers_are_sent_once_per_name():
    server = MCPServer(connection=CONNECTION, discover_per_run=True)
    llm = OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o")
    agent = Agent(name="a", llm=llm, tools=[server])
    params = ToolParams.model_validate(
        {
            "global": {"mcp_http_headers": {"Authorization": "Bearer service"}},
            "by_id": {server.id: {"mcp_http_headers": {"authorization": "Bearer user"}}},
        }
    )

    headers = agent._mcp_run_headers(server, params)

    assert headers == {"authorization": "Bearer user"}


def per_run_agent(connection_headers: dict | None = None) -> Agent:
    connection = MCPSse(url="https://mcp.example.com/sse", headers=connection_headers)
    server = MCPServer(id=SERVER_ID, name="github", connection=connection, discover_per_run=True)
    return Agent(name="a", llm=make_llm(), tools=[server], inference_mode=InferenceMode.FUNCTION_CALLING)


def sent_headers(agent: Agent, tool_params: dict) -> list[dict]:
    """The headers of each MCP session in one run that lists the tools, then calls `search`."""
    sent = []

    @contextlib.asynccontextmanager
    async def sse_client(url, headers=None, **kwargs):
        sent.append(dict(headers or {}))
        yield (None, None)

    class Session:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def initialize(self):
            return None

        async def list_tools(self, params=None):
            schema = {"type": "object", "properties": {"q": {"type": "string"}}}
            return ListToolsResult(tools=[Tool(name="search", description="Searches.", inputSchema=schema)])

        async def call_tool(self, name, arguments):
            return CallToolResult(content=[TextContent(type="text", text="found")])

    steps = [tool_step("search", q="bug"), final_step("done")]
    with patch("mcp.client.sse.sse_client", sse_client), patch("mcp.ClientSession", Session):
        run_scripted(agent, InferenceMode.FUNCTION_CALLING, steps, input_extra={"tool_params": tool_params})
    return sent


@pytest.mark.parametrize(
    "tool_params",
    [
        {
            "global": {"mcp_http_headers": {"Authorization": "Bearer global"}},
            "by_id": {SERVER_ID: {"mcp_http_headers": {"authorization": "Bearer by-id"}}},
        },
        {
            "global": {"mcp_http_headers": {"authorization": "Bearer global"}},
            "by_name": {"github": {"mcp_http_headers": {"Authorization": "Bearer by-name"}}},
            "by_id": {SERVER_ID: {"mcp_http_headers": {"authorization": "Bearer by-id"}}},
        },
    ],
    ids=["two-layers", "three-layers-alternating-case"],
)
def test_a_tool_call_sends_the_authorization_its_tool_was_listed_with(tool_params):
    agent = per_run_agent(connection_headers={"Authorization": "Bearer service-account"})

    listing, call = sent_headers(agent, tool_params)

    assert listing == call == {"authorization": "Bearer by-id"}


def test_tool_call_header_layers_keep_their_order():
    agent = per_run_agent()
    tool_params = {
        "global": {"mcp_http_headers": {"X-A": "global", "X-B": "global", "X-C": "global"}},
        "by_name": {
            "github": {"mcp_http_headers": {"X-B": "server name", "X-C": "server name"}},
            "search": {"mcp_http_headers": {"X-C": "tool name"}},
        },
        "by_id": {SERVER_ID: {"mcp_http_headers": {"X-D": "server id"}}},
    }

    listing, call = sent_headers(agent, tool_params)

    assert listing == {"X-A": "global", "X-B": "server name", "X-C": "server name", "X-D": "server id"}
    assert call == {"X-A": "global", "X-B": "server name", "X-C": "tool name", "X-D": "server id"}
