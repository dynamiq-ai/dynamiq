"""What an MCP session follows and when it is opened."""

import contextlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamiq.connections.connections import MCPSse
from dynamiq.nodes.tools.mcp import MCPServer, MCPTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import CanceledException, CancellationConfig, CancellationToken

CONNECTION = MCPSse(url="https://mcp.example.com/sse")


def mcp_tool(name: str, properties: dict, required: list[str] | None = None, **schema) -> MCPTool:
    json_schema = {"type": "object", "properties": properties, "required": required or [], **schema}
    return MCPTool(name=name, description=f"{name} tool", json_input_schema=json_schema, connection=CONNECTION)


def test_discovery_stops_when_the_server_repeats_a_page_cursor():
    server = MCPServer(connection=CONNECTION)
    page = SimpleNamespace(tools=[], nextCursor="same")
    session = MagicMock()
    session.initialize = AsyncMock()
    session.list_tools = AsyncMock(return_value=page)

    @contextlib.asynccontextmanager
    async def fake_connect(self, headers=None):
        yield (object(), object())

    @contextlib.asynccontextmanager
    async def fake_session(*args, **kwargs):
        yield session

    with patch.object(MCPSse, "connect", fake_connect), patch("mcp.ClientSession", fake_session):
        tools = server.get_mcp_tools()

    assert tools == []
    assert session.list_tools.await_count <= 3


@pytest.mark.asyncio
async def test_a_canceled_run_does_not_call_the_mcp_server():
    tool = mcp_tool("search", {"query": {"type": "string"}}, ["query"])
    token = CancellationToken()
    token.cancel()
    config = RunnableConfig(callbacks=[], cancellation=CancellationConfig(token=token))
    connect = MagicMock()

    with patch.object(MCPSse, "connect", connect), pytest.raises(CanceledException):
        await tool.execute_async(tool.input_schema(query="x"), config=config)

    connect.assert_not_called()
