import uuid
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import Field, create_model

from dynamiq import connections
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.mcp_adapter import MCPServerAdapter, MCPSse, MCPTool, ToolFilterMode


def assert_tool_matches(tool, expected, connection):
    assert tool.name == expected["name"]
    assert tool.description == expected["description"]
    assert tool.connection == connection
    assert set(tool.input_schema.__annotations__.items()) == expected["schema"]


def mock_get_input_schema(schema_dict: dict[str, Any]):
    """
    Creates an BaseClass based on provided JSON schema.

    Args:
        schema_dict (dict[str, Any]): A JSON schema dictionary describing the tool's expected input.
    """
    fields = {}
    props = schema_dict.get("properties", {})
    required = set(schema_dict.get("required", []))
    json_mapping = {
        "string": "str",
        "integer": "int",
        "number": "float",
        "boolean": "bool",
        "object": "dict",
        "array": "list",
    }
    for field_name, field_spec in props.items():
        field_type = json_mapping.get(field_spec.get("type", "string"))
        default = ... if field_name in required else None
        description = field_spec.get("description", None)
        fields[field_name] = (field_type, Field(default, description=description))
    return create_model(schema_dict.get("title", "MCPToolSchema"), **fields)


@pytest.fixture
def llm_model():
    connection = connections.OpenAI(id=str(uuid.uuid4()), api_key="api-key")
    return OpenAI(name="OpenAI", model="gpt-4o-mini", connection=connection)


@pytest.fixture
def sse_server_connection():
    return MCPSse(url="https://example.com/")


@pytest.fixture
def mock_mcp_tools(sse_server_connection):
    with patch.object(MCPTool, "get_input_schema", side_effect=mock_get_input_schema):
        return {
            "add": MCPTool(
                name="add",
                description="Add two numbers",
                json_input_schema={
                    "title": "AddSchema",
                    "type": "object",
                    "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                    "required": ["a", "b"],
                },
                connection=sse_server_connection,
            ),
            "multiply": MCPTool(
                name="multiply",
                description="Multiply two numbers",
                json_input_schema={
                    "title": "MultiplySchema",
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
                connection=sse_server_connection,
            ),
            "subtract": MCPTool(
                name="subtract",
                description="Subtract two numbers",
                json_input_schema={
                    "title": "SubtractSchema",
                    "type": "object",
                    "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                    "required": ["a", "b"],
                },
                connection=sse_server_connection,
            ),
        }


@pytest.fixture
def mcp_adapter_tool(sse_server_connection, mock_mcp_tools):
    tool = MCPServerAdapter(connection=sse_server_connection)
    tool._mcp_tools = mock_mcp_tools
    return tool


@pytest.mark.asyncio
async def test_get_mcp_tools(mcp_adapter_tool, sse_server_connection):
    with patch.object(MCPServerAdapter, "initialize_tools", new=AsyncMock()) as mock_init:
        tools = mcp_adapter_tool.get_mcp_tools()
        mock_init.assert_not_awaited()

        expected_tools = [
            {"name": "add", "description": "Add two numbers", "schema": {("a", "int"), ("b", "int")}},
            {"name": "multiply", "description": "Multiply two numbers", "schema": {("a", "float"), ("b", "float")}},
            {"name": "subtract", "description": "Subtract two numbers", "schema": {("a", "float"), ("b", "float")}},
        ]

        assert len(tools) == len(expected_tools)

        for tool, expected in zip(tools, expected_tools):
            assert_tool_matches(tool, expected, sse_server_connection)


@pytest.mark.asyncio
async def test_mock_tool_execute(mcp_adapter_tool):
    tool = mcp_adapter_tool._mcp_tools["add"]  # "add"

    mocked_result = {"content": {"result": 42}}

    with patch("dynamiq.nodes.tools.mcp_adapter.MCPTool.execute", return_value=mocked_result) as mock_exec:
        result = tool.execute(tool.input_schema(a=20, b=22))
        mock_exec.assert_called_once()
        assert result == mocked_result

    with patch("dynamiq.nodes.tools.mcp_adapter.MCPTool.execute", return_value=mocked_result) as mock_exec:
        result = await tool.run(input_data={"a": 20, "b": 22})
        mock_exec.assert_called_once()
        assert result.output == mocked_result


def test_mcp_tool_filter_names(mcp_adapter_tool, mock_mcp_tools):
    mcp_adapter_tool._mcp_tools = mock_mcp_tools

    mcp_adapter_tool.tool_filter_names = ["add", "multiply"]
    mcp_adapter_tool.tool_filter_mode = ToolFilterMode.INCLUDE
    tools = mcp_adapter_tool.get_mcp_tools()
    assert len(tools) == 2
    assert tools[0].name == "add" and tools[1].name == "multiply"

    mcp_adapter_tool.tool_filter_mode = ToolFilterMode.EXCLUDE
    tools = mcp_adapter_tool.get_mcp_tools()
    assert len(tools) == 1
    assert tools[0].name == "subtract"

    mcp_adapter_tool.tool_filter_names = ["multiply"]
    mcp_adapter_tool.tool_filter_mode = ToolFilterMode.INCLUDE
    tools = mcp_adapter_tool.get_mcp_tools()
    assert len(tools) == 1
    assert tools[0].name == "multiply"

    mcp_adapter_tool.tool_filter_mode = ToolFilterMode.EXCLUDE
    tools = mcp_adapter_tool.get_mcp_tools()
    assert len(tools) == 2
    assert tools[0].name == "add" and tools[1].name == "subtract"

    tools = mcp_adapter_tool.get_mcp_tools(select_all=True)
    assert len(tools) == 3


def test_agent_integration_with_mcp_tools(sse_server_connection, mock_mcp_tools, llm_model):
    mcp_server = MCPServerAdapter(connection=sse_server_connection)
    mcp_server._mcp_tools = {"add": mock_mcp_tools["add"], "multiply": mock_mcp_tools["multiply"]}
    mcp_tool = mock_mcp_tools["subtract"]

    agent = ReActAgent(llm=llm_model, tools=[mcp_server, mcp_tool])

    agent_tools = agent.tools
    expected_tools = [
        {"name": "add", "description": "Add two numbers", "schema": {("a", "int"), ("b", "int")}},
        {"name": "multiply", "description": "Multiply two numbers", "schema": {("a", "float"), ("b", "float")}},
        {"name": "subtract", "description": "Subtract two numbers", "schema": {("a", "float"), ("b", "float")}},
    ]

    assert len(agent_tools) == 3
    assert all(isinstance(tool, MCPTool) for tool in agent_tools)
    for tool, expected in zip(agent_tools, expected_tools):
        assert_tool_matches(tool, expected, sse_server_connection)

    dict_tools = agent.to_dict()["tools"]
    assert len(dict_tools) == 2
    assert dict_tools[0]["name"] == "subtract"
    assert dict_tools[0]["type"] == "dynamiq.nodes.tools.MCPTool"
    assert dict_tools[1]["type"] == "dynamiq.nodes.tools.MCPServerAdapter"
