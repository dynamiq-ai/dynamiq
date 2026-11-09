"""MCP tool input schemas as the model sees them."""

import contextlib
import json
from unittest.mock import patch

import pytest
from mcp.types import CallToolResult, TextContent

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.connections import MCPSse
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.components import schema_generator
from dynamiq.nodes.images.edit import ImageEditInputSchema
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.llms._strict import to_openai_strict_function, to_strict_subset_function
from dynamiq.nodes.tools.mcp import MCPTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus
from tests.helpers.agent_hooks import final_step, run_scripted, tool_step

CONNECTION = MCPSse(url="https://mcp.example.com/sse")


def mcp_tool(name: str, properties: dict, required: list[str] | None = None, **schema) -> MCPTool:
    json_schema = {"type": "object", "properties": properties, "required": required or [], **schema}
    return MCPTool(name=name, description=f"{name} tool", json_input_schema=json_schema, connection=CONNECTION)


def function_parameters(tool: MCPTool) -> dict:
    schemas = schema_generator.generate_function_calling_schemas([tool], False, lambda name: name)
    return next(s for s in schemas if s["function"]["name"] == tool.name)["function"]["parameters"]


@pytest.mark.parametrize(
    "prop",
    [{"description": "Any JSON value."}, {"const": "fixed"}, {"$ref": "#/properties/missing"}],
    ids=["untyped", "const", "unresolved-ref"],
)
def test_an_optional_property_without_a_mapped_type_can_still_be_set(prop):
    # Advertised as {"type": "null"}, the model could only ever send null for it.
    parameters = function_parameters(mcp_tool("search", {"query": {"type": "string"}, "filter": prop}, ["query"]))

    filter_type = parameters["properties"]["filter"]["type"]
    assert filter_type != "null"
    assert "string" in (filter_type if isinstance(filter_type, list) else [filter_type])


def test_an_optional_binary_parameter_keeps_the_schema_it_had():
    # Only untyped parameters are offered as strings: a mask takes image bytes, not text the model writes.
    properties = {}

    schema_generator.generate_property_schema(properties, "mask", ImageEditInputSchema.model_fields["mask"])

    assert properties["mask"] == {"description": "Optional mask image indicating areas to edit.", "type": "null"}


def tree_tool() -> MCPTool:
    """A tool taking a tree: each node has a value and children that are nodes."""
    node = {
        "type": "object",
        "properties": {"value": {"type": "string"}, "children": {"type": "array", "items": {"$ref": "#/$defs/Node"}}},
    }
    return mcp_tool("query", {"root": {"$ref": "#/$defs/Node"}}, ["root"], **{"$defs": {"Node": node}})


def test_a_recursive_tool_schema_does_not_stop_a_function_calling_agent_from_building():
    llm = OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o")

    agent = Agent(name="a", llm=llm, tools=[tree_tool()], inference_mode=InferenceMode.FUNCTION_CALLING)

    query = next(t for t in agent._tools if t["function"]["name"] == "query")
    root = query["function"]["parameters"]["properties"]["root"]
    assert root["type"] == "object"
    assert root["properties"]["children"]["type"] == "array"


@pytest.mark.parametrize("to_strict", [to_openai_strict_function, to_strict_subset_function])
def test_a_tree_two_levels_deep_reaches_the_server_when_tool_calls_are_strict(to_strict):
    llm = OpenAI(connection=OpenAIConnection(api_key="test-api-key"), model="gpt-4o", strict_tools=True)
    agent = Agent(name="a", llm=llm, tools=[tree_tool()], inference_mode=InferenceMode.FUNCTION_CALLING)
    query = next(t for t in agent._tools if t["function"]["name"] == "query")["function"]
    children = to_strict(query)["parameters"]["properties"]["root"]["properties"]["children"]
    # Strict mode cannot describe the open object below the recursion, so the model writes each child as JSON.
    assert children["items"]["type"] == "string"
    calls = []

    @contextlib.asynccontextmanager
    async def connect(connection, headers=None):
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

        async def call_tool(self, name, arguments):
            calls.append((name, arguments))
            return CallToolResult(content=[TextContent(type="text", text="matched")])

    child = json.dumps({"value": "b", "children": []})
    steps = [tool_step("query", root={"value": "a", "children": [child]}), final_step("done")]
    with patch.object(MCPSse, "connect", new=connect), patch("mcp.ClientSession", new=Session):
        result, _ = run_scripted(agent, InferenceMode.FUNCTION_CALLING, steps)

    assert result.status == RunnableStatus.SUCCESS
    assert calls == [("query", {"root": {"value": "a", "children": [{"value": "b", "children": []}]}})]
