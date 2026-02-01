"""Meta-tool that enables parallel tool execution capability for agents."""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import ErrorHandling, Node, NodeGroup


class ParallelToolCallsInputSchema(BaseModel):
    """Input schema for ParallelToolCallsTool.

    Example:
        {
            "tools": [
                {"name": "<ToolA>", "input": {"query": "AI news"}},
                {"name": "<ToolB>", "input": {"city": "NYC"}}
            ]
        }
    """

    tools: list[dict[str, Any]] = Field(
        ...,
        description=(
            "List of tools to execute in parallel. Each item must have 'name' (from available tool names) "
            "and 'input' (dict of tool parameters). Example: "
            '[{"name": "<ToolA>", "input": {"query": "AI"}}, '
            '{"name": "<ToolB>", "input": {"expr": "2+2"}}]'
        ),
        min_length=1,
    )

    model_config = ConfigDict(extra="forbid")


class ParallelToolCallsTool(Node):
    """
    A meta-tool that signals the agent can execute multiple tools in parallel.

    This tool does not execute tools itself - it provides a schema that allows
    the agent to specify multiple tools to run in parallel. The agent's internal
    parallel execution logic handles the actual execution.

    When the agent uses this tool, it passes a list of tool calls which are
    then executed in parallel by the agent's execution engine.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "RunParallelTool"
    description: str = (
        "Run multiple tools at once. "
        "Input is a JSON object with a 'tools' array. "
        "Each array item has 'name' (string) and 'input' (object). "
        'Example input: {"tools": [{"name": "<ToolA>", "input": {"x": 1}}, {"name": "<ToolB>", "input": {"y": 2}}]}'
    )

    error_handling: ErrorHandling = Field(
        default_factory=lambda: ErrorHandling(timeout_seconds=600)
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[ParallelToolCallsInputSchema]] = ParallelToolCallsInputSchema

    def execute(self, input_data: ParallelToolCallsInputSchema, config=None, **kwargs):
        """Agent intercepts this tool - execute is never called directly."""
        return None
