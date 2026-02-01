"""Meta-tool that enables parallel tool execution capability for agents."""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import Node, NodeGroup

PARALLEL_TOOL_NAME = "run_parallel"  # if change here, also change in ParallelToolCallsTool


class ToolCallItem(BaseModel):
    """Schema for a single tool call within parallel execution."""

    name: str = Field(
        ...,
        description="Name of the tool to execute (must match an available tool name)",
    )
    input: dict[str, Any] = Field(
        default_factory=dict,
        description="Input parameters for the tool as key-value pairs",
    )

    model_config = ConfigDict(extra="forbid")


class ParallelToolCallsInputSchema(BaseModel):
    """Input schema for ParallelToolCallsTool."""

    tools: list[ToolCallItem] = Field(
        ...,
        description="List of tools to execute in parallel",
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
    name: Literal["run_parallel"] = PARALLEL_TOOL_NAME  # Frozen - cannot be overridden
    description: str = "Tool that enables running multiple other tools simultaneously in parallel execution."

    input_schema: ClassVar[type[ParallelToolCallsInputSchema]] = ParallelToolCallsInputSchema

    def execute(self, input_data: ParallelToolCallsInputSchema, config=None, **kwargs):
        """Agent intercepts this tool - execute is never called directly."""
        return None
