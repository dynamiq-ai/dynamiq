import io
from typing import Any, Callable, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from dynamiq.nodes.agents.base import AgentInputSchema


class SubAgentTool(Node):
    """A tool that spawns a fresh agent from a factory on each invocation.

    The factory callable defines everything about the agent (name, role, LLM,
    tools, etc.). SubAgentTool only manages how the sub-agent is exposed to
    the parent agent — accepting text input and optional files.

    Shared resources (LLM, sandbox, connections) can be captured via closures
    in the factory callable, allowing multiple SubAgentTools to reuse the same
    backends while getting fresh agent state on every invocation.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "SubAgent"
    description: str = Field(default="A sub-agent that can be delegated tasks.")
    error_handling: ErrorHandling = Field(
        default_factory=lambda: ErrorHandling(timeout_seconds=3600)
    )
    factory: Callable[..., Any] = Field(
        ...,
        exclude=True,
        description="Callable that returns a fresh Agent instance on each invocation.",
    )

    input_schema: ClassVar[type[AgentInputSchema]] = AgentInputSchema

    def execute(
        self,
        input_data: AgentInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        pass
