"""SubAgentTool — wraps an agent instance or factory as a callable tool for parent agents."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.nodes.agents.base import Agent


class SubAgentToolInputSchema(BaseModel):
    """Input schema for SubAgentTool, mirrors the core agent input format."""

    input: str = Field(default="", description="Task or query to pass to the sub-agent.")
    images: list[str | bytes | io.BytesIO] | None = Field(
        default=None,
        description="Image inputs (URLs, bytes, or file objects).",
    )
    files: list[io.BytesIO | bytes] | None = Field(
        default=None,
        description="Files to pass to the sub-agent.",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class SubAgentTool(Node):
    """Wraps an agent instance or factory as a callable tool for parent agents.

    Three modes:
      - **Initialized** (``agent``): reuses a single agent instance across calls.
      - **Factory callable** (``agent_factory`` as callable): creates a fresh agent
        per invocation, enabling isolated state and parallel execution.
      - **Factory dict** (``agent_factory`` as dict): JSON-serializable Agent kwargs;
        a fresh ``Agent(**agent_factory)`` is created per invocation.

    Examples::

        # Mode 1 — reuse an existing agent
        tool = SubAgentTool(agent=researcher_agent)

        # Mode 2a — callable factory
        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory=lambda: Agent(name="Researcher", llm=llm, tools=[exa]),
        )

        # Mode 2b — dict factory (JSON-serializable)
        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory={"name": "Researcher", "llm": llm, "tools": [exa]},
        )
    """

    group: NodeGroup = NodeGroup.AGENTS
    name: str = Field(
        ...,
        description="Name of the sub-agent tool. Exposed to the LLM as the tool/function name during schema generation.",
    )
    description: str = Field(
        ...,
        description="Short description of what this sub-agent does. Exposed to the LLM as the tool/function description.",
    )
    error_handling: ErrorHandling = Field(
        default_factory=lambda: ErrorHandling(timeout_seconds=3600),
    )

    agent: Any = Field(
        default=None,
        description="Initialized agent instance to reuse across invocations.",
    )
    agent_factory: Callable[..., Any] | dict | None = Field(
        default=None,
        description=(
            "Factory for creating a new Agent per invocation. "
            "Either a callable returning an Agent, or a dict of Agent kwargs (JSON-serializable)."
        ),
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[SubAgentToolInputSchema]] = SubAgentToolInputSchema

    FACTORY_HINT: ClassVar[str] = " [Independent agent: each call spawns a fresh instance — safe to call in parallel.]"
    INITIALIZED_HINT: ClassVar[str] = (
        " [Shared agent: all calls reuse the same instance — calls are executed sequentially.]"
    )

    @model_validator(mode="after")
    def validate_agent_config(self):
        if self.agent is None and self.agent_factory is None:
            raise ValueError("SubAgentTool requires either 'agent' or 'agent_factory'.")
        if self.agent is not None and self.agent_factory is not None:
            raise ValueError("SubAgentTool cannot have both 'agent' and 'agent_factory'.")

        self.is_parallel_execution_allowed = self.is_factory_mode

        hint = self.FACTORY_HINT if self.is_factory_mode else self.INITIALIZED_HINT
        if hint not in self.description:
            self.description = self.description.rstrip() + hint

        return self

    @property
    def is_factory_mode(self) -> bool:
        """True when the tool creates a fresh agent per call."""
        return self.agent_factory is not None

    def _create_agent_from_factory(self) -> Agent:
        """Create an Agent from the factory (callable or dict)."""
        if isinstance(self.agent_factory, dict):
            from dynamiq.nodes.agents.agent import Agent as ReActAgent

            return ReActAgent(**self.agent_factory)
        return self.agent_factory()

    def init_components(self, connection_manager=None):
        super().init_components(connection_manager)
        if self.agent_factory is not None:
            from dynamiq.nodes.agents.base import Agent as BaseAgent

            trial = self._create_agent_from_factory()
            if not isinstance(trial, BaseAgent):
                raise TypeError(
                    f"SubAgentTool '{self.name}': agent_factory must return an Agent, " f"got {type(trial).__name__}"
                )

    def get_or_create_agent(self) -> Agent:
        """Return the initialized agent or create a new one from the factory."""
        if self.agent is not None:
            return self.agent

        agent = self._create_agent_from_factory()
        logger.info(f"SubAgentTool '{self.name}': created new agent from factory")
        return agent

    def to_dict(self, **kwargs) -> dict:
        if self.agent is not None:
            return self.agent.to_dict(**kwargs)
        return super().to_dict(**kwargs)

    def execute(
        self,
        input_data: SubAgentToolInputSchema,
        config: RunnableConfig | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Delegate execution to the underlying agent.

        Normally the parent agent's ``_run_tool`` will resolve the agent
        directly, but this provides a working fallback for standalone use.
        """
        raise NotImplementedError(
            "SubAgentTool does not implement execute directly. Use get_or_create_agent() to get the agent."
        )
