"""SubAgentTool — wraps an agent instance or factory as a callable tool for parent agents."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger

if TYPE_CHECKING:
    from dynamiq.connections.managers import ConnectionManager
    from dynamiq.nodes.agents.base import Agent


class SubAgentToolInputSchema(BaseModel):
    """Input schema for SubAgentTool, mirrors the core agent input format."""

    brief: str = Field(
        default="Delegating task to a sub-agent.",
        description="Very brief description of the action being performed. Example: 'Research latest AI papers'.",
    )
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

    .. note::
        **Dict factory and shared tool references** — In dict factory mode the
        kwargs (including ``llm`` and ``tools``) are passed by reference to each
        new ``Agent``.  This is safe for stateless objects (LLMs, most built-in
        tools) because they hold no per-agent mutable state.  If your tools are
        **stateful** (e.g. they accumulate results or maintain a session), use
        the callable factory instead so you can create fresh tool instances on
        every invocation.

    Examples::

        # Mode 1 — reuse an existing agent
        tool = SubAgentTool(agent=researcher_agent)

        # Mode 2a — callable factory (use for stateful tools)
        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory=lambda: Agent(name="Researcher", llm=llm, tools=[exa]),
        )

        # Mode 2b — dict factory (JSON-serializable, tools are shared)
        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory={"name": "Researcher", "llm": llm, "tools": [exa]},
        )
    """

    group: NodeGroup = NodeGroup.AGENTS
    name: str = Field(
        ...,
        description=(
            "Name of the sub-agent tool. Exposed to the LLM as the tool/function name during schema generation."
        ),
    )
    description: str = Field(
        ...,
        description=(
            "Short description of what this sub-agent does. Exposed to the LLM as the tool/function description."
        ),
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

    validate_factory: bool = Field(
        default=True,
        description="If True, init_components creates a trial agent to validate the factory.",
    )

    is_postponed_component_init: bool = True
    _connection_manager: Any = PrivateAttr(default=None)

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
        """Create an Agent from the factory (callable or dict) and initialize its components.

        Dict factories pass kwargs by reference, so LLM and tool objects are
        shared across agents.  This is intentional — stateless components
        benefit from reuse.  For stateful tools, use a callable factory.
        """
        if isinstance(self.agent_factory, dict):
            from dynamiq.nodes.agents.agent import Agent as ReActAgent

            agent = ReActAgent(**self.agent_factory)
        else:
            agent = self.agent_factory()

        from dynamiq.nodes.agents.base import Agent as BaseAgent

        if not isinstance(agent, BaseAgent):
            raise TypeError(
                f"SubAgentTool '{self.name}': agent_factory must return an Agent, " f"got {type(agent).__name__}"
            )

        agent.id = generate_uuid()

        if self._connection_manager is not None:
            agent.init_components(self._connection_manager)

        return agent

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        super().init_components(connection_manager)
        self._connection_manager = connection_manager
        if self.agent_factory is not None and self.validate_factory:
            trial = self._create_agent_from_factory()
            logger.info(
                f"SubAgentTool '{self.name}': successfully created a trial agent from factory with id {trial.name}"
            )
            del trial
        elif self.agent is not None:
            if self.agent.is_postponed_component_init:
                self.agent.init_components(connection_manager)

    def get_or_create_agent(self) -> Agent:
        """Return the initialized agent or create a new one from the factory."""
        if self.agent is not None:
            return self.agent

        agent = self._create_agent_from_factory()
        logger.info(f"SubAgentTool '{self.name}': created new agent from factory")
        return agent

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"agent": True, "agent_factory": True}

    @staticmethod
    def _serialize_value(value: Any, **kwargs) -> Any:
        """Recursively convert live objects to plain dicts for serialization."""
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return value.to_dict(**kwargs)
        if isinstance(value, dict):
            return {k: SubAgentTool._serialize_value(v, **kwargs) for k, v in value.items()}
        if isinstance(value, list):
            return [SubAgentTool._serialize_value(item, **kwargs) for item in value]
        return value

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        if self.agent is not None:
            data["agent"] = self.agent.to_dict(**kwargs)
        elif isinstance(self.agent_factory, dict):
            data["agent_factory"] = self._serialize_value(self.agent_factory, **kwargs)
        return data

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
