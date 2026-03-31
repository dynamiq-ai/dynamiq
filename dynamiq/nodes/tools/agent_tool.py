"""SubAgentTool — wraps an agent instance or factory as a callable tool for parent agents."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any, Callable, ClassVar

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.nodes import ErrorHandling, Node, NodeGroup
from dynamiq.runnables import RunnableConfig
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
    files: list[str] | None = Field(
        default=None,
        description="Full file paths to upload to the sub-agent.",
    )

    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)


class SubAgentTool(Node):
    """Wraps an agent instance or factory as a callable tool for parent agents.

    Three modes:
      - **Initialized** (``agent``): reuses a single agent instance across calls.
      - **Factory callable** (``agent_factory`` as callable): creates a fresh agent
        per invocation, enabling isolated state and parallel execution.
        The callable must **not** capture shared ``BaseModel`` objects (e.g. a
        shared LLM or tool instance).  Captured objects are used as-is and are
        not deep-copied, so mutations applied to the returned agent propagate
        back to those shared originals.
      - **Factory dict** (``agent_factory`` as dict): a blueprint dict using the
        same format as workflow YAML node definitions.  Resolved via
        ``WorkflowYAMLLoader`` on each invocation, producing completely fresh
        and isolated Agent instances.

    Examples::

        # Mode 1 — reuse an existing agent
        tool = SubAgentTool(agent=researcher_agent)

        # Mode 2a — callable factory
        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory=lambda: Agent(name="Researcher", llm=llm, tools=[exa]),
        )

        # Mode 2b — dict blueprint (same format as workflow YAML files)
        tool = SubAgentTool(
            name="Researcher",
            description="Performs web research",
            agent_factory={
                "connections": {
                    "openai-conn": {"type": "dynamiq.connections.OpenAI"},
                },
                "name": "Researcher",
                "llm": {
                    "type": "dynamiq.nodes.llms.OpenAI",
                    "connection": "openai-conn",
                    "model": "gpt-4o",
                },
                "role": "You are a research agent.",
                "tools": [],
            },
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
            "Either a callable returning an Agent, or a dict blueprint (same format as workflow YAML). "
            "Blueprint only — not instantiated at init time. "
            "A fresh Agent is created from this on each call to _create_agent_from_factory(). "
            "For callable factories, every nested BaseModel (llm, tools, etc.) must be constructed "
            "inside the callable — capturing shared instances causes unintended mutation of those "
            "shared objects."
        ),
    )

    validate_factory: bool = Field(
        default=True,
        description="If True, init_components creates a trial agent to validate the factory.",
    )
    max_calls: int | None = Field(
        default=None,
        description="Maximum number of invocations allowed per agent run. None means unlimited.",
    )

    is_postponed_component_init: bool = True
    _connection_manager: Any = PrivateAttr(default=None)
    _call_count: int = PrivateAttr(default=0)

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

    def _build_agent_from_factory(self) -> Agent:
        """Construct an Agent from the factory WITHOUT initializing components.

        For dict factories the blueprint is deep-copied and resolved via
        ``WorkflowYAMLLoader`` methods — the same resolution path used for
        real workflow YAML files.  Each call produces completely fresh objects
        with no shared state.

        No side effects: ``init_components`` is NOT called, so no network
        connections or sandbox resources are created.  Use this for
        serialization or inspection; use ``_create_agent_from_factory`` when
        a fully initialized agent is needed.
        """
        from dynamiq.nodes.agents.base import Agent as BaseAgent

        if isinstance(self.agent_factory, dict):
            from dynamiq.nodes.agents.agent import Agent as ReActAgent
            from dynamiq.serializers.loaders.yaml import WorkflowYAMLLoader

            data = copy.deepcopy(self.agent_factory)
            registry: dict = {}
            connections = WorkflowYAMLLoader.get_connections(data, registry)
            agent_data = {k: v for k, v in data.items() if k != "connections"}
            resolved = WorkflowYAMLLoader.get_updated_node_init_data_with_initialized_nodes(
                node_init_data=agent_data,
                nodes={},
                flows={},
                connections=connections,
                prompts={},
                registry=registry,
                connection_manager=self._connection_manager,
            )
            resolved.setdefault("is_postponed_component_init", True)
            resolved.pop("id", None)
            agent = ReActAgent(**resolved)
        elif callable(self.agent_factory):
            agent = self.agent_factory()
        else:
            raise TypeError(
                f"SubAgentTool '{self.name}': agent_factory must be a dict or callable, "
                f"got {type(self.agent_factory).__name__}"
            )

        if not isinstance(agent, BaseAgent):
            raise TypeError(
                f"SubAgentTool '{self.name}': agent_factory must return an Agent, " f"got {type(agent).__name__}"
            )

        return agent

    def _create_agent_from_factory(self) -> Agent:
        """Create an Agent from the factory and initialize its components."""
        agent = self._build_agent_from_factory()

        if self._connection_manager is not None:
            agent.init_components(self._connection_manager)
        else:
            agent.init_components()

        return agent

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        super().init_components(connection_manager)
        self._connection_manager = connection_manager
        if self.agent_factory is not None and self.validate_factory:
            trial = self._create_agent_from_factory()
            logger.info(f"SubAgentTool '{self.name}': successfully created a trial agent with id {trial.id}")
            self.cleanup_factory_agent(trial)
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

    def reset_call_count(self) -> None:
        """Reset the per-run invocation counter."""
        self._call_count = 0

    @staticmethod
    def cleanup_factory_agent(agent: Agent) -> None:
        """Kill sandbox resources on a factory-created agent."""
        if getattr(agent, "sandbox_backend", None):
            try:
                agent.sandbox_backend.close(kill=True)
                logger.info(f"SubAgentTool '{agent.id}': successfully cleaned up factory agent sandbox")
            except Exception as e:
                logger.warning("Factory agent sandbox cleanup failed: %s", e)

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {"agent": True, "agent_factory": True}

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        if self.agent is not None:
            data["agent"] = self.agent.to_dict(**kwargs)
        elif isinstance(self.agent_factory, dict):
            agent = self._build_agent_from_factory()
            data["agent_factory"] = agent.to_dict(**kwargs)
        elif callable(self.agent_factory):
            data["agent_factory"] = {
                "_type": "callable",
                "_repr": getattr(self.agent_factory, "__name__", repr(self.agent_factory)),
            }
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
