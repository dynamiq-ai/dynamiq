from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import generate_uuid

if TYPE_CHECKING:
    from dynamiq.nodes.agents.base import Agent


class AgentTool(Node):
    """
    Wraps an Agent so it can be used as a tool within another agent.

    - Exposes the wrapped agent's input_schema so ReActAgent can generate tool schemas
    - Forwards execution to the wrapped agent, preserving tracing/streaming
    - Allows configuring memory/streaming propagation guards and recursion depth
    """

    group: NodeGroup = NodeGroup.TOOLS
    name: str = "Agent Tool"
    description: str | None = None

    # Allow file propagation by default to sub-agent
    is_files_allowed: bool = True

    # Wrapped agent instance (annotated as Node to avoid forward-ref build issues)
    agent: Node

    # Behavior controls
    share_memory: bool = False
    propagate_streaming: bool = True
    max_depth: int = 3

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Default name/description from wrapped agent if not provided
        if not self.name:
            self.name = self.agent.name
        if not self.description:
            self.description = (self.agent.role or self.agent.description or "").strip()

    # Input schema is class-level for schema generation; set on dynamic subclass in from_agent()
    input_schema: ClassVar[type[BaseModel] | None] = None

    @classmethod
    def from_agent(
        cls,
        agent: "Agent",
        name: str | None = None,
        description: str | None = None,
        share_memory: bool = False,
        propagate_streaming: bool = True,
        max_depth: int = 3,
    ) -> "AgentTool":
        # Create a dynamic subclass so input_schema can be a true ClassVar for schema generation
        class_name = f"AgentToolFor_{agent.__class__.__name__}"
        dynamic_cls = type(
            class_name,
            (AgentTool,),
            {
                "__module__": AgentTool.__module__,
                "__qualname__": class_name,
                "input_schema": getattr(agent.__class__, "input_schema", None),
            },
        )

        tool = dynamic_cls(
            agent=agent,
            name=name or agent.name,
            description=description or (agent.role or agent.description or ""),
        )
        tool.share_memory = share_memory
        tool.propagate_streaming = propagate_streaming
        tool.max_depth = max_depth
        return tool

    def execute(self, input_data: dict[str, Any], config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the wrapped agent with provided input_data.

        Returns dict with key 'content' to align with tool contract expected by Agent._run_tool.
        """
        from dynamiq.nodes.agents.base import Agent  # Local import to avoid circular

        if not isinstance(self.agent, Agent):
            raise ValueError("AgentTool requires an instance of Agent")

        current_depth = int(kwargs.get("depth", 0))
        if current_depth >= self.max_depth:
            raise ValueError(f"AgentTool max_depth {self.max_depth} exceeded.")

        config = ensure_config(config)

        # Clone per execution to avoid shared state mutations in parallel runs
        agent_run = self.agent.clone()
        # Regenerate id to isolate tracing/streaming entity per parallel call
        try:
            agent_run.id = generate_uuid()
        except Exception as e:
            # Non-fatal: keep existing id if regeneration fails
            from dynamiq.utils.logger import logger

            logger.debug(f"AgentTool: could not regenerate clone id: {e}")

        # If a per-node override exists for the original agent id, mirror it for the clone
        try:
            if config and getattr(config, "nodes_override", None):
                if (node_cfg := config.nodes_override.get(self.agent.id)) is not None:
                    config.nodes_override[agent_run.id] = node_cfg
        except Exception as e:
            from dynamiq.utils.logger import logger

            logger.debug(f"AgentTool: nodes_override propagation skipped: {e}")

        # Optional propagation of streaming configuration
        if self.propagate_streaming:
            agent_run.streaming = self.streaming

        # Optional sharing of memory reference
        if self.share_memory and agent_run.memory is None:
            parent_memory = kwargs.get("parent_memory", None)
            if parent_memory is not None:
                agent_run.memory = parent_memory

        # Sanitize None-valued fields coming from LLM action inputs
        if isinstance(input_data, dict):
            sanitized_input = {k: v for k, v in input_data.items() if v is not None}
        else:
            sanitized_input = input_data

        # Ensure user/session identity is propagated to sub-agent input to enable memory
        # Only set user/session if the wrapped agent actually has a Memory configured
        if self.share_memory and agent_run.memory is not None:
            try:
                parent_vars = getattr(self, "_prompt_variables", {}) or {}
                user_id = parent_vars.get("user_id") or sanitized_input.get("user_id")
                session_id = parent_vars.get("session_id") or sanitized_input.get("session_id")
                if isinstance(sanitized_input, dict):
                    if user_id is not None and sanitized_input.get("user_id") is None:
                        sanitized_input["user_id"] = user_id
                    if session_id is not None and sanitized_input.get("session_id") is None:
                        sanitized_input["session_id"] = session_id
            except Exception as e:
                from dynamiq.utils.logger import logger

                logger.debug(f"AgentTool: user/session propagation skipped: {e}")

        # If tool_params are provided by the parent agent, pass them via kwargs only
        # Do NOT merge into input_data to avoid leaking secrets into traces/logs
        if "tool_params" in kwargs:
            # Keep as-is; Agent._run_tool expects kwargs propagation
            pass

        # Validate against wrapped agent's input schema using its own context
        validated = agent_run.input_schema.model_validate(
            sanitized_input,
            context=agent_run.get_context_for_input_schema(),
        )

        # Delegate to wrapped agent's execute; returns dict with content + intermediate_steps
        rr_content = agent_run.execute(
            input_data=validated,
            config=config,
            **(kwargs | {"parent_run_id": kwargs.get("run_id"), "depth": current_depth + 1}),
        )

        # Return a tool-compatible dict.
        # Tools are expected to return {"content": ...}; preserve sub-agent intermediates for observability if present.
        if isinstance(rr_content, dict) and "content" in rr_content:
            return {"content": rr_content["content"], "intermediate_steps": rr_content.get("intermediate_steps")}
        return {"content": rr_content}


# Ensure forward references (if any) are resolved for Pydantic models (safe no-op)
try:
    AgentTool.model_rebuild()
except Exception as e:
    from dynamiq.utils.logger import logger

    logger.debug(f"AgentTool: model_rebuild skipped: {e}")
