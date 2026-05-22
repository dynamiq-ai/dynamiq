"""Agent-specific checkpoint state and the iterative checkpoint mixin.

Separated from ``base.py`` so the Agent's checkpoint/resume machinery — the
loop-level (iteration) state, LLM/tool sub-state, and the pending tool call
captured for HITL-timeout replay — lives in one focused module.
"""

from typing import Any

from pydantic import BaseModel, Field

from dynamiq.checkpoints.checkpoint import (
    BaseCheckpointState,
    CheckpointNodeMixin,
    IterationState,
    IterativeCheckpointMixin,
)
from dynamiq.prompts import Prompt

# Default prompt prefix length: [system_message, user_message].
# At runtime, _history_offset is recalculated to len(prompt.messages) before the ReAct loop,
# which may be larger when memory history messages are injected.
DEFAULT_HISTORY_OFFSET = 1


class AgentIterationData(BaseModel):
    """Typed iteration data for Agent loop-level checkpoints."""

    prompt_messages: list[dict] | None = None
    agent_state: dict | None = None
    history_offset: int = DEFAULT_HISTORY_OFFSET
    # Tool call selected by the LLM but not yet completed (e.g. interrupted by an
    # input-streaming/HITL timeout). Persisted so it can be replayed exactly on
    # resume instead of asking the LLM to regenerate the same action.
    pending_action: str | None = None
    pending_action_input: Any = None
    pending_thought: str | None = None


class AgentCheckpointState(BaseCheckpointState):
    """Checkpoint state for Agent nodes.

    Loop-level resume data is stored in the inherited ``iteration`` field
    (from BaseCheckpointState) via the IterativeCheckpointMixin protocol.
    """

    history_offset: int = Field(default=DEFAULT_HISTORY_OFFSET, description="Offset for agent conversation history")
    llm_state: dict = Field(default_factory=dict, description="LLM component checkpoint state")
    tool_states: dict[str, dict] = Field(
        default_factory=dict, description="Tool component checkpoint states keyed by tool ID"
    )


class AgentIterativeCheckpointMixin(IterativeCheckpointMixin):
    """Checkpoint/resume behaviour for the Agent's ReAct loop.

    Extends ``IterativeCheckpointMixin`` (per-iteration resume) with the
    Agent-specific pieces: LLM and tool sub-state, serialized prompt messages,
    and the pending tool call replayed after an input-streaming timeout.

    Designed to be mixed into ``Agent``; the methods read host attributes
    (``llm``, ``tools``, ``state``, ``_prompt``, ``_history_offset``) provided
    by the concrete class.
    """

    # Loop-level progress and the in-flight tool call captured before tool
    # execution, so an interruption (e.g. HITL input timeout) can persist them.
    _completed_loops: int = 0
    _pending_action: str | None = None
    _pending_action_input: Any = None
    _pending_thought: str | None = None

    def to_checkpoint_state(self) -> AgentCheckpointState:
        """Extract agent state for checkpointing, including LLM, tool, and loop-level states."""
        llm_checkpoint = self.llm.to_checkpoint_state()

        tool_states = {}
        for tool in self.tools:
            if isinstance(tool, CheckpointNodeMixin):
                tool_checkpoint = tool.to_checkpoint_state()
                tool_state = tool_checkpoint.model_dump() if hasattr(tool_checkpoint, "model_dump") else tool_checkpoint
                if tool_state:
                    tool_states[tool.id] = tool_state

        base_fields = super().to_checkpoint_state().model_dump(exclude_none=True)
        state = AgentCheckpointState(
            history_offset=self._history_offset,
            llm_state=llm_checkpoint.model_dump() if hasattr(llm_checkpoint, "model_dump") else llm_checkpoint,
            tool_states=tool_states,
            **base_fields,
        )
        self._save_iteration_to_checkpoint(state)
        return state

    def set_pending_tool_call(self, action: str | None, action_input: Any, thought: str | None) -> None:
        """Record the tool call about to run so it can be checkpointed on interruption."""
        self._pending_action = action
        self._pending_action_input = action_input
        self._pending_thought = thought

    def clear_pending_tool_call(self) -> None:
        """Drop the recorded tool call once execution has completed."""
        self._pending_action = None
        self._pending_action_input = None
        self._pending_thought = None

    def get_iteration_state(self) -> IterationState:
        """Serialize ReAct loop progress for checkpoint persistence."""
        data = AgentIterationData(
            prompt_messages=self._serialize_prompt_messages(),
            agent_state=(
                self.state.model_dump() if hasattr(self, "state") and hasattr(self.state, "model_dump") else None
            ),
            history_offset=self._history_offset,
            pending_action=self._pending_action,
            pending_action_input=self._pending_action_input,
            pending_thought=self._pending_thought,
        )
        return IterationState(completed_iterations=self._completed_loops, iteration_data=data.model_dump())

    def restore_iteration_state(self, state: IterationState) -> None:
        """Restore prompt messages and AgentState from a checkpoint IterationState."""
        data = AgentIterationData(**state.iteration_data)
        if data.prompt_messages:
            self._prompt.messages = Prompt.deserialize_messages(data.prompt_messages)
        if data.agent_state:
            from dynamiq.nodes.agents.agent import AgentState

            self.state = AgentState(**data.agent_state)
        self._history_offset = data.history_offset
        self._pending_action = data.pending_action
        self._pending_action_input = data.pending_action_input
        self._pending_thought = data.pending_thought
        # Mirror the completed-loop count back onto the instance so a snapshot
        # taken before any new loop finishes (e.g. an input timeout during the
        # replayed tool call) doesn't overwrite the saved progress with 0.
        self._completed_loops = state.completed_iterations

    def _serialize_prompt_messages(self) -> list[dict] | None:
        """Serialize current prompt messages for checkpoint persistence."""
        if not hasattr(self, "_prompt") or not self._prompt or not self._prompt.messages:
            return None
        return self._prompt.serialize_messages() or None

    def from_checkpoint_state(self, state: AgentCheckpointState | dict[str, Any]) -> None:
        """Restore agent state from checkpoint, including LLM, tool, and loop-level states."""
        super().from_checkpoint_state(state)
        state_dict = state if isinstance(state, dict) else state.model_dump()

        self._history_offset = state_dict.get("history_offset", DEFAULT_HISTORY_OFFSET)

        if (llm_state := state_dict.get("llm_state")) is not None:
            self.llm.from_checkpoint_state(llm_state)
        if (tool_states := state_dict.get("tool_states")) is not None:
            self._restore_tool_states(tool_states)

        self._restore_iteration_from_checkpoint(state_dict)

    def _restore_tool_states(self, tool_states: dict[str, dict]) -> None:
        """Restore checkpoint states for agent's tools."""
        tools_by_id = {tool.id: tool for tool in self.tools}
        for tool_id, tool_state in tool_states.items():
            tool = tools_by_id.get(tool_id)
            if tool and isinstance(tool, CheckpointNodeMixin):
                tool.from_checkpoint_state(tool_state)
