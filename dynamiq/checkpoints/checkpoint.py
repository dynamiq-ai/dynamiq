from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

import orjson
from pydantic import BaseModel, ConfigDict, Field

from dynamiq.checkpoints.utils import decode_checkpoint_data, encode_checkpoint_data
from dynamiq.utils import encode_reversible, generate_uuid


def utc_now() -> datetime:
    """Get current UTC time with timezone info."""
    return datetime.now(timezone.utc)


class CheckpointStatus(str, Enum):
    """Checkpoint execution status."""

    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING_INPUT = "pending_input"


class BaseCheckpointState(BaseModel):
    """Base checkpoint state model. All node checkpoint states inherit from this."""

    model_config = ConfigDict(extra="allow")

    iteration: dict | None = Field(default=None, description="IterativeCheckpointMixin state for loop-level resume")
    approval_response: dict | None = Field(default=None, description="Stored HITL approval response for resume")


class CheckpointMixin:
    """
    Mixin providing default checkpoint implementation for nodes.

    Nodes inherit this mixin to gain checkpoint support with sensible defaults.
    Override to_checkpoint_state() and from_checkpoint_state() for custom state handling.
    """

    _is_resumed: bool = False

    def to_checkpoint_state(self) -> BaseCheckpointState:
        """Return node-specific state for checkpointing. Override in subclasses."""
        return BaseCheckpointState()

    def from_checkpoint_state(self, state: BaseCheckpointState | dict[str, Any]) -> None:
        """Restore node state from checkpoint. Override in subclasses."""
        self._is_resumed = True

    @property
    def is_resumed(self) -> bool:
        """Check if this node was restored from a checkpoint."""
        return self._is_resumed

    def reset_resumed_flag(self) -> None:
        """Reset the resumed flag after handling resume logic."""
        self._is_resumed = False


class IterationState(BaseModel):
    """Snapshot of iterative progress within a long-running node.

    Used by nodes that perform iterative work (ReAct loops, orchestrator
    state transitions, etc.) to persist per-iteration progress so the node
    can resume from the last completed iteration instead of restarting.
    """

    model_config = ConfigDict(extra="allow")

    completed_iterations: int = Field(default=0, description="Number of fully completed iterations")
    iteration_data: dict = Field(default_factory=dict, description="Node-specific iteration state")


class IterativeCheckpointMixin:
    """Mixin for nodes that perform iterative work and support per-iteration resume.

    Provides a standardised save/restore contract so any long-running node
    (Agent, GraphOrchestrator, AdaptiveOrchestrator, …) can skip already-completed
    iterations on resume without duplicating the staging-field pattern.

    Subclasses must implement:
        get_iteration_state()     – serialize current iteration progress
        restore_iteration_state() – restore from a previously saved IterationState
    """

    _iteration_state: IterationState | None = None
    _has_restored_iteration: bool = False

    def get_iteration_state(self) -> IterationState:
        """Serialize current iteration progress for checkpointing."""
        raise NotImplementedError

    def restore_iteration_state(self, state: IterationState) -> None:
        """Restore node to the state after the last completed iteration."""
        raise NotImplementedError

    def get_start_iteration(self) -> int:
        """Return the number of completed iterations (0 = fresh start).

        When a restored iteration exists, calls ``restore_iteration_state()``
        to apply the saved state (prompt messages, agent state, etc.) before
        returning the loop offset.  After returning > 0 the restored data is
        cleared so subsequent calls return 0.
        """
        if not self._has_restored_iteration or not self._iteration_state:
            return 0
        completed = self._iteration_state.completed_iterations
        self.restore_iteration_state(self._iteration_state)
        self._has_restored_iteration = False
        self._iteration_state = None
        return completed

    def _save_iteration_to_checkpoint(self, checkpoint_state: "BaseCheckpointState") -> None:
        """Attach current iteration data to an outgoing checkpoint state."""
        iteration = self.get_iteration_state()
        checkpoint_state.iteration = iteration.model_dump()

    def _restore_iteration_from_checkpoint(self, state_dict: dict) -> None:
        """Extract iteration data from an incoming checkpoint state dict."""
        if iteration_data := state_dict.get("iteration"):
            self._iteration_state = (
                IterationState(**iteration_data) if isinstance(iteration_data, dict) else iteration_data
            )
            self._has_restored_iteration = True


class CheckpointContext:
    """Context for checkpoint operations passed to nodes during execution.

    Provides callbacks for:
    - HITL: notify the Flow when nodes are waiting for human input
    - Mid-loop: request a checkpoint save during long agent loops
    """

    def __init__(
        self,
        on_pending_input: Callable[[str, str, dict | None], None] | None = None,
        on_input_received: Callable[[str], None] | None = None,
        on_save_mid_run: Callable[[str], None] | None = None,
    ):
        self._on_pending_input = on_pending_input
        self._on_input_received = on_input_received
        self._on_save_mid_run = on_save_mid_run

    def mark_pending_input(self, node_id: str, prompt: str, metadata: dict | None = None) -> None:
        """Notify that a node is waiting for human input."""
        if self._on_pending_input:
            self._on_pending_input(node_id, prompt, metadata)

    def mark_input_received(self, node_id: str) -> None:
        """Notify that human input has been received for a specific node."""
        if self._on_input_received:
            self._on_input_received(node_id)

    def save_mid_run(self, node_id: str) -> None:
        """Request a checkpoint save during a long-running node (e.g., agent loop iteration)."""
        if self._on_save_mid_run:
            self._on_save_mid_run(node_id)


class NodeCheckpointState(BaseModel):
    """Checkpoint state for a single node execution."""

    model_config = ConfigDict(extra="allow")

    node_id: str = Field(description="Unique identifier of the node")
    node_type: str = Field(description="Class name of the node")
    status: str = Field(description="Execution status (RunnableStatus value)")
    input_data: Any | None = Field(default=None, description="Input data passed to the node")
    output_data: Any | None = Field(default=None, description="Output data from the node")
    error: dict | None = Field(default=None, description="Error information if failed")
    internal_state: dict = Field(default_factory=dict, description="Node-specific state from to_checkpoint_state()")
    started_at: datetime | None = Field(default=None, description="When node execution started")
    completed_at: datetime | None = Field(default=None, description="When node execution completed")


class PendingInputContext(BaseModel):
    """Context for a workflow waiting for human input (HITL)."""

    node_id: str = Field(description="ID of the node waiting for input")
    prompt: str = Field(description="The question/prompt shown to user")
    timestamp: datetime = Field(description="When input was requested")
    metadata: dict = Field(default_factory=dict, description="Additional context (tool name, etc.)")


class FlowCheckpoint(BaseModel):
    """Complete checkpoint capturing workflow state at a point in time."""

    id: str = Field(default_factory=generate_uuid, description="Unique checkpoint identifier")
    flow_id: str = Field(description="ID of the Flow being executed")
    workflow_id: str | None = Field(default=None, description="ID of the parent Workflow")
    run_id: str = Field(description="Flow-level execution run ID")
    wf_run_id: str | None = Field(
        default=None, description="Workflow-level run ID from RunnableConfig (groups checkpoints across flows)"
    )

    status: CheckpointStatus = Field(default=CheckpointStatus.ACTIVE, description="Current checkpoint status")
    node_states: dict[str, NodeCheckpointState] = Field(
        default_factory=dict, description="State of each node, keyed by node_id"
    )
    completed_node_ids: list[str] = Field(default_factory=list, description="List of completed node IDs")
    pending_node_ids: list[str] = Field(default_factory=list, description="List of pending node IDs")

    original_input: Any = Field(default=None, description="Original input data for resume")
    original_config: dict | None = Field(default=None, description="Original RunnableConfig for resume")
    pending_inputs: dict[str, PendingInputContext] = Field(
        default_factory=dict, description="HITL contexts keyed by node_id, supports parallel approval requests"
    )

    created_at: datetime = Field(default_factory=utc_now, description="When checkpoint was created")
    updated_at: datetime = Field(default_factory=utc_now, description="When checkpoint was last updated")

    version: str = Field(default="1.0", description="Schema version for forward compatibility")
    dynamiq_version: str | None = Field(default=None, description="Library version that created this checkpoint")
    metadata: dict = Field(default_factory=dict, description="Additional debugging info")
    parent_checkpoint_id: str | None = Field(default=None, description="Parent checkpoint ID for time travel")

    def mark_node_complete(self, node_id: str, state: NodeCheckpointState) -> None:
        """Mark a node as completed and store its state."""
        self.node_states[node_id] = state
        if node_id not in self.completed_node_ids:
            self.completed_node_ids.append(node_id)
        if node_id in self.pending_node_ids:
            self.pending_node_ids.remove(node_id)
        self.updated_at = utc_now()

    def mark_pending_input(self, node_id: str, prompt: str, metadata: dict | None = None) -> None:
        """Mark a node as waiting for human input (HITL). Supports multiple parallel pending inputs."""
        self.pending_inputs[node_id] = PendingInputContext(
            node_id=node_id,
            prompt=prompt,
            timestamp=utc_now(),
            metadata=metadata or {},
        )
        self.status = CheckpointStatus.PENDING_INPUT
        self.updated_at = utc_now()

    def clear_pending_input(self, node_id: str) -> None:
        """Clear pending input for a specific node after input received."""
        self.pending_inputs.pop(node_id, None)
        if not self.pending_inputs:
            self.status = CheckpointStatus.ACTIVE
        self.updated_at = utc_now()

    def has_pending_inputs(self) -> bool:
        """Check if any nodes are waiting for human input."""
        return len(self.pending_inputs) > 0

    def get_pending_input(self, node_id: str) -> PendingInputContext | None:
        """Get pending input context for a specific node."""
        return self.pending_inputs.get(node_id)

    def is_node_completed(self, node_id: str) -> bool:
        """Check if a node has completed execution."""
        return node_id in self.completed_node_ids

    def get_node_output(self, node_id: str) -> Any | None:
        """Get output data of a completed node."""
        if node_id in self.node_states:
            return self.node_states[node_id].output_data
        return None

    def to_dict(self) -> dict:
        """Convert to a fully JSON-serializable dict, handling complex types in Any fields.

        Fields typed as Any (original_input, node input_data/output_data) can contain
        BytesIO, bytes, datetime etc. These are pre-encoded to avoid serialization issues.
        """
        any_fields = {"original_input"}
        data = self.model_dump(exclude=any_fields)
        data["original_input"] = encode_checkpoint_data(self.original_input)

        for node_id, node_state in data.get("node_states", {}).items():
            raw_state = self.node_states.get(node_id)
            if raw_state:
                node_state["input_data"] = encode_checkpoint_data(raw_state.input_data)
                node_state["output_data"] = encode_checkpoint_data(raw_state.output_data)

        return data

    def to_json(self) -> str:
        """Serialize checkpoint to JSON string."""
        return orjson.dumps(self.to_dict(), default=encode_reversible).decode("utf-8")

    def to_bytes(self) -> bytes:
        """Serialize checkpoint to bytes."""
        return orjson.dumps(self.to_dict(), default=encode_reversible)

    @classmethod
    def from_json(cls, json_str: str) -> "FlowCheckpoint":
        """Deserialize checkpoint from JSON string."""
        raw = orjson.loads(json_str)
        return cls(**decode_checkpoint_data(raw))

    @classmethod
    def from_bytes(cls, data: bytes) -> "FlowCheckpoint":
        """Deserialize checkpoint from bytes."""
        raw = orjson.loads(data)
        return cls(**decode_checkpoint_data(raw))


class CheckpointConfig(BaseModel):
    """Checkpoint configuration for flow execution.

    Used at two levels:
    - Flow-level: defines structural defaults (backend, retention, behavior). Can be expressed in YAML.
    - Run-level: passed via RunnableConfig.checkpoint to override any field per run.

    When both are provided, run-level values override flow-level defaults.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(default=False, description="Whether checkpointing is active")
    backend: Any | None = Field(default=None, description="CheckpointBackend instance for storage")
    resume_from: str | None = Field(default=None, description="Checkpoint ID to resume from (per-run)")

    checkpoint_after_node_enabled: bool = Field(default=True, description="Create checkpoint after each node")
    checkpoint_on_failure_enabled: bool = Field(default=True, description="Create checkpoint when workflow fails")
    checkpoint_mid_agent_loop_enabled: bool = Field(default=False, description="Checkpoint during long agent loops")

    max_checkpoints: int = Field(
        default=10,
        description="Maximum checkpoints to keep per flow_id. When exceeded, oldest checkpoints are removed.",
    )
    max_ttl_minutes: int | None = Field(default=None, description="Delete checkpoints older than this many minutes")
    exclude_node_ids: list[str] = Field(default_factory=list, description="Node IDs to skip checkpointing")

    context: CheckpointContext | None = Field(default=None, description="Runtime checkpoint context (set by Flow)")
