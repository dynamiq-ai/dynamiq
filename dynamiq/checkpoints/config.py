from enum import Enum
from typing import Callable

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.checkpoints.backends.base import CheckpointBackend
from dynamiq.checkpoints.backends.in_memory import InMemory


class CheckpointBehavior(str, Enum):
    """How checkpoint saves are handled during execution."""

    REPLACE = "replace"
    APPEND = "append"


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


class CheckpointConfig(BaseModel):
    """Checkpoint configuration for flow execution.

    Used at two levels:
    - Flow-level: defines structural defaults (backend, retention, behavior). Can be expressed in YAML.
    - Run-level: passed via RunnableConfig.checkpoint to override any field per run.

    When both are provided, run-level values override flow-level defaults.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    enabled: bool = Field(default=False, description="Whether checkpointing is active")
    backend: CheckpointBackend = Field(default_factory=InMemory, description="CheckpointBackend instance for storage")
    resume_from: str | None = Field(default=None, description="Checkpoint ID to resume from (per-run)")
    behavior: CheckpointBehavior = Field(
        default=CheckpointBehavior.APPEND,
        description="APPEND creates a new snapshot per save for time-travel; REPLACE overwrites the same checkpoint",
    )

    checkpoint_after_node_enabled: bool = Field(default=True, description="Create checkpoint after each node")
    checkpoint_on_failure_enabled: bool = Field(default=True, description="Create checkpoint when workflow fails")
    checkpoint_mid_agent_loop_enabled: bool = Field(default=False, description="Checkpoint during long agent loops")

    max_checkpoints: int = Field(
        default=50,
        description="Maximum checkpoints to keep per flow_id. When exceeded, oldest checkpoints are removed.",
    )
    max_ttl_minutes: int | None = Field(default=None, description="Delete checkpoints older than this many minutes")
    exclude_node_ids: list[str] = Field(default_factory=list, description="Node IDs to skip checkpointing")

    context: CheckpointContext | None = Field(default=None, description="Runtime checkpoint context (set by Flow)")

    def to_dict(self) -> dict:
        data = self.model_dump(mode="json", exclude={"backend", "context", "resume_from"})
        if self.backend:
            data["backend"] = self.backend.to_dict()
        return data
