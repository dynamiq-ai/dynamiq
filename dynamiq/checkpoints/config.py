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
    - Mid-loop: request a checkpoint save during long agent loops
    - Input timeout: request a checkpoint save when an input wait times out
    """

    def __init__(
        self,
        on_save_mid_run: Callable[[str], None] | None = None,
        on_input_timeout: Callable[[str], None] | None = None,
    ):
        self._on_save_mid_run = on_save_mid_run
        self._on_input_timeout = on_input_timeout

    def save_mid_run(self, node_id: str) -> None:
        """Request a checkpoint save during a long-running node (e.g., agent loop iteration)."""
        if self._on_save_mid_run:
            self._on_save_mid_run(node_id)

    def save_on_input_timeout(self, node_id: str) -> None:
        """Request a checkpoint save when StreamingConfig input wait times out."""
        if self._on_input_timeout:
            self._on_input_timeout(node_id)


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

    checkpoint_on_start_enabled: bool = Field(
        default=True, description="Persist the checkpoint at run start, before any node executes"
    )
    checkpoint_after_node_enabled: bool = Field(default=True, description="Create checkpoint after each node")
    checkpoint_on_failure_enabled: bool = Field(default=True, description="Create checkpoint when workflow fails")
    checkpoint_on_cancel_enabled: bool = Field(default=True, description="Create checkpoint when workflow is canceled")
    checkpoint_mid_agent_loop_enabled: bool = Field(default=False, description="Checkpoint during long agent loops")
    checkpoint_on_input_timeout_enabled: bool = Field(
        default=True,
        description="Create checkpoint when StreamingConfig input wait times out",
    )

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
