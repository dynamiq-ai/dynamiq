import threading
from datetime import datetime
from typing import Any
from uuid import uuid4

import orjson
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.checkpoints.config import CheckpointBehavior, CheckpointConfig, CheckpointContext
from dynamiq.checkpoints.types import CheckpointStatus, utc_now
from dynamiq.checkpoints.utils import decode_checkpoint_data, encode_checkpoint_data
from dynamiq.runnables import RunnableConfig, RunnableResult
from dynamiq.utils import encode_reversible, generate_uuid
from dynamiq.utils.logger import logger


class BaseCheckpointState(BaseModel):
    """Base checkpoint state model. All node checkpoint states inherit from this."""

    model_config = ConfigDict(extra="allow")

    iteration: dict | None = Field(default=None, description="IterativeCheckpointMixin state for loop-level resume")
    approval_response: dict | None = Field(default=None, description="Stored HITL approval response for resume")


class CheckpointNodeMixin:
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

    def _save_iteration_to_checkpoint(self, checkpoint_state: BaseCheckpointState) -> None:
        """Attach current iteration data to an outgoing checkpoint state."""
        iteration = self.get_iteration_state()
        checkpoint_state.iteration = iteration.model_dump()

    def _restore_iteration_from_checkpoint(self, state_dict: dict) -> None:
        """Extract iteration data from an incoming checkpoint state dict."""
        if (iteration_data := state_dict.get("iteration")) is not None:
            self._iteration_state = (
                IterationState(**iteration_data) if isinstance(iteration_data, dict) else iteration_data
            )
            self._has_restored_iteration = True


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
        Dict keys that are non-primitive (e.g. UUID) are coerced to strings.
        """
        any_fields = {"original_input"}
        data = self.model_dump(exclude=any_fields)
        data["original_input"] = encode_checkpoint_data(self.original_input)

        for node_id, node_state in data.get("node_states", {}).items():
            raw_state = self.node_states.get(node_id)
            if raw_state:
                node_state["input_data"] = encode_checkpoint_data(raw_state.input_data)
                node_state["output_data"] = encode_checkpoint_data(raw_state.output_data)
                node_state["internal_state"] = encode_checkpoint_data(raw_state.internal_state)

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


class CheckpointFlowMixin(BaseModel):
    """Mixin providing all checkpoint fields and lifecycle methods for Flow.

    Separates checkpoint orchestration from flow scheduling so the logic can be
    tested and evolved independently.

    The host class must additionally expose (set in ``__init__``):
    - ``_node_by_id: dict``  – node lookup map keyed by node ID
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    checkpoint: CheckpointConfig = Field(
        default_factory=CheckpointConfig, description="Configuration for checkpoint/resume functionality"
    )

    _node_by_id: dict = PrivateAttr(default_factory=dict)
    _checkpoint: FlowCheckpoint | None = PrivateAttr(default=None)
    _effective_checkpoint_config: CheckpointConfig | None = PrivateAttr(default=None)
    _checkpoint_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)
    _checkpoint_persisted: bool = PrivateAttr(default=False)

    def _get_effective_checkpoint_config(self, config: RunnableConfig | None) -> CheckpointConfig:
        """Merge flow-level checkpoint defaults with per-run overrides from RunnableConfig.

        Run-level values override flow-level defaults. Fields not explicitly set at run-level
        (i.e. left at their default) fall back to the flow-level value.
        """
        base = self.checkpoint
        if not config or not config.checkpoint:
            return base

        run_cfg = config.checkpoint
        run_fields = run_cfg.model_fields_set

        merged_values: dict = {}
        for field_name in CheckpointConfig.model_fields:
            if field_name in run_fields:
                merged_values[field_name] = getattr(run_cfg, field_name)
            else:
                merged_values[field_name] = getattr(base, field_name)

        return CheckpointConfig(**merged_values)

    def _is_checkpoint_active(self) -> bool:
        """Check if any checkpoint feature is enabled (init, failure, HITL, mid-loop)."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        return cfg.enabled and cfg.backend is not None

    def _is_checkpoint_after_node_enabled(self) -> bool:
        """Check if checkpointing after each node completion is enabled."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        return cfg.enabled and cfg.backend is not None and cfg.checkpoint_after_node_enabled

    def _is_checkpoint_on_failure_enabled(self) -> bool:
        """Check if checkpointing on failure is enabled."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        return cfg.enabled and cfg.backend is not None and cfg.checkpoint_on_failure_enabled

    def _is_checkpoint_on_cancel_enabled(self) -> bool:
        """Check if checkpointing on cancel is enabled."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        return cfg.enabled and cfg.backend is not None and cfg.checkpoint_on_cancel_enabled

    def _setup_checkpoint_context(self, config: RunnableConfig | None) -> RunnableConfig | None:
        """Setup checkpoint context for HITL and mid-agent-loop checkpointing.

        Callbacks are always invoked from thread-pool worker threads (both in the sync
        ``run()`` path via ThreadExecutor and in the async ``run_async()`` path via
        ``asyncio.to_thread``).  All mutation + persist operations are wrapped in the
        ``_checkpoint_lock`` (threading.Lock) so concurrent node threads cannot interleave
        checkpoint state changes.
        """
        if not self._is_checkpoint_active():
            return config

        def on_pending_input(node_id: str, prompt: str, metadata: dict | None) -> None:
            with self._checkpoint_lock:
                if self._checkpoint:
                    self._checkpoint.mark_pending_input(node_id, prompt, metadata)
                    self._save_checkpoint_unlocked()
                    logger.info(f"Flow {self.id}: checkpoint saved with PENDING_INPUT status for node {node_id}")

        def on_input_received(node_id: str) -> None:
            with self._checkpoint_lock:
                if self._checkpoint:
                    self._checkpoint.clear_pending_input(node_id)
                    self._save_checkpoint_unlocked()
                    logger.debug(f"Flow {self.id}: cleared pending input for node {node_id}")

        def on_save_mid_run(node_id: str) -> None:
            with self._checkpoint_lock:
                cfg = self._effective_checkpoint_config or self.checkpoint
                if self._checkpoint and cfg.checkpoint_mid_agent_loop_enabled:
                    node = self._node_by_id.get(node_id)
                    if not node or not isinstance(node, CheckpointNodeMixin):
                        return
                    checkpoint_state = node.to_checkpoint_state()
                    internal_state = (
                        checkpoint_state.model_dump() if hasattr(checkpoint_state, "model_dump") else checkpoint_state
                    )
                    if node_id in self._checkpoint.node_states:
                        self._checkpoint.node_states[node_id].internal_state = internal_state
                    else:
                        self._checkpoint.node_states[node_id] = NodeCheckpointState(
                            node_id=node_id,
                            node_type=node.type,
                            status=CheckpointStatus.ACTIVE.value,
                            internal_state=internal_state,
                        )
                    self._save_checkpoint_unlocked()
                    logger.debug(f"Flow {self.id}: mid-run checkpoint saved for node {node_id}")

        checkpoint_context = CheckpointContext(
            on_pending_input=on_pending_input,
            on_input_received=on_input_received,
            on_save_mid_run=on_save_mid_run,
        )

        if config is None:
            config = RunnableConfig(checkpoint=CheckpointConfig(context=checkpoint_context))
        elif config.checkpoint:
            config.checkpoint.context = checkpoint_context
        else:
            config.checkpoint = CheckpointConfig(context=checkpoint_context)

        return config

    def _load_checkpoint(self, resume_from: str | FlowCheckpoint) -> FlowCheckpoint | None:
        """Load checkpoint from ID or return if already a FlowCheckpoint instance."""
        if isinstance(resume_from, FlowCheckpoint):
            return resume_from

        cfg = self._effective_checkpoint_config or self.checkpoint
        if cfg.backend:
            return cfg.backend.load(resume_from)

        return None

    def _save_checkpoint_unlocked(self) -> None:
        """Persist the current checkpoint without acquiring ``_checkpoint_lock``.

        Callers that already hold the lock (e.g. the context callbacks created by
        ``_setup_checkpoint_context``) use this to avoid re-entering the non-reentrant lock.

        In APPEND mode every save after the initial one creates a new snapshot
        with a fresh ID and a ``parent_checkpoint_id`` link to the previous one,
        building a chain suitable for time-travel.  The very first save stores
        the checkpoint as-is (the chain root with no parent).
        In REPLACE mode the same checkpoint is overwritten in-place every time.
        """
        cfg = self._effective_checkpoint_config or self.checkpoint
        if not self._checkpoint or not cfg.backend:
            return

        try:
            now = utc_now()
            if cfg.behavior == CheckpointBehavior.APPEND and self._checkpoint_persisted:
                old_id = self._checkpoint.id
                new_checkpoint = self._checkpoint.model_copy(
                    update={
                        "id": str(uuid4()),
                        "parent_checkpoint_id": old_id,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                cfg.backend.save(new_checkpoint)
                self._checkpoint = new_checkpoint
                logger.debug(f"Flow {self.id}: checkpoint appended - {self._checkpoint.id} (parent={old_id})")
            else:
                self._checkpoint.updated_at = now
                cfg.backend.save(self._checkpoint)
                self._checkpoint_persisted = True
                logger.debug(f"Flow {self.id}: checkpoint saved - {self._checkpoint.id}")
        except Exception as e:
            logger.warning(f"Flow {self.id}: failed to save checkpoint - {e}")

    def _save_checkpoint(self) -> None:
        """Save current checkpoint to backend (thread-safe).

        Acquires ``_checkpoint_lock`` then delegates to ``_save_checkpoint_unlocked``.
        """
        with self._checkpoint_lock:
            self._save_checkpoint_unlocked()

    def _update_checkpoint(self, new_results: dict[str, RunnableResult], status: CheckpointStatus) -> None:
        """Update checkpoint with new node results (thread-safe).

        Holds ``_checkpoint_lock`` for the entire mutation+save so concurrent node-thread
        callbacks cannot interleave with the update.
        """
        with self._checkpoint_lock:
            if not self._checkpoint:
                return

            for node_id, result in new_results.items():
                node = self._node_by_id.get(node_id)
                if not node:
                    continue

                cfg = self._effective_checkpoint_config or self.checkpoint
                if node_id in cfg.exclude_node_ids:
                    continue

                internal_state = {}
                if isinstance(node, CheckpointNodeMixin):
                    checkpoint_state = node.to_checkpoint_state()
                    internal_state = (
                        checkpoint_state.model_dump() if hasattr(checkpoint_state, "model_dump") else checkpoint_state
                    )

                node_state = NodeCheckpointState(
                    node_id=node_id,
                    node_type=node.type,
                    status=result.status.value,
                    input_data=result.input,
                    output_data=result.output,
                    error=result.error.to_dict() if result.error else None,
                    internal_state=internal_state,
                    completed_at=utc_now(),
                )

                self._checkpoint.mark_node_complete(node_id, node_state)

            self._checkpoint.status = status
            self._save_checkpoint_unlocked()

    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        if not cfg.backend:
            return

        try:
            deleted = cfg.backend.cleanup_by_flow(
                self.id,
                keep_count=cfg.max_checkpoints,
                max_ttl_minutes=cfg.max_ttl_minutes,
            )
            if deleted > 0:
                logger.debug(f"Flow {self.id}: cleaned up {deleted} old checkpoints")
        except Exception as e:
            logger.warning(f"Flow {self.id}: failed to cleanup checkpoints - {e}")

    async def _load_checkpoint_async(self, resume_from: str | FlowCheckpoint) -> FlowCheckpoint | None:
        """Async load checkpoint from ID or return if already a FlowCheckpoint instance."""
        if isinstance(resume_from, FlowCheckpoint):
            return resume_from

        cfg = self._effective_checkpoint_config or self.checkpoint
        if cfg.backend:
            return await cfg.backend.load_async(resume_from)

        return None

    async def _save_checkpoint_async_unlocked(self) -> None:
        """Async persist without acquiring ``_checkpoint_lock``.

        Callers that already hold the lock use this to avoid deadlocking.
        Mirrors ``_save_checkpoint_unlocked`` but uses the backend's async API.
        """
        cfg = self._effective_checkpoint_config or self.checkpoint
        if not self._checkpoint or not cfg.backend:
            return

        try:
            now = utc_now()
            if cfg.behavior == CheckpointBehavior.APPEND and self._checkpoint_persisted:
                old_id = self._checkpoint.id
                new_checkpoint = self._checkpoint.model_copy(
                    update={
                        "id": str(uuid4()),
                        "parent_checkpoint_id": old_id,
                        "created_at": now,
                        "updated_at": now,
                    }
                )
                await cfg.backend.save_async(new_checkpoint)
                self._checkpoint = new_checkpoint
                logger.debug(f"Flow {self.id}: checkpoint appended - {self._checkpoint.id} (parent={old_id})")
            else:
                self._checkpoint.updated_at = now
                await cfg.backend.save_async(self._checkpoint)
                self._checkpoint_persisted = True
                logger.debug(f"Flow {self.id}: checkpoint saved - {self._checkpoint.id}")
        except Exception as e:
            logger.warning(f"Flow {self.id}: failed to save checkpoint - {e}")

    async def _save_checkpoint_async(self) -> None:
        """Async save of the current checkpoint to backend.

        Only called from the event loop after ``asyncio.gather`` returns (all node
        threads finished), so no thread-level contention with ``_checkpoint_lock``
        exists at call time. Uses the unlocked helper directly.
        """
        await self._save_checkpoint_async_unlocked()

    async def _update_checkpoint_async(self, new_results: dict[str, RunnableResult], status: CheckpointStatus) -> None:
        """Async update of checkpoint with new node results.

        Only called from the event loop after ``asyncio.gather`` returns (all node
        threads finished), so no thread-level contention exists. The event loop is
        single-threaded so no additional lock is needed.
        """
        if not self._checkpoint:
            return

        for node_id, result in new_results.items():
            node = self._node_by_id.get(node_id)
            if not node:
                continue

            cfg = self._effective_checkpoint_config or self.checkpoint
            if node_id in cfg.exclude_node_ids:
                continue

            internal_state = {}
            if isinstance(node, CheckpointNodeMixin):
                checkpoint_state = node.to_checkpoint_state()
                internal_state = (
                    checkpoint_state.model_dump() if hasattr(checkpoint_state, "model_dump") else checkpoint_state
                )

            node_state = NodeCheckpointState(
                node_id=node_id,
                node_type=node.type,
                status=result.status.value,
                input_data=result.input,
                output_data=result.output,
                error=result.error.to_dict() if result.error else None,
                internal_state=internal_state,
                completed_at=utc_now(),
            )

            self._checkpoint.mark_node_complete(node_id, node_state)

        self._checkpoint.status = status
        await self._save_checkpoint_async_unlocked()

    async def _cleanup_old_checkpoints_async(self) -> None:
        """Async removal of old checkpoints beyond max_checkpoints."""
        cfg = self._effective_checkpoint_config or self.checkpoint
        if not cfg.backend:
            return

        try:
            deleted = await cfg.backend.cleanup_by_flow_async(
                self.id,
                keep_count=cfg.max_checkpoints,
                max_ttl_minutes=cfg.max_ttl_minutes,
            )
            if deleted > 0:
                logger.debug(f"Flow {self.id}: cleaned up {deleted} old checkpoints")
        except Exception as e:
            logger.warning(f"Flow {self.id}: failed to cleanup checkpoints - {e}")

    def list_checkpoints(self, limit: int = 10) -> list[FlowCheckpoint]:
        """List checkpoints for this flow, newest first.

        Args:
            limit: Maximum number of checkpoints to return.
        """
        if not self.checkpoint.backend:
            return []
        return self.checkpoint.backend.get_list_by_flow(self.id, limit=limit)

    def get_latest_checkpoint(self, status: CheckpointStatus | None = None) -> FlowCheckpoint | None:
        """Get the most recent checkpoint.

        Args:
            status: Optional status filter.
        """
        if not self.checkpoint.backend:
            return None
        return self.checkpoint.backend.get_latest_by_flow(self.id, status=status)

    def get_pending_inputs(self, checkpoint_id: str | None = None) -> dict[str, PendingInputContext]:
        """Get pending input contexts (HITL) from a checkpoint.

        Args:
            checkpoint_id: Specific checkpoint ID, or None to use current/latest.
        """
        checkpoint = None
        if checkpoint_id:
            checkpoint = self.checkpoint.backend.load(checkpoint_id) if self.checkpoint.backend else None
        elif self._checkpoint:
            checkpoint = self._checkpoint
        elif self.checkpoint.backend:
            checkpoint = self.checkpoint.backend.get_latest_by_flow(self.id)

        if checkpoint:
            return checkpoint.pending_inputs
        return {}

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint.

        Args:
            checkpoint_id: The checkpoint to delete.

        Returns:
            True if deleted, False if not found.
        """
        if not self.checkpoint.backend:
            return False
        return self.checkpoint.backend.delete(checkpoint_id)

    def clear_all_checkpoints(self) -> int:
        """Delete all checkpoints for this flow.

        Returns:
            Number of checkpoints deleted.
        """
        if not self.checkpoint.backend:
            return 0
        return self.checkpoint.backend.cleanup_by_flow(self.id, keep_count=0)
