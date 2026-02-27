import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.checkpoints.checkpoint import CheckpointStatus, FlowCheckpoint
from dynamiq.utils import generate_uuid

DEFAULT_CLEANUP_FETCH_LIMIT = 1000
DEFAULT_LIST_LIMIT = 10


class CheckpointBackend(ABC, BaseModel):
    """Abstract base class for checkpoint storage backends."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="CheckpointBackend", description="Backend name")
    id: str = Field(default_factory=generate_uuid, description="Unique backend instance ID")
    max_list_results: int = Field(default=DEFAULT_LIST_LIMIT, description="Max checkpoints returned by list queries")
    max_cleanup_results: int = Field(
        default=DEFAULT_CLEANUP_FETCH_LIMIT, description="Max checkpoints fetched during cleanup"
    )

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return {}

    def to_dict(self, include_secure_params: bool = False, **kwargs) -> dict[str, Any]:
        kwargs.pop("include_secure_params", None)
        kwargs.pop("for_tracing", None)
        return self.model_dump(exclude=kwargs.pop("exclude", self.to_dict_exclude_params), **kwargs)

    @computed_field
    @cached_property
    def type(self) -> str:
        return f"{self.__module__.rsplit('.', 1)[0]}.{self.__class__.__name__}"

    @abstractmethod
    def save(self, checkpoint: FlowCheckpoint) -> str:
        """Save a checkpoint and return its ID."""
        raise NotImplementedError

    @abstractmethod
    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Load a checkpoint by ID, returns None if not found."""
        raise NotImplementedError

    @abstractmethod
    def update(self, checkpoint: FlowCheckpoint) -> None:
        """Update an existing checkpoint."""
        raise NotImplementedError

    @abstractmethod
    def delete(self, checkpoint_id: str) -> bool:
        """Delete a checkpoint, returns True if deleted."""
        raise NotImplementedError

    @abstractmethod
    def get_list_by_flow(
        self,
        flow_id: str,
        *,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a flow, newest first. Use limit=None to get all."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_by_flow(self, flow_id: str, *, status: CheckpointStatus | None = None) -> FlowCheckpoint | None:
        """Get the most recent checkpoint for a flow."""
        raise NotImplementedError

    def cleanup_by_flow(self, flow_id: str, *, keep_count: int = 10, max_ttl_minutes: int | None = None) -> int:
        """Remove old checkpoints for a flow, returns count deleted."""
        checkpoints = self.get_list_by_flow(flow_id, limit=self.max_cleanup_results)
        deleted = 0

        for i, cp in enumerate(checkpoints):
            should_delete = i >= keep_count
            if max_ttl_minutes is not None and not should_delete:
                age_minutes = (datetime.now(timezone.utc) - cp.created_at).total_seconds() / 60
                should_delete = age_minutes > max_ttl_minutes
            if should_delete and self.delete(cp.id):
                deleted += 1

        return deleted

    def get_list_by_run(self, run_id: str, *, limit: int | None = None) -> list[FlowCheckpoint]:
        """List checkpoints matching run_id or wf_run_id. Use limit=None to get all."""
        raise NotImplementedError("get_list_by_run not implemented for this backend")

    def get_list_by_flow_and_run(
        self,
        flow_id: str,
        run_id: str,
        *,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a specific flow and run (matches run_id or wf_run_id), newest first."""
        raise NotImplementedError("get_list_by_flow_and_run not implemented for this backend")

    def get_latest_by_flow_and_run(
        self,
        flow_id: str,
        run_id: str,
        *,
        status: CheckpointStatus | None = None,
    ) -> FlowCheckpoint | None:
        """Get the most recent checkpoint for a specific flow and run."""
        results = self.get_list_by_flow_and_run(flow_id, run_id, status=status, limit=1)
        return results[0] if results else None

    def get_chain(self, checkpoint_id: str) -> list[FlowCheckpoint]:
        """Walk parent_checkpoint_id links to build a checkpoint chain (newest first)."""
        chain: list[FlowCheckpoint] = []
        current_id: str | None = checkpoint_id
        seen: set[str] = set()
        while current_id and current_id not in seen:
            seen.add(current_id)
            cp = self.load(current_id)
            if not cp:
                break
            chain.append(cp)
            current_id = cp.parent_checkpoint_id
        return chain

    async def save_async(self, checkpoint: FlowCheckpoint) -> str:
        """Async save - runs sync method in thread pool to avoid blocking event loop."""
        return await asyncio.to_thread(self.save, checkpoint)

    async def load_async(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Async load - runs sync method in thread pool to avoid blocking event loop."""
        return await asyncio.to_thread(self.load, checkpoint_id)

    async def update_async(self, checkpoint: FlowCheckpoint) -> None:
        """Async update - runs sync method in thread pool to avoid blocking event loop."""
        return await asyncio.to_thread(self.update, checkpoint)

    async def delete_async(self, checkpoint_id: str) -> bool:
        """Async delete - runs sync method in thread pool to avoid blocking event loop."""
        return await asyncio.to_thread(self.delete, checkpoint_id)
