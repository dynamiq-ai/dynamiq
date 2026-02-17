"""Abstract base class for checkpoint storage backends."""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from functools import cached_property
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

from dynamiq.checkpoints.checkpoint import CheckpointStatus, FlowCheckpoint
from dynamiq.utils import generate_uuid

# Maximum number of checkpoints to fetch during cleanup operations
CLEANUP_FETCH_LIMIT = 1000

# Default limit for list operations (None means no limit)
DEFAULT_LIST_LIMIT = 10


class CheckpointBackend(ABC, BaseModel):
    """Abstract base class for checkpoint storage backends."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="CheckpointBackend", description="Backend name")
    id: str = Field(default_factory=generate_uuid, description="Unique backend instance ID")

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
        limit: int | None = DEFAULT_LIST_LIMIT,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a flow, newest first. Use limit=None to get all."""
        raise NotImplementedError

    @abstractmethod
    def get_latest_by_flow(self, flow_id: str, *, status: CheckpointStatus | None = None) -> FlowCheckpoint | None:
        """Get the most recent checkpoint for a flow."""
        raise NotImplementedError

    def cleanup_by_flow(self, flow_id: str, *, keep_count: int = 10, older_than_hours: int | None = None) -> int:
        """Remove old checkpoints for a flow, returns count deleted."""
        checkpoints = self.get_list_by_flow(flow_id, limit=CLEANUP_FETCH_LIMIT)
        deleted = 0

        for i, cp in enumerate(checkpoints):
            should_delete = i >= keep_count
            if older_than_hours and not should_delete:
                age_seconds = (datetime.now(timezone.utc) - cp.created_at).total_seconds()
                age_hours = age_seconds / 3600
                should_delete = age_hours > older_than_hours
            if should_delete and self.delete(cp.id):
                deleted += 1

        return deleted

    def get_list_by_run(self, run_id: str, *, limit: int | None = DEFAULT_LIST_LIMIT) -> list[FlowCheckpoint]:
        """List checkpoints for a specific run. Use limit=None to get all."""
        raise NotImplementedError("get_list_by_run not implemented for this backend")

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
