from datetime import datetime, timezone
from threading import Lock
from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.checkpoints.backends.base import CheckpointBackend
from dynamiq.checkpoints.checkpoint import CheckpointStatus, FlowCheckpoint


class InMemory(CheckpointBackend):
    """Thread-safe in-memory checkpoint storage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="InMemoryCheckpoint")

    _checkpoints: dict[str, FlowCheckpoint] = PrivateAttr(default_factory=dict)
    _lock: Lock = PrivateAttr(default_factory=Lock)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return {"_checkpoints": True, "_lock": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        """Convert backend to dictionary representation."""
        return super().to_dict(include_secure_params=include_secure_params, **kwargs)

    def save(self, checkpoint: FlowCheckpoint) -> str:
        """Save checkpoint to memory."""
        checkpoint.updated_at = datetime.now(timezone.utc)
        with self._lock:
            self._checkpoints[checkpoint.id] = checkpoint.model_copy(deep=True)
        return checkpoint.id

    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Load checkpoint from memory."""
        with self._lock:
            cp = self._checkpoints.get(checkpoint_id)
            return cp.model_copy(deep=True) if cp else None

    def update(self, checkpoint: FlowCheckpoint) -> None:
        """Update existing checkpoint in memory."""
        checkpoint.updated_at = datetime.now(timezone.utc)
        with self._lock:
            if checkpoint.id not in self._checkpoints:
                raise FileNotFoundError(f"Checkpoint {checkpoint.id} not found")
            self._checkpoints[checkpoint.id] = checkpoint.model_copy(deep=True)

    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint from memory."""
        with self._lock:
            if checkpoint_id in self._checkpoints:
                del self._checkpoints[checkpoint_id]
                return True
            return False

    def get_list_by_flow(
        self,
        flow_id: str,
        *,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a flow, newest first. Use limit=None to get all."""
        with self._lock:
            result = [
                cp.model_copy(deep=True)
                for cp in self._checkpoints.values()
                if cp.flow_id == flow_id
                and (status is None or cp.status == status)
                and (before is None or cp.created_at < before)
            ]
            result.sort(key=lambda x: x.created_at, reverse=True)
            return result[:limit] if limit is not None else result

    def get_latest_by_flow(self, flow_id: str, *, status: CheckpointStatus | None = None) -> FlowCheckpoint | None:
        """Get the most recent checkpoint for a flow."""
        results = self.get_list_by_flow(flow_id, status=status, limit=1)
        return results[0] if results else None

    def _matches_run(self, cp: FlowCheckpoint, run_id: str) -> bool:
        return cp.run_id == run_id or cp.wf_run_id == run_id

    def get_list_by_run(self, run_id: str, *, limit: int | None = None) -> list[FlowCheckpoint]:
        """List checkpoints matching run_id or wf_run_id. Use limit=None to get all."""
        with self._lock:
            result = [cp.model_copy(deep=True) for cp in self._checkpoints.values() if self._matches_run(cp, run_id)]
            result.sort(key=lambda x: x.created_at, reverse=True)
            return result[:limit] if limit is not None else result

    def get_list_by_flow_and_run(
        self,
        flow_id: str,
        run_id: str,
        *,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a specific flow and run (matches run_id or wf_run_id), newest first."""
        with self._lock:
            result = [
                cp.model_copy(deep=True)
                for cp in self._checkpoints.values()
                if cp.flow_id == flow_id and self._matches_run(cp, run_id) and (status is None or cp.status == status)
            ]
            result.sort(key=lambda x: x.updated_at, reverse=True)
            return result[:limit] if limit is not None else result

    def clear(self) -> None:
        """Clear all checkpoints from memory."""
        with self._lock:
            self._checkpoints.clear()
