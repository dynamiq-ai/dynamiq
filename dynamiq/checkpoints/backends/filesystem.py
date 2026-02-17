import fcntl
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.checkpoints.backends.base import DEFAULT_LIST_LIMIT, CheckpointBackend
from dynamiq.checkpoints.checkpoint import CheckpointStatus, FlowCheckpoint
from dynamiq.utils import decode_reversible, encode_reversible


class FileSystem(CheckpointBackend):
    """Filesystem-based checkpoint storage with file locking."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(default="FileSystemCheckpoint")
    base_path: str = Field(default=".dynamiq/checkpoints", description="Root directory for checkpoint storage")

    _base_dir: Path = PrivateAttr()
    _data_dir: Path = PrivateAttr()
    # Index files store checkpoint IDs ordered by creation time for fast flow-based lookups
    # without scanning all checkpoint files. Each flow has its own index file.
    _flow_index_dir: Path = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        """Initialize directory structure."""
        self._base_dir = Path(self.base_path)
        self._data_dir = self._base_dir / "data"
        self._flow_index_dir = self._base_dir / "indexes"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._flow_index_dir.mkdir(parents=True, exist_ok=True)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Define parameters to exclude during serialization."""
        return {"_base_dir": True, "_data_dir": True, "_flow_index_dir": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        """Convert backend to dictionary representation."""
        return super().to_dict(include_secure_params=include_secure_params, **kwargs)

    def _checkpoint_path(self, checkpoint_id: str) -> Path:
        """Get file path for a checkpoint."""
        return self._data_dir / f"{checkpoint_id}.json"

    def _flow_index_path(self, flow_id: str) -> Path:
        """Get file path for a flow's checkpoint index."""
        safe_id = flow_id.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self._flow_index_dir / f"{safe_id}.json"

    def _lock_file(self, file_obj, exclusive: bool = True) -> None:
        """Acquire file lock."""
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)

    def _unlock_file(self, file_obj) -> None:
        """Release file lock."""
        fcntl.flock(file_obj.fileno(), fcntl.LOCK_UN)

    def save(self, checkpoint: FlowCheckpoint) -> str:
        """Save checkpoint to file."""
        checkpoint.updated_at = datetime.now(timezone.utc)
        path = self._checkpoint_path(checkpoint.id)

        with open(path, "w") as f:
            self._lock_file(f, exclusive=True)
            try:
                json.dump(checkpoint.model_dump(), f, default=encode_reversible, ensure_ascii=False)
            finally:
                self._unlock_file(f)

        self._update_flow_index(checkpoint.flow_id, checkpoint.id, checkpoint.created_at)
        return checkpoint.id

    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        """Load checkpoint from file."""
        path = self._checkpoint_path(checkpoint_id)
        if not path.exists():
            return None

        with open(path) as f:
            self._lock_file(f, exclusive=False)
            try:
                data = json.load(f, object_hook=decode_reversible)
            finally:
                self._unlock_file(f)

        return FlowCheckpoint(**data)

    def update(self, checkpoint: FlowCheckpoint) -> None:
        """Update existing checkpoint file."""
        if not self._checkpoint_path(checkpoint.id).exists():
            raise FileNotFoundError(f"Checkpoint {checkpoint.id} not found")
        self.save(checkpoint)

    def delete(self, checkpoint_id: str) -> bool:
        """Delete checkpoint file."""
        path = self._checkpoint_path(checkpoint_id)
        if not path.exists():
            return False

        checkpoint = self.load(checkpoint_id)
        path.unlink()
        if checkpoint:
            self._remove_from_flow_index(checkpoint.flow_id, checkpoint_id)
        return True

    def get_list_by_flow(
        self,
        flow_id: str,
        *,
        status: CheckpointStatus | None = None,
        limit: int | None = DEFAULT_LIST_LIMIT,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        """List checkpoints for a flow, newest first. Use limit=None to get all."""
        index_path = self._flow_index_path(flow_id)
        if not index_path.exists():
            return []

        with open(index_path) as f:
            index_data = json.load(f)

        checkpoints = []
        for entry in index_data:
            if before and datetime.fromisoformat(entry["created_at"]) >= before:
                continue

            cp = self.load(entry["id"])
            if cp is None:
                continue
            if status and cp.status != status:
                continue

            checkpoints.append(cp)
            if limit is not None and len(checkpoints) >= limit:
                break

        return checkpoints

    def get_latest_by_flow(self, flow_id: str, *, status: CheckpointStatus | None = None) -> FlowCheckpoint | None:
        """Get the most recent checkpoint for a flow."""
        results = self.get_list_by_flow(flow_id, status=status, limit=1)
        return results[0] if results else None

    def _update_flow_index(self, flow_id: str, checkpoint_id: str, created_at: datetime) -> None:
        """Add or update checkpoint entry in the flow's index file."""
        index_path = self._flow_index_path(flow_id)
        index_data = []

        if index_path.exists():
            with open(index_path) as f:
                index_data = json.load(f)

        index_data = [e for e in index_data if e["id"] != checkpoint_id]
        index_data.insert(0, {"id": checkpoint_id, "created_at": created_at.isoformat()})

        with open(index_path, "w") as f:
            json.dump(index_data, f)

    def _remove_from_flow_index(self, flow_id: str, checkpoint_id: str) -> None:
        """Remove checkpoint entry from the flow's index file."""
        index_path = self._flow_index_path(flow_id)
        if not index_path.exists():
            return

        with open(index_path) as f:
            index_data = json.load(f)

        index_data = [e for e in index_data if e["id"] != checkpoint_id]

        with open(index_path, "w") as f:
            json.dump(index_data, f)

    def get_list_by_run(self, run_id: str, *, limit: int | None = DEFAULT_LIST_LIMIT) -> list[FlowCheckpoint]:
        """List checkpoints for a specific run. Use limit=None to get all."""
        checkpoints = [
            cp for path in self._data_dir.glob("*.json") if (cp := self.load(path.stem)) and cp.run_id == run_id
        ]
        checkpoints.sort(key=lambda x: x.created_at, reverse=True)
        return checkpoints[:limit] if limit is not None else checkpoints
