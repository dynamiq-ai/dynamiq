import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.checkpoints.backends.base import CheckpointBackend
from dynamiq.checkpoints.checkpoint import CheckpointStatus, FlowCheckpoint
from dynamiq.utils import decode_reversible, encode_reversible


class FileSystem(CheckpointBackend):
    """Filesystem-based checkpoint storage.

    Directory layout::

        {base_path}/
          {flow_id}/
            {created_at}__{wf_run_id}/          # wf_run_id if available, otherwise run_id
              {created_at}__{checkpoint_id}.json

    Using ``wf_run_id`` (workflow-level run ID) for the run directory groups all
    checkpoints from the same logical workflow execution together, including resumes.
    When the flow runs standalone (no parent Workflow), ``run_id`` is used as fallback.
    """

    TIMESTAMP_FORMAT: ClassVar[str] = "%Y-%m-%dT%H-%M-%S-%f"

    name: str = Field(default="FileSystemCheckpoint")
    base_path: str = Field(default=".dynamiq/checkpoints", description="Root directory for checkpoint storage")

    _base_dir: Path = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self._base_dir = Path(self.base_path)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return {"_base_dir": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        return super().to_dict(include_secure_params=include_secure_params, **kwargs)

    @staticmethod
    def _safe_id(raw_id: str) -> str:
        """Sanitize an ID for use as a directory/file name."""
        return re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_id)

    def _format_timestamp(self, dt: datetime) -> str:
        raw = dt.strftime(self.TIMESTAMP_FORMAT)
        return raw[:-4]

    def _flow_dir(self, flow_id: str) -> Path:
        return self._base_dir / self._safe_id(flow_id)

    def _effective_run_id(self, run_id: str, wf_run_id: str | None = None) -> str:
        return wf_run_id or run_id

    def _new_run_dir(self, flow_id: str, run_id: str, created_at: datetime, wf_run_id: str | None = None) -> Path:
        effective = self._effective_run_id(run_id, wf_run_id)
        return self._flow_dir(flow_id) / f"{self._format_timestamp(created_at)}__{self._safe_id(effective)}"

    def _find_existing_run_dir(self, flow_id: str, run_id: str, wf_run_id: str | None = None) -> Path | None:
        """Find an existing run directory by matching the run ID suffix via glob."""
        effective = self._effective_run_id(run_id, wf_run_id)
        pattern = f"*__{self._safe_id(effective)}"
        flow_dir = self._flow_dir(flow_id)
        if not flow_dir.exists():
            return None
        for match in flow_dir.glob(pattern):
            if match.is_dir():
                return match
        return None

    def _get_or_create_run_dir(
        self, flow_id: str, run_id: str, created_at: datetime, wf_run_id: str | None = None
    ) -> Path:
        """Return existing run directory for this run ID, or create a new one."""
        existing = self._find_existing_run_dir(flow_id, run_id, wf_run_id)
        if existing:
            return existing
        run_dir = self._new_run_dir(flow_id, run_id, created_at, wf_run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _checkpoint_filename(self, checkpoint_id: str, created_at: datetime) -> str:
        return f"{self._format_timestamp(created_at)}__{self._safe_id(checkpoint_id)}.json"

    def _find_checkpoint_path(self, checkpoint_id: str) -> Path | None:
        """Search all flow/run dirs for a checkpoint file containing this ID."""
        safe = self._safe_id(checkpoint_id)
        for flow_dir in self._base_dir.iterdir():
            if not flow_dir.is_dir():
                continue
            for run_dir in flow_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                for cp_file in run_dir.glob(f"*__{safe}.json"):
                    return cp_file
        return None

    def save(self, checkpoint: FlowCheckpoint) -> str:
        checkpoint.updated_at = datetime.now(timezone.utc)

        run_dir = self._get_or_create_run_dir(
            checkpoint.flow_id, checkpoint.run_id, checkpoint.created_at, checkpoint.wf_run_id
        )

        filename = self._checkpoint_filename(checkpoint.id, checkpoint.created_at)
        target = run_dir / filename

        fd, tmp_path = tempfile.mkstemp(dir=run_dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(checkpoint.to_dict(), f, default=encode_reversible, ensure_ascii=False)
            os.replace(tmp_path, target)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        return checkpoint.id

    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        path = self._find_checkpoint_path(checkpoint_id)
        if not path:
            return None
        try:
            with open(path) as f:
                data = json.load(f, object_hook=decode_reversible)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
        return FlowCheckpoint(**data)

    def update(self, checkpoint: FlowCheckpoint) -> None:
        if not self._find_checkpoint_path(checkpoint.id):
            raise FileNotFoundError(f"Checkpoint {checkpoint.id} not found")
        self.save(checkpoint)

    def delete(self, checkpoint_id: str) -> bool:
        path = self._find_checkpoint_path(checkpoint_id)
        if not path:
            return False
        path.unlink()
        return True

    def get_list_by_flow(
        self,
        flow_id: str,
        *,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        flow_dir = self._flow_dir(flow_id)
        if not flow_dir.exists():
            return []

        checkpoints = []
        for run_dir in flow_dir.iterdir():
            if not run_dir.is_dir():
                continue
            for cp_file in run_dir.glob("*.json"):
                cp = self._load_file(cp_file)
                if not cp:
                    continue
                if status and cp.status != status:
                    continue
                if before and cp.created_at >= before:
                    continue
                checkpoints.append(cp)

        checkpoints.sort(key=lambda c: c.updated_at, reverse=True)
        return checkpoints[:limit] if limit is not None else checkpoints

    def get_latest_by_flow(self, flow_id: str, *, status: CheckpointStatus | None = None) -> FlowCheckpoint | None:
        results = self.get_list_by_flow(flow_id, status=status, limit=1)
        return results[0] if results else None

    def get_list_by_run(self, run_id: str, *, limit: int | None = None) -> list[FlowCheckpoint]:
        pattern = f"*__{self._safe_id(run_id)}"
        checkpoints = []
        for flow_dir in self._base_dir.iterdir():
            if not flow_dir.is_dir():
                continue
            for run_dir in flow_dir.glob(pattern):
                if not run_dir.is_dir():
                    continue
                for cp_file in run_dir.glob("*.json"):
                    cp = self._load_file(cp_file)
                    if cp and (cp.run_id == run_id or cp.wf_run_id == run_id):
                        checkpoints.append(cp)

        checkpoints.sort(key=lambda x: x.created_at, reverse=True)
        return checkpoints[:limit] if limit is not None else checkpoints

    def get_list_by_flow_and_run(
        self,
        flow_id: str,
        run_id: str,
        *,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
    ) -> list[FlowCheckpoint]:
        run_dir = self._find_existing_run_dir(flow_id, run_id)
        if not run_dir:
            return []

        checkpoints = []
        for cp_file in run_dir.glob("*.json"):
            cp = self._load_file(cp_file)
            if not cp:
                continue
            if cp.run_id != run_id and cp.wf_run_id != run_id:
                continue
            if status and cp.status != status:
                continue
            checkpoints.append(cp)

        checkpoints.sort(key=lambda c: c.updated_at, reverse=True)
        return checkpoints[:limit] if limit is not None else checkpoints

    def _load_file(self, path: Path) -> FlowCheckpoint | None:
        try:
            with open(path) as f:
                data = json.load(f, object_hook=decode_reversible)
            return FlowCheckpoint(**data)
        except (json.JSONDecodeError, FileNotFoundError, Exception):
            return None
