from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Iterator

from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.checkpoints.backends._filesystem_lock import directory_lock
from dynamiq.checkpoints.backends.base import CheckpointBackend
from dynamiq.checkpoints.types import CheckpointStatus
from dynamiq.utils import decode_reversible, encode_reversible

if TYPE_CHECKING:
    from dynamiq.checkpoints.checkpoint import FlowCheckpoint


class FileSystem(CheckpointBackend):
    """Filesystem-based checkpoint storage.

    Directory layout::

        {base_path}/
          {digest}__{flow_slug}/
            {created_at}__{digest}__{run_slug}/
              {created_at}__{digest}__{checkpoint_slug}.json

    Using ``wf_run_id`` (workflow-level run ID) for the run directory groups all
    checkpoints from the same logical workflow execution together, including resumes.
    When the flow runs standalone (no parent Workflow), ``run_id`` is used as fallback.
    Paths identify the initial placement, not the current run: an upsert replaces the
    existing file even when a resume changes its run IDs. Queries use the stored data.
    Legacy paths remain readable and are updated in place. Duplicate legacy records
    raise on reads; saving an authoritative checkpoint consolidates those copies.
    Listings fail on invalid records rather than silently resuming an older snapshot:
    an unreadable payload cannot safely be assigned to a flow or run. Queries scan
    the store because a record's current scope may differ from its initial directory.
    All processes writing this directory must use this backend's locking protocol.
    """

    TIMESTAMP_FORMAT: ClassVar[str] = "%Y-%m-%dT%H-%M-%S-%f"

    name: str = Field(default="FileSystemCheckpoint")
    base_path: str = Field(default=".dynamiq/checkpoints", description="Root directory for checkpoint storage")

    _base_dir: Path = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        self._base_dir = Path(self.base_path).resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        return {"_base_dir": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        return super().to_dict(include_secure_params=include_secure_params, **kwargs)

    @staticmethod
    def _safe_id(raw_id: str) -> str:
        """Keep names bounded and distinct on case-insensitive filesystems too."""
        slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", raw_id)[:8]
        # 128 digest bits keep names collision-resistant without tripling path
        # length for UUID identities on Windows installations with MAX_PATH limits.
        return f"{sha256(raw_id.encode('utf-8')).hexdigest()[:32]}__{slug}"

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
        """Resolve an exact identity; caller must hold the directory lock."""
        paths = self._matching_paths(checkpoint_id)
        if len(paths) > 1:
            raise ValueError(f"Ambiguous checkpoint {checkpoint_id!r}: {len(paths)} stored copies")
        return paths[0] if paths else None

    def _checkpoint_paths(self) -> Iterator[Path]:
        """Visit only the supported three-level layout, without following symlinks."""
        for flow_dir in self._base_dir.iterdir():
            if flow_dir.is_symlink() or not flow_dir.is_dir():
                continue
            for run_dir in flow_dir.iterdir():
                if run_dir.is_symlink() or not run_dir.is_dir():
                    continue
                for cp_file in run_dir.glob("*.json"):
                    if not cp_file.is_symlink() and cp_file.is_file():
                        yield cp_file

    def _matching_paths(self, checkpoint_id: str) -> list[Path]:
        legacy_id = re.sub(r"[^a-zA-Z0-9_\-]", "_", checkpoint_id)
        suffixes = (f"__{self._safe_id(checkpoint_id)}.json", f"__{legacy_id}.json")
        paths = []
        for path in self._checkpoint_paths():
            if path.name.endswith(suffixes):
                # A legacy filename can describe several IDs. Never trust it for
                # loading, overwriting or deleting the actual record.
                checkpoint = self._load_file(path)
                if checkpoint.id == checkpoint_id:
                    paths.append(path)
        return sorted(paths)

    def save(self, checkpoint: FlowCheckpoint) -> str:
        copy = checkpoint.model_copy(deep=True)
        with directory_lock(self._base_dir):
            self._save(copy, self._matching_paths(copy.id))
        return copy.id

    def _save(self, checkpoint: FlowCheckpoint, existing: list[Path]) -> None:
        if existing:
            target = existing[0]
        else:
            run_dir = self._get_or_create_run_dir(
                checkpoint.flow_id, checkpoint.run_id, checkpoint.created_at, checkpoint.wf_run_id
            )
            target = run_dir / self._checkpoint_filename(checkpoint.id, checkpoint.created_at)

        fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(checkpoint.to_dict(), f, default=encode_reversible, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, target)
        except BaseException:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

        # Do not remove the last known good data until the replacement is published.
        # If consolidation fails or the process dies, reads detect the ambiguity.
        for duplicate in existing[1:]:
            duplicate.unlink()

    def load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        with directory_lock(self._base_dir):
            return self._load(checkpoint_id)

    def _load(self, checkpoint_id: str) -> FlowCheckpoint | None:
        path = self._find_checkpoint_path(checkpoint_id)
        return self._load_file(path) if path else None

    def update(self, checkpoint: FlowCheckpoint) -> None:
        copy = checkpoint.model_copy(deep=True)
        with directory_lock(self._base_dir):
            existing = self._matching_paths(copy.id)
            if not existing:
                raise FileNotFoundError(f"Checkpoint {copy.id} not found")
            self._save(copy, existing)

    def delete(self, checkpoint_id: str) -> bool:
        with directory_lock(self._base_dir):
            return self._delete(checkpoint_id)

    def _delete(self, checkpoint_id: str) -> bool:
        paths = self._matching_paths(checkpoint_id)
        for path in paths:
            path.unlink()
        return bool(paths)

    def _list(self, matches: Callable[[FlowCheckpoint], bool]) -> list[FlowCheckpoint]:
        checkpoints = {}
        seen = set()
        duplicates = set()
        for path in self._checkpoint_paths():
            checkpoint = self._load_file(path)
            if checkpoint.id in seen:
                duplicates.add(checkpoint.id)
            seen.add(checkpoint.id)
            if matches(checkpoint):
                checkpoints[checkpoint.id] = checkpoint
        # A stale version may fail the query's filters while the other passes.
        # Filtering first would make a conflicting identity look unambiguous.
        if ambiguous := duplicates.intersection(checkpoints):
            raise ValueError(f"Ambiguous checkpoints with duplicate stored copies: {sorted(ambiguous)!r}")
        return sorted(checkpoints.values(), key=lambda cp: cp.created_at, reverse=True)

    def get_list_by_flow(
        self,
        flow_id: str,
        *,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
        before: datetime | None = None,
    ) -> list[FlowCheckpoint]:
        with directory_lock(self._base_dir):
            checkpoints = self._list(
                lambda cp: cp.flow_id == flow_id
                and (status is None or cp.status == status)
                and (before is None or cp.created_at < before)
            )
        return checkpoints[:limit] if limit is not None else checkpoints

    def get_latest_by_flow(self, flow_id: str, *, status: CheckpointStatus | None = None) -> FlowCheckpoint | None:
        results = self.get_list_by_flow(flow_id, status=status, limit=1)
        return results[0] if results else None

    def get_list_by_run(self, run_id: str, *, limit: int | None = None) -> list[FlowCheckpoint]:
        with directory_lock(self._base_dir):
            checkpoints = self._list(lambda cp: cp.run_id == run_id or cp.wf_run_id == run_id)
        return checkpoints[:limit] if limit is not None else checkpoints

    def get_list_by_flow_and_run(
        self,
        flow_id: str,
        run_id: str,
        *,
        wf_run_id: str | None = None,
        status: CheckpointStatus | None = None,
        limit: int | None = None,
    ) -> list[FlowCheckpoint]:
        ids_to_match = {run_id}
        if wf_run_id:
            ids_to_match.add(wf_run_id)

        with directory_lock(self._base_dir):
            checkpoints = self._list(
                lambda cp: cp.flow_id == flow_id
                and (status is None or cp.status == status)
                and (cp.run_id in ids_to_match or cp.wf_run_id in ids_to_match)
            )
        return checkpoints[:limit] if limit is not None else checkpoints

    def cleanup_by_flow(self, flow_id: str, *, keep_count: int = 10, max_ttl_minutes: int | None = None) -> int:
        # A resumed checkpoint must not be deleted between retention's query and delete.
        with directory_lock(self._base_dir):
            checkpoints = self._list(lambda cp: cp.flow_id == flow_id)[: self.max_cleanup_results]
            continued_from = {cp.parent_checkpoint_id for cp in checkpoints}
            deleted = 0
            now = datetime.now(timezone.utc)
            for index, cp in enumerate(checkpoints):
                waiting = (
                    keep_count > 0
                    and cp.status in (CheckpointStatus.PAUSED, CheckpointStatus.PENDING_INPUT)
                    and cp.id not in continued_from
                )
                expired = max_ttl_minutes is not None and (now - cp.created_at).total_seconds() / 60 > max_ttl_minutes
                if (index >= keep_count and not waiting) or expired:
                    deleted += self._delete(cp.id)
            return deleted

    def get_chain(self, checkpoint_id: str) -> list[FlowCheckpoint]:
        with directory_lock(self._base_dir):
            chain = []
            seen = set()
            current_id = checkpoint_id
            while current_id and current_id not in seen:
                seen.add(current_id)
                checkpoint = self._load(current_id)
                if checkpoint is None:
                    break
                chain.append(checkpoint)
                current_id = checkpoint.parent_checkpoint_id
            return chain

    def _load_file(self, path: Path) -> FlowCheckpoint:
        from dynamiq.checkpoints.checkpoint import FlowCheckpoint  # deferred: avoids import cycle at package init

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f, object_hook=decode_reversible)
            required = {"id", "flow_id", "run_id", "created_at", "updated_at", "status"}
            if not isinstance(data, dict) or not required.issubset(data):
                # Defaults generate fresh identities/timestamps when creating a
                # checkpoint, but cannot reconstruct missing persisted state.
                raise ValueError("Missing persisted checkpoint identity, timestamps or status")
            return FlowCheckpoint(**data)
        except (ValueError, TypeError, UnicodeError) as exc:
            raise ValueError(f"Invalid checkpoint in {path}") from exc
