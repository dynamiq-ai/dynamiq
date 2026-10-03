"""In-memory file storage implementation."""

import base64
import mimetypes
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO, Collection

from pydantic import ConfigDict

from dynamiq.utils.logger import logger

from .base import FileInfo, FileNotFoundError, FileStore, StorageError


class InMemoryFileStore(FileStore):
    """In-memory file storage implementation.

    This implementation stores files in memory using Python dictionaries.
    Files are lost when the process terminates.

    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **kwargs):
        """Initialize the in-memory storage.

        Args:
            **kwargs: Additional keyword arguments (ignored)
        """
        super().__init__(**kwargs)
        self._files: dict[str, dict[str, Any]] = {}

    def list_files_bytes(self, file_paths: list[str] | None = None) -> list[BytesIO]:
        """Return stored files as BytesIO objects.

        Args:
            file_paths: If provided, return only these files. Otherwise return all files.

        Returns:
            List of BytesIO objects with name, description, and content_type attributes.
        """
        keys = file_paths if file_paths else list(self._files.keys())
        files = []
        for file_path in keys:
            if file_path not in self._files:
                continue
            data = self._files[file_path]
            file = BytesIO(data["content"])
            file.name = file_path
            file.description = data["metadata"].get("description", "")
            file.content_type = data["content_type"]
            files.append(file)
        return files

    def is_empty(self) -> bool:
        """Check if the file store is empty."""
        return len(self._files) == 0

    def store(
        self,
        file_path: str | Path,
        content: str | bytes | BinaryIO,
        content_type: str = None,
        metadata: dict[str, Any] = None,
        overwrite: bool = False,
    ) -> FileInfo:
        """Store a file in memory."""
        file_path = str(file_path)

        if file_path in self._files and not overwrite:
            logger.info(f"File '{file_path}' already exists. Skipping...")
            return self._create_file_info(file_path, self._files[file_path])

        # Convert content to bytes
        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        elif isinstance(content, bytes):
            content_bytes = content
        elif hasattr(content, "read"):  # BinaryIO-like object
            content_bytes = content.read()
            if hasattr(content, "seek"):
                content.seek(0)  # Reset position for future reads
        else:
            raise StorageError(f"Unsupported content type: {type(content)}", operation="store", path=file_path)

        if content_type is None:
            content_type, _ = mimetypes.guess_type(file_path)
            if content_type is None:
                content_type = "application/octet-stream"

        now = datetime.now()
        file_info = {
            "content": content_bytes,
            "size": len(content_bytes),
            "content_type": content_type,
            "created_at": now,
            "metadata": metadata or {},
        }

        self._files[file_path] = file_info

        return self._create_file_info(file_path, file_info)

    def retrieve(self, file_path: str | Path) -> bytes:
        """Retrieve file content as bytes."""
        file_path = str(file_path)

        if file_path not in self._files:
            raise FileNotFoundError(f"File '{file_path}' not found", operation="retrieve", path=file_path)

        return self._files[file_path]["content"]

    def exists(self, file_path: str | Path) -> bool:
        """Check if file exists."""
        return str(file_path) in self._files

    def delete(self, file_path: str | Path) -> bool:
        """Delete a file."""
        file_path = str(file_path)

        if file_path in self._files:
            del self._files[file_path]
            return True

        return False

    def list_files(
        self,
        directory: str | Path = "",
        recursive: bool = False,
    ) -> list[FileInfo]:
        """List files in storage."""
        directory = str(directory)
        files_list = []

        for file_path in self._files.keys():
            if directory and not file_path.startswith(directory):
                continue

            if not recursive:
                rel_path = file_path[len(directory) :].lstrip("/")
                if "/" in rel_path:
                    continue

            files_list.append(self._create_file_info(file_path, self._files[file_path]))

        return files_list

    def to_checkpoint_state(self, max_bytes: int, file_paths: Collection[str] | None = None) -> dict[str, Any]:
        """Return the stored files in a JSON-safe form, for a checkpoint to carry.

        Only the files at ``file_paths`` are taken when it is given. Files are taken in the order
        they were stored while their contents fit in ``max_bytes``. A file that does not fit is
        left out with a warning, so the checkpoint stays small enough to save.
        """
        files: dict[str, dict[str, Any]] = {}
        skipped: list[str] = []
        total = 0
        for file_path, file_data in self._files.items():
            if file_paths is not None and file_path not in file_paths:
                continue
            if total + file_data["size"] > max_bytes:
                skipped.append(file_path)
                continue
            total += file_data["size"]
            files[file_path] = {
                "content": base64.b64encode(file_data["content"]).decode("ascii"),
                "content_type": file_data["content_type"],
                "created_at": file_data["created_at"].isoformat(),
                "metadata": file_data["metadata"],
            }
        if skipped:
            logger.warning(
                f"InMemoryFileStore: {len(skipped)} file(s) do not fit the {max_bytes}-byte checkpoint "
                f"budget and are not saved: {skipped}"
            )
        return {"files": files}

    def from_checkpoint_state(self, state: dict[str, Any]) -> None:
        """Store the files saved by ``to_checkpoint_state``, replacing any file at the same path."""
        for file_path, file_data in (state.get("files") or {}).items():
            content = base64.b64decode(file_data["content"])
            self._files[file_path] = {
                "content": content,
                "size": len(content),
                "content_type": file_data["content_type"],
                "created_at": datetime.fromisoformat(file_data["created_at"]),
                "metadata": file_data["metadata"],
            }

    def _create_file_info(self, file_path: str, file_data: dict[str, Any]) -> FileInfo:
        """Create a FileInfo object from internal file data."""
        return FileInfo(
            name=os.path.basename(file_path),
            path=file_path,
            size=file_data["size"],
            content_type=file_data["content_type"],
            created_at=file_data["created_at"],
            metadata=file_data.get("metadata", {}),
            content=file_data["content"],
        )
