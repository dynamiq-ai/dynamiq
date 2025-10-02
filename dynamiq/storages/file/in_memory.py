"""In-memory file storage implementation."""

import mimetypes
import os
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

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

    def list_files_bytes(self) -> list[BytesIO]:
        """List files in storage and return the content as bytes in BytesIO objects.

        Returns:
            List of BytesIO objects
        """
        files = []

        for file_path in self._files.keys():
            file = BytesIO(self._files[file_path]["content"])
            file.name = file_path
            file.description = self._files[file_path]["metadata"].get("description", "")
            file.content_type = self._files[file_path]["content_type"]
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
