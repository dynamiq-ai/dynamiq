"""Composite file storage that routes path prefixes to different backends."""

import os
import posixpath
from io import BytesIO
from pathlib import Path
from typing import Any, BinaryIO

from pydantic import ConfigDict, Field, field_validator

from .base import FileInfo, FileStore


def normalize_path(file_path: str | Path) -> str:
    """Normalize a path for route matching.

    Strips a leading ``/`` and collapses ``.``/redundant separators, so ``/memories/notes.md`` and
    ``memories/notes.md`` address the same file. The leading slash matters because the file tools
    reject absolute paths before a store ever sees them (``validate_file_path``), while a config
    author may still naturally write the prefix as ``/memories/``.
    """
    path = str(file_path).replace(os.sep, "/").lstrip("/")
    if not path:
        return ""
    normalized = posixpath.normpath(path)
    return "" if normalized == "." else normalized


class CompositeFileStore(FileStore):
    """Routes file operations to different backends by path prefix.

    Paths matching a route prefix are served by that route's store; everything else falls through to
    ``default``. This is what lets one set of file tools span an ephemeral workspace and a
    persistent, cross-conversation namespace:

    >>> CompositeFileStore(
    ...     default=InMemoryFileStore(),
    ...     routes={"memories/": DynamiqFileStore(memory_store_id="ms-123")},
    ... )

    Routed stores receive the **full** normalized path, prefix included, so nothing has to rewrite
    paths on the way back out and each backend stays independently usable.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    default: FileStore = Field(..., description="Store handling paths that match no route.")
    routes: dict[str, FileStore] = Field(
        default_factory=dict,
        description="Mapping of path prefix to the store serving it. Longest matching prefix wins.",
    )

    @field_validator("routes")
    @classmethod
    def normalize_route_prefixes(cls, routes: dict[str, FileStore]) -> dict[str, FileStore]:
        """Normalize route prefixes so ``/memories/`` and ``memories/`` behave identically."""
        normalized: dict[str, FileStore] = {}
        for prefix, store in routes.items():
            key = normalize_path(prefix)
            if not key:
                raise ValueError("Route prefix must not be empty or the store root.")
            normalized[f"{key}/"] = store
        return normalized

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Sub-stores serialize themselves, so exclude them from the plain dump."""
        return {"default": True, "routes": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Serialize the composite and delegate sub-store serialization to each backend."""
        for_tracing = kwargs.pop("for_tracing", False)
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["type"] = self.type

        sub_kwargs = {"for_tracing": for_tracing, "include_secure_params": include_secure_params, **kwargs}
        data["default"] = self.default.to_dict(**sub_kwargs)
        data["routes"] = {prefix: store.to_dict(**sub_kwargs) for prefix, store in self.routes.items()}
        return data

    def _resolve(self, file_path: str | Path) -> tuple[FileStore, str]:
        """Return the store serving ``file_path`` and its normalized form. Longest prefix wins."""
        normalized = normalize_path(file_path)
        match = ""
        for prefix in self.routes:
            if normalized.startswith(prefix) and len(prefix) > len(match):
                match = prefix
        if match:
            return self.routes[match], normalized
        return self.default, normalized

    def _route_for_directory(self, directory: str) -> tuple[FileStore, str] | None:
        """Return the store owning ``directory`` when the directory sits inside a single route."""
        normalized = normalize_path(directory)
        if not normalized:
            return None
        candidate = f"{normalized}/"

        # Inside a route: the longest matching prefix owns it, as in `_resolve`.
        owner, match = None, ""
        for prefix, store in self.routes.items():
            if candidate.startswith(prefix) and len(prefix) > len(match):
                owner, match = store, prefix
        if owner is not None:
            return owner, normalized

        # A directory above routes belongs to a single store only when it holds exactly one. Above
        # several - `memories/` over `memories/handbook/` and `memories/me/` - it belongs to none of
        # them, and the caller merges instead of picking whichever route happened to come first.
        below = [store for prefix, store in self.routes.items() if prefix.startswith(candidate)]
        return (below[0], normalized) if len(below) == 1 else None

    @staticmethod
    def _list_files(store: FileStore, directory: str, recursive: bool, pattern: str | None) -> list[FileInfo]:
        """Call ``list_files`` on a store that may not accept the optional ``pattern`` argument.

        ``InMemoryFileStore.list_files`` drops ``pattern`` even though the base class declares it;
        ``FileReadTool`` carries the same fallback for the same reason.
        """
        try:
            return store.list_files(directory=directory, recursive=recursive, pattern=pattern)
        except TypeError:
            return store.list_files(directory=directory, recursive=recursive)

    def supports_extracted_text_cache(self, file_path: str | Path = "") -> bool:
        """Defer to the backend owning the path, so a workspace file may cache while a routed one does not."""
        store, path = self._resolve(file_path)
        return store.supports_extracted_text_cache(path)

    def store(
        self,
        file_path: str | Path,
        content: str | bytes | BinaryIO,
        content_type: str = None,
        metadata: dict[str, Any] = None,
        overwrite: bool = False,
    ) -> FileInfo:
        """Store a file in the backend owning its path."""
        store, path = self._resolve(file_path)
        return store.store(
            file_path=path,
            content=content,
            content_type=content_type,
            metadata=metadata,
            overwrite=overwrite,
        )

    def retrieve(self, file_path: str | Path) -> bytes:
        """Retrieve a file from the backend owning its path."""
        store, path = self._resolve(file_path)
        return store.retrieve(path)

    def exists(self, file_path: str | Path) -> bool:
        """Check existence in the backend owning the path."""
        store, path = self._resolve(file_path)
        return store.exists(path)

    def delete(self, file_path: str | Path) -> bool:
        """Delete a file from the backend owning its path."""
        store, path = self._resolve(file_path)
        return store.delete(path)

    def list_files(
        self,
        directory: str | Path = "",
        recursive: bool = False,
        pattern: str = None,
    ) -> list[FileInfo]:
        """List files, delegating to one route or merging across all of them at the root."""
        routed = self._route_for_directory(str(directory))
        if routed is not None:
            store, normalized = routed
            return self._list_files(store, normalized, recursive, pattern)

        normalized = normalize_path(directory)
        results = [
            info
            for info in self._list_files(self.default, normalized, recursive, pattern)
            if not any(normalize_path(info.path).startswith(prefix) for prefix in self.routes)
        ]

        # Every route nested under the requested directory answers too, so listing the root the
        # memory protocol names sees all of them. At the store root that is every route, which is
        # the behaviour this generalises.
        scope = f"{normalized}/" if normalized else ""
        for prefix, store in self.routes.items():
            if prefix.startswith(scope):
                results.extend(self._list_files(store, prefix.rstrip("/"), recursive, pattern))

        return results

    def list_files_bytes(self, file_paths: list[str] | None = None) -> list[BytesIO]:
        """Return files as ``BytesIO`` objects, grouped by the store that owns each path.

        With no explicit paths only the default store answers. ``AgentBase._inject_files_into_tool``
        calls this with no arguments on *every* tool invocation to build its attachment map, so
        fanning out to a remote route would mean downloading the whole persistent store per tool
        call. Persistent files are read by path, not injected as attachments.
        """
        if not file_paths:
            return self.default.list_files_bytes()

        grouped: dict[int, tuple[FileStore, list[str]]] = {}
        for file_path in file_paths:
            store, path = self._resolve(file_path)
            grouped.setdefault(id(store), (store, []))[1].append(path)

        files: list[BytesIO] = []
        for store, paths in grouped.values():
            files.extend(store.list_files_bytes(paths))
        return files
