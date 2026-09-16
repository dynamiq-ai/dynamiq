"""Composite memory store: several memories behind one interface, routed by path prefix."""

import os
import posixpath
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from .base import MemoryEntry, MemoryNotFoundError, MemoryStore, MemoryStoreError


def normalize_path(path: str) -> str:
    """Normalize a memory path for route matching.

    Strips a leading ``/`` and collapses ``.`` and redundant separators, so ``/team/naming.md`` and
    ``team/naming.md`` address the same memory.
    """
    cleaned = str(path).replace(os.sep, "/").lstrip("/")
    if not cleaned:
        return ""
    normalized = posixpath.normpath(cleaned)
    return "" if normalized == "." else normalized


class CompositeMemoryStore(MemoryStore):
    """Routes memories to different backends by path prefix.

    >>> CompositeMemoryStore(routes={
    ...     "user/": DynamiqMemoryStore(..., description="What you learn about this user."),
    ...     "team/": DynamiqMemoryStore(..., description="Conventions the whole team follows."),
    ... })

    The prefix is a **mount point**, not part of the key: it is stripped before the store is called
    and restored on everything handed back, exactly as a mount point is absent from a file's path on
    the device. So a route can be renamed without orphaning what is already stored, and one store
    can be mounted at different prefixes by different agents.

    That makes each route's store its own namespace, so two routes must not address the same one —
    their paths would collide after stripping. The validator refuses it rather than letting writes
    silently overwrite each other.

    There is deliberately no default: a catch-all would file a mistyped path somewhere plausible
    instead of failing, so an unrouted path raises and names the prefixes that do exist.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    routes: dict[str, MemoryStore] = Field(
        ...,
        description="Mapping of path prefix to the memory serving it. Longest matching prefix wins.",
    )

    @field_validator("routes")
    @classmethod
    def normalize_route_prefixes(cls, routes: dict[str, MemoryStore]) -> dict[str, MemoryStore]:
        """Normalize prefixes, reject an empty set, and refuse two routes over one prefix or one store."""
        if not routes:
            raise ValueError("CompositeMemoryStore needs at least one route.")

        normalized: dict[str, MemoryStore] = {}
        mounted_as: dict[str, str] = {}
        for prefix, store in routes.items():
            key = normalize_path(prefix)
            if not key:
                raise ValueError("Route prefix must not be empty: every memory is addressed by a prefix.")
            mount = f"{key}/"
            if mount in normalized:
                # Two spellings of one prefix: keying on the normalized form keeps only the last.
                raise ValueError(
                    f"Routes '{mounted_as[mount]}' and '{prefix}' both mount at '{mount}'. Give each store "
                    "its own prefix: only one of them would be reachable."
                )
            normalized[mount] = store
            mounted_as[mount] = prefix

        seen: dict[Any, str] = {}
        for prefix, store in normalized.items():
            identity = store.identity()
            if identity in seen:
                raise ValueError(
                    f"Routes '{seen[identity]}' and '{prefix}' address the same memory store. Give each route "
                    "its own store: the prefix is stripped before the store is called, so both would write "
                    "to the same keys."
                )
            seen[identity] = prefix
        return normalized

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Sub-stores serialize themselves."""
        return {"routes": True}

    def to_dict(self, **kwargs) -> dict[str, Any]:
        """Serialize the composite, delegating each route to its own backend."""
        for_tracing = kwargs.pop("for_tracing", False)
        include_secure_params = kwargs.pop("include_secure_params", False)
        exclude = kwargs.pop("exclude", self.to_dict_exclude_params)
        data = self.model_dump(exclude=exclude, **kwargs)
        data["type"] = self.type
        data["routes"] = {
            prefix: store.to_dict(for_tracing=for_tracing, include_secure_params=include_secure_params)
            for prefix, store in self.routes.items()
        }
        return data

    def describe_namespaces(self) -> dict[str, str]:
        """Compose the routes' descriptions the same way routing composes the routes themselves."""
        described: dict[str, str] = {}
        for prefix, store in self.routes.items():
            for sub_prefix, text in store.describe_namespaces().items():
                described[f"{prefix}{sub_prefix}"] = text
        return described

    def identity(self):
        """A composite is distinct from its routes, and from any other set of routes."""
        return (self.type, tuple(sorted((prefix, store.identity()) for prefix, store in self.routes.items())))

    def _resolve(self, path: str) -> tuple[MemoryStore, str, str]:
        """Return the store owning ``path``, the path *within* it, and the route it was mounted at."""
        normalized = normalize_path(path)
        match = ""
        for prefix in self.routes:
            if normalized.startswith(prefix) and len(prefix) > len(match):
                match = prefix
        if not match:
            known = ", ".join(sorted(self.routes)) or "none"
            raise MemoryStoreError(
                f"'{path}' is not in any memory. Paths must start with one of: {known}.",
                path=path,
            )
        return self.routes[match], normalized[len(match) :], match

    @staticmethod
    def _mounted(route: str, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        """Put the mount point back on paths coming out of a store."""
        return [entry.model_copy(update={"path": f"{route}{entry.path}"}) for entry in entries]

    def list(self, prefix: str = "", user_id: str | None = None) -> list[MemoryEntry]:
        """List memories under ``prefix``: the route it falls inside, plus every route at or below it.

        Both must answer, or a nested route's memories are missing from a listing of their own
        directory. They never overlap: an enclosing route is no longer than ``prefix``, one at or
        below it is strictly longer.
        """
        normalized = normalize_path(prefix)
        scope = f"{normalized}/" if normalized else ""
        entries: list[MemoryEntry] = []

        if normalized:
            for route in sorted(self.routes, key=len, reverse=True):
                if normalized.startswith(route):
                    entries.extend(
                        self._mounted(route, self.routes[route].list(normalized[len(route) :], user_id))
                    )
                    break

        entries.extend(
            entry
            for route, store in self.routes.items()
            if route.startswith(scope)
            for entry in self._mounted(route, store.list(user_id=user_id))
        )
        return entries

    def read(self, path: str, user_id: str | None = None) -> str:
        """Read from the memory owning the path."""
        store, inner, route = self._resolve(path)
        try:
            return store.read(inner, user_id)
        except MemoryNotFoundError:
            # Name the path the caller used, not the one inside the store.
            raise MemoryNotFoundError(f"Memory '{route}{inner}' not found", operation="read", path=path) from None

    def write(self, path: str, content: str, user_id: str | None = None) -> MemoryEntry:
        """Write to the memory owning the path."""
        store, inner, route = self._resolve(path)
        entry = store.write(inner, content, user_id)
        return entry.model_copy(update={"path": f"{route}{entry.path}"})

    def delete(self, path: str, user_id: str | None = None) -> bool:
        """Delete from the memory owning the path."""
        store, inner, _route = self._resolve(path)
        return store.delete(inner, user_id)
