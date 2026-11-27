"""Composite memory store: several memories behind one interface, routed by path prefix."""

import builtins
import os
import posixpath
from typing import Any, TypeVar

from pydantic import ConfigDict, Field, field_validator

from .base import MemoryEntry, MemoryHit, MemoryNote, MemoryNotFoundError, MemoryStore, MemoryStoreError


_Entry = TypeVar("_Entry", bound=MemoryEntry)


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

    def scope_of(self, path: str) -> str:
        store, inner, route = self._resolve(path)
        return f"{route}{store.scope_of(inner)}"

    def describe_namespaces(self) -> dict[str, str]:
        """Compose the routes' descriptions the same way routing composes the routes themselves."""
        described: dict[str, str] = {}
        for prefix, store in self.routes.items():
            for sub_prefix, text in store.describe_namespaces().items():
                described[f"{prefix}{sub_prefix}"] = text
        return described

    def content_limit(self) -> int | None:
        """The tightest of the routes' limits, since a memory may be written to any of them."""
        return min((limit for store in self.routes.values() if (limit := store.content_limit())), default=None)

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

    def _visible_entries(self, route: str, entries: builtins.list[_Entry]) -> builtins.list[_Entry]:
        """Mount paths and keep entries owned by this route."""
        visible: list[_Entry] = []
        for entry in entries:
            path = f"{route}{entry.path}"
            _, _, owning_route = self._resolve(path)
            if owning_route == route:
                visible.append(entry.model_copy(update={"path": path}))
        return visible

    def list(self, prefix: str = "", user_id: str | None = None) -> builtins.list[MemoryEntry]:
        """List visible memories under ``prefix``, using the longest matching mount for each path."""
        entries: list[MemoryEntry] = []
        for route, store, inner in self._routes_under(prefix):
            entries.extend(self._visible_entries(route, store.list(inner, user_id)))
        return entries

    def _routes_under(self, prefix: str) -> builtins.list[tuple[str, MemoryStore, str]]:
        """Each route a listing or search of ``prefix`` reaches, with the prefix inside that route.

        Both the route ``prefix`` falls inside and every route at or below it, or a nested route's
        memories go missing from a listing of their own directory. They never overlap: an enclosing
        route is no longer than ``prefix``, one at or below it is strictly longer.
        """
        normalized = normalize_path(prefix)
        scope = f"{normalized}/" if normalized else ""
        reached: list[tuple[str, MemoryStore, str]] = []
        if normalized:
            for route in sorted(self.routes, key=len, reverse=True):
                if normalized.startswith(route):
                    reached.append((route, self.routes[route], normalized[len(route) :]))
                    break
        reached.extend((route, store, "") for route, store in self.routes.items() if route.startswith(scope))
        return reached

    def read(self, path: str, user_id: str | None = None) -> str:
        """Read from the memory owning the path."""
        store, inner, route = self._resolve(path)
        try:
            return store.read(inner, user_id)
        except MemoryNotFoundError:
            # Name the path the caller used, not the one inside the store.
            raise MemoryNotFoundError(f"Memory '{route}{inner}' not found", operation="read", path=path) from None

    def read_note(self, path: str, user_id: str | None = None) -> MemoryNote:
        """Read from the memory owning the path, with its links mounted like its own path.

        Links resolve inside one store, so a link and a backlink always share the note's route.
        """
        store, inner, route = self._resolve(path)
        try:
            note = store.read_note(inner, user_id)
        except MemoryNotFoundError:
            raise MemoryNotFoundError(f"Memory '{route}{inner}' not found", operation="read", path=path) from None
        return note.model_copy(
            update={
                "path": f"{route}{note.path}",
                "links": [
                    link.model_copy(update={"path": f"{route}{link.path}"}) if link.path else link
                    for link in note.links
                ],
                "backlinks": [f"{route}{backlink}" for backlink in note.backlinks],
            }
        )

    def search(
        self, query: str, prefix: str = "", limit: int = 10, user_id: str | None = None
    ) -> builtins.list[MemoryHit]:
        """Search visible memories under the prefix and merge the hits by score.

        A score is the share of the query's words a memory holds, so scores for one query compare
        across routes; equal scores keep each route's own order.
        """
        hits: list[MemoryHit] = []
        for route, store, inner in self._routes_under(prefix):
            route_limit = limit
            while True:
                found = store.search(query, inner, route_limit, user_id)
                visible = self._visible_entries(route, found)
                if len(visible) >= limit or len(found) < route_limit:
                    hits.extend(visible)
                    break
                # Hidden results can crowd visible matches out of a full page.
                route_limit *= 2
        hits.sort(key=lambda hit: -hit.score)
        return hits[:limit]

    def write(self, path: str, content: str, user_id: str | None = None) -> MemoryEntry:
        """Write to the memory owning the path."""
        store, inner, route = self._resolve(path)
        entry = store.write(inner, content, user_id)
        return entry.model_copy(update={"path": f"{route}{entry.path}"})

    def write_if_unchanged(
        self, path: str, content: str, expected_hash: str | None, user_id: str | None = None
    ) -> MemoryEntry:
        """Write to the memory owning the path, only while it holds what was read, or is absent."""
        store, inner, route = self._resolve(path)
        entry = store.write_if_unchanged(inner, content, expected_hash, user_id)
        return entry.model_copy(update={"path": f"{route}{entry.path}"})

    def delete(self, path: str, user_id: str | None = None) -> bool:
        """Delete from the memory owning the path."""
        store, inner, _route = self._resolve(path)
        return store.delete(inner, user_id)

    def delete_if_unchanged(self, path: str, expected_hash: str, user_id: str | None = None) -> bool:
        """Delete from the memory owning the path, only while it holds what was read."""
        store, inner, _route = self._resolve(path)
        return store.delete_if_unchanged(inner, expected_hash, user_id)
