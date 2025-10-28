from typing import Any

from dynamiq.auth import AuthConfig
from dynamiq.nodes.node import Node
from dynamiq.utils.utils import deep_merge


class InMemoryCredentialManager:
    """
    Prototype credential manager that keeps per-tool auth payloads in memory.

    It stores ready-to-use payload dictionaries (typically produced by `AuthConfig.to_tool_payload`)
    and merges new data on top of existing entries.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _build_key(tool: Node | None) -> str:
        if tool is None:
            return "global"
        tool_id = getattr(tool, "id", None) or ""
        tool_name = getattr(tool, "name", "") or ""
        return f"{tool_id}:{tool_name}"

    def get_payload(self, tool: Node | None) -> dict[str, Any] | None:
        """Return cached payload for the specified tool."""
        return self._store.get(self._build_key(tool))

    def store_payload(self, tool: Node | None, payload: dict[str, Any] | AuthConfig) -> None:
        """Persist payload for the tool, merging with any existing data."""
        if payload is None:
            return

        if isinstance(payload, AuthConfig):
            payload = payload.to_tool_payload()

        key = self._build_key(tool)
        existing = self._store.get(key, {})
        merged = deep_merge(payload, existing) if existing else payload
        self._store[key] = merged

    def clear(self, tool: Node | None) -> None:
        """Clear cached credentials for the tool."""
        key = self._build_key(tool)
        self._store.pop(key, None)

    def clear_all(self) -> None:
        """Clear the entire credential cache."""
        self._store = {}
