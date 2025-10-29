from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field

from dynamiq.connections import Dynamiq as DynamiqConnection, HTTPMethod
from dynamiq.memory.backends.base import MemoryBackend
from dynamiq.prompts import Message, MessageRole
from dynamiq.utils.logger import logger


class DynamiqMemoryError(Exception):
    """Base exception for errors raised by the Dynamiq memory backend."""

    pass


class Dynamiq(MemoryBackend):
    """Memory backend backed by the Dynamiq API."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "Dynamiq"
    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    project_id: str | None = Field(
        default=None,
        description="Optional project identifier used when locating existing memory resources.",
    )
    memory_id: str | None = Field(
        default=None,
        description="Identifier of the remote memory resource.",
    )
    timeout: float = Field(default=10.0, description="Timeout in seconds for API requests.")
    base_path: str = Field(default="/v1/memories", description="Base path for memory-related API endpoints.")

    def model_post_init(self, __context: Any) -> None:
        """Ensure the backend is associated with a remote memory resource."""
        if not self.memory_id:
            self.memory_id = self._resolve_memory_id()
            if not self.memory_id:
                raise DynamiqMemoryError(
                    "memory_id must be provided or resolvable via project_id for Dynamiq backend."
                )
            logger.debug("Dynamiq backend resolved memory_id=%s", self.memory_id)

    @property
    def to_dict_exclude_params(self) -> dict[str, bool]:
        """Exclude connection details from serialization."""
        return super().to_dict_exclude_params | {"connection": True}

    def to_dict(self, include_secure_params: bool = False, for_tracing: bool = False, **kwargs) -> dict[str, Any]:
        """Serialize backend configuration."""
        data = super().to_dict(include_secure_params=include_secure_params, **kwargs)
        data["connection"] = self.connection.to_dict(for_tracing=for_tracing)
        return data

    def add(self, message: Message) -> None:
        """Create a new memory item via the remote API."""
        if not self.memory_id:
            raise DynamiqMemoryError("Memory ID is not configured.")

        payload = {
            "role": message.role.value if isinstance(message.role, MessageRole) else message.role,
            "content": message.content,
            "metadata": message.metadata or {},
        }

        logger.debug("Creating remote memory item for memory_id=%s", self.memory_id)
        self._request(
            HTTPMethod.POST,
            f"{self.base_path}/{self.memory_id}/items",
            json=payload,
        )

    def get_all(self, limit: int | None = None) -> list[Message]:
        """Return all memory items from the remote store."""
        items = self._list_items(limit=limit)
        messages = self._items_to_messages(items)
        return self._apply_limit(messages, limit, newest=True)

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """Retrieve memory items optionally filtering by metadata and simple text query."""
        items = self._list_items(filters=filters)
        messages = self._items_to_messages(items)

        if query:
            lowered_query = query.lower()
            messages = [msg for msg in messages if lowered_query in (msg.content or "").lower()]

        if filters:
            messages = self._filter_messages(messages, filters)

        return self._apply_limit(messages, limit, newest=True)

    def is_empty(self) -> bool:
        """Check whether the remote memory is empty."""
        messages = self.search(limit=1)
        return len(messages) == 0

    def clear(self) -> None:
        """Not supported by the Dynamiq API."""
        raise DynamiqMemoryError("Clearing remote memories is not supported by the Dynamiq backend.")

    def _apply_limit(self, messages: list[Message], limit: int | None, newest: bool = False) -> list[Message]:
        """Apply limit to message list optionally keeping the most recent entries."""
        if limit is None or len(messages) <= limit:
            return messages

        if newest:
            return messages[-limit:]
        return messages[:limit]

    def _filter_messages(self, messages: list[Message], filters: dict[str, Any]) -> list[Message]:
        """Filter messages by metadata."""
        if not filters:
            return messages

        def matches(message: Message) -> bool:
            metadata = message.metadata or {}
            for key, value in filters.items():
                if isinstance(value, list):
                    if metadata.get(key) not in value:
                        return False
                else:
                    if metadata.get(key) != value:
                        return False
            return True

        return [msg for msg in messages if matches(msg)]

    def _list_items(self, limit: int | None = None, filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Fetch raw memory items from the API."""
        if not self.memory_id:
            raise DynamiqMemoryError("Memory ID is not configured.")

        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if filters:
            params.update(filters)

        response = self._request(
            HTTPMethod.GET,
            f"{self.base_path}/{self.memory_id}/items",
            params=params or None,
        )
        if not response:
            return []

        if isinstance(response, dict):
            for candidate in ("items", "data", "results"):
                if candidate in response and isinstance(response[candidate], list):
                    return response[candidate]
            if isinstance(response.get("item"), list):
                return response["item"]
            if isinstance(response.get("records"), list):
                return response["records"]
            if isinstance(response.get("messages"), list):
                return response["messages"]
        elif isinstance(response, list):
            return response

        logger.warning("Unexpected response shape when listing memory items: %s", response)
        return []

    def _resolve_memory_id(self) -> str | None:
        """Determine a memory_id via remote lookup if possible."""
        record = self._find_memory()
        if record and isinstance(record, dict):
            return record.get("id") or record.get("memory_id")
        return None

    def _find_memory(self) -> dict[str, Any] | None:
        """Attempt to find an existing memory for the configured project."""
        params = {"project_id": self.project_id} if self.project_id else None
        response = self._request(HTTPMethod.GET, self.base_path, params=params)

        if isinstance(response, dict):
            records = (
                response.get("items")
                or response.get("data")
                or response.get("results")
                or response.get("records")
                or response.get("memories")
            )
        elif isinstance(response, list):
            records = response
        else:
            records = None

        if not records:
            return None

        if not isinstance(records, list):
            logger.warning("Unexpected records format when listing memories: %s", records)
            return None

        first = next((record for record in records if isinstance(record, dict)), None)
        return first

    def _items_to_messages(self, items: list[dict[str, Any]]) -> list[Message]:
        """Convert raw API records into Message objects."""
        messages: list[Message] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            role_value = item.get("role") or item.get("message_role") or MessageRole.USER.value
            try:
                role = MessageRole(role_value)
            except ValueError:
                role = MessageRole.USER

            metadata = dict(item.get("metadata") or {})
            timestamp = metadata.get("timestamp") or item.get("timestamp") or item.get("created_at")
            if timestamp and "timestamp" not in metadata:
                metadata["timestamp"] = self._coerce_timestamp(timestamp)

            messages.append(Message(role=role, content=item.get("content", ""), metadata=metadata))

        messages.sort(key=lambda msg: msg.metadata.get("timestamp", 0))
        return messages

    def _coerce_timestamp(self, value: Any) -> float:
        """Convert timestamp representations to float seconds."""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.timestamp()
            except ValueError:
                pass
            try:
                return float(value)
            except ValueError:
                logger.debug("Unable to parse timestamp string '%s'", value)
        return datetime.utcnow().timestamp()

    def _request(
        self,
        method: HTTPMethod,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> Any:
        """Execute an HTTP request against the Dynamiq API."""
        base_url = (self.connection.conn_params.get("api_base") or "").rstrip("/")
        if not base_url:
            raise DynamiqMemoryError("Dynamiq API base URL is not configured.")

        if not path.startswith("/"):
            path = f"/{path}"

        url = f"{base_url}{path}"
        headers = {"Content-Type": "application/json"}
        api_key = self.connection.conn_params.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        client = self.connection.connect()
        try:
            response = client.request(
                method.value if isinstance(method, HTTPMethod) else method,
                url,
                headers=headers,
                params=params,
                json=json,
                timeout=self.timeout,
            )
        except Exception as exc:
            raise DynamiqMemoryError(f"Failed to call Dynamiq API: {exc}") from exc

        if response.status_code >= 400:
            raise DynamiqMemoryError(f"Request to Dynamiq API failed: {response.status_code} {response.text}")

        if response.status_code == 204 or not response.content:
            return None

        try:
            return response.json()
        except ValueError:
            logger.debug("Received non-JSON response from Dynamiq API for %s: %s", url, response.text)
            return response.text
