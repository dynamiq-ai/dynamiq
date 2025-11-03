from datetime import datetime
from typing import Any

from pydantic import ConfigDict, Field, PrivateAttr

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.connections import HTTPMethod
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
    memory_id: str = Field(min_length=1, description="Identifier of the remote memory resource.")
    user_id: str | None = Field(default=None, description="Optional default user identifier for memory items.")
    session_id: str | None = Field(
        default=None,
        description="Optional default session identifier for memory items.",
    )
    timeout: float = Field(
        default=10,
        description="Timeout in seconds for API requests.",
    )
    limit: int = Field(
        default=100,
        ge=1,
        description="Default limit used when retrieving memory items.",
    )
    _base_path: str = PrivateAttr(default="/v1/memories")

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
        metadata = dict(message.metadata or {})
        effective_user_id = metadata.get("user_id") or self.user_id
        effective_session_id = metadata.get("session_id") or self.session_id

        if not effective_user_id:
            raise DynamiqMemoryError("User identifier is required to create a memory item.")
        if not effective_session_id:
            raise DynamiqMemoryError("Session identifier is required to create a memory item.")

        data_payload = {
            "role": message.role.value if isinstance(message.role, MessageRole) else message.role,
            "content": self._format_content_payload(message.content),
        }

        payload: dict[str, Any] = {
            "user_id": effective_user_id,
            "session_id": effective_session_id,
            "type": "message",
            "data": data_payload,
        }

        logger.debug("Creating remote memory item for memory_id=%s", self.memory_id)
        self._request(
            HTTPMethod.POST,
            f"{self._base_path}/{self.memory_id}/items",
            json=payload,
        )

    def get_all(self, limit: int | None = None) -> list[Message]:
        """Return all memory items from the remote store."""
        items = self._list_items(limit=limit)
        messages = self._items_to_messages(items)
        return messages

    def search(
        self, query: str | None = None, filters: dict[str, Any] | None = None, limit: int | None = None
    ) -> list[Message]:
        """Retrieve memory items optionally filtering by metadata and simple text query."""
        items = self._list_items(limit=limit, filters=filters)
        messages = self._items_to_messages(items)

        if query:
            lowered_query = query.lower()
            messages = [msg for msg in messages if lowered_query in (msg.content or "").lower()]

        if filters:
            messages = self._filter_messages(messages, filters)

        return messages

    def is_empty(self) -> bool:
        """Check whether the remote memory is empty."""
        messages = self.search(limit=1)
        return len(messages) == 0

    def clear(self) -> None:
        """Not supported by the Dynamiq API."""
        raise DynamiqMemoryError("Clearing remote memories is not supported by the Dynamiq backend.")

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
        filters = filters.copy() if filters else {}
        effective_user_id = filters.pop("user_id", None) or self.user_id
        effective_session_id = filters.pop("session_id", None) or self.session_id

        if not effective_user_id:
            raise DynamiqMemoryError("User identifier is required to list memory items.")
        if not effective_session_id:
            raise DynamiqMemoryError("Session identifier is required to list memory items.")

        params: dict[str, Any] = {
            "user_id": effective_user_id,
            "session_id": effective_session_id,
            "page_size": limit if limit is not None else self.limit,
            "sort": "-created_at",
        }
        if filters:
            params.update(filters)

        response = self._request(
            HTTPMethod.GET,
            f"{self._base_path}/{self.memory_id}/items",
            params=params or None,
        )
        if not response:
            return []

        if isinstance(response, dict):
            data = response.get("data")
            if isinstance(data, list):
                return data

        logger.warning("Unexpected response shape when listing memory items: %s", response)
        return []

    def _format_content_payload(self, content: Any) -> list[dict[str, Any]]:
        """Normalize message content into the Dynamiq API payload structure."""
        if isinstance(content, list):
            return content

        if isinstance(content, str):
            text = content
        else:
            text = str(content) if content is not None else ""

        return [{"type": "text", "text": text}]

    def _items_to_messages(self, items: list[dict[str, Any]]) -> list[Message]:
        """Convert raw API records into Message objects."""
        messages: list[Message] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            data = item.get("data") if isinstance(item.get("data"), dict) else {}
            role_value = data.get("role") or MessageRole.USER.value
            try:
                role = MessageRole(role_value)
            except ValueError:
                role = MessageRole.USER

            metadata = dict(data.get("metadata") or {})
            item_metadata = item.get("metadata")
            if isinstance(item_metadata, dict):
                metadata.update({k: v for k, v in item_metadata.items() if k not in metadata})

            metadata["user_id"] = item.get("user_id")
            metadata["session_id"] = item.get("session_id")

            item_type = item.get("type")
            if item_type and "type" not in metadata:
                metadata["type"] = item_type

            created_at = item.get("created_at") or data.get("created_at")
            updated_at = item.get("updated_at") or data.get("updated_at")
            if created_at and "created_at" not in metadata:
                metadata["created_at"] = created_at
            if updated_at and "updated_at" not in metadata:
                metadata["updated_at"] = updated_at

            if created_at and "timestamp" not in metadata:
                metadata["timestamp"] = self._coerce_timestamp(created_at)

            content = self._extract_content_text(data) or item.get("content", "")
            messages.append(Message(role=role, content=content, metadata=metadata))

        messages.sort(key=lambda msg: (msg.metadata or {}).get("timestamp") or 0)
        return messages

    def _extract_content_text(self, data: dict[str, Any]) -> str:
        """Extract textual content from Dynamiq API message payloads."""
        content = data.get("content")
        if isinstance(content, list):
            text_chunks = [
                str(chunk.get("text"))
                for chunk in content
                if isinstance(chunk, dict) and chunk.get("type") == "text" and chunk.get("text")
            ]
            return "\n\n".join(text_chunks).strip()

        if isinstance(content, str):
            return content

        if content is not None:
            return str(content)

        return ""

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
        conn_params = self.connection.conn_params
        base_url = (conn_params.get("api_base") or "").rstrip("/")
        if not base_url:
            raise DynamiqMemoryError("Dynamiq API base URL is not configured.")

        url = f"{base_url}/{path.lstrip('/')}"
        headers = {"Content-Type": "application/json"}
        conn_headers = conn_params.get("headers")
        if isinstance(conn_headers, dict):
            headers.update(conn_headers)

        client = self.connection.connect()
        verb = method.value if isinstance(method, HTTPMethod) else method
        try:
            response = client.request(
                verb,
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
