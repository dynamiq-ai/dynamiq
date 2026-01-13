from abc import ABC, abstractmethod
from typing import Any, Iterable


class BaseGraphStore(ABC):
    """Base interface for graph backends that support Cypher-like queries."""

    @abstractmethod
    def run_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, Any, list[str]]:
        """Execute a Cypher query and return (records, summary, keys)."""
        raise NotImplementedError

    @abstractmethod
    def introspect_schema(self, *, include_properties: bool, **kwargs: Any) -> dict[str, Any]:
        """Return schema details: labels, relationship_types, node_properties, relationship_properties."""
        raise NotImplementedError

    def supports_graph_result(self) -> bool:
        """Whether the backend can return native graph objects."""
        return False

    def update_client(self, client: Any) -> None:
        """Update the underlying client reference if the connection is reinitialized."""
        if hasattr(self, "client"):
            self.client = client

    @staticmethod
    def format_records(records: Iterable[Any]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        for record in records:
            if isinstance(record, dict):
                formatted.append(record)
            elif hasattr(record, "data") and callable(record.data):
                formatted.append(record.data())
            else:
                formatted.append({"value": record})
        return formatted
