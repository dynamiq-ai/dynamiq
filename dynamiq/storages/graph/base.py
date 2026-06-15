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

    def write_graph(
        self,
        *,
        nodes: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Upsert nodes and relationships described in the provider-neutral graph payload format.

        Nodes must include:
            - labels: list[str]
            - properties: dict (must contain the identity_key)
            - identity_key: str (defaults to 'id' if missing)

        Relationships must include:
            - start_label, end_label: str
            - start_identity_key, end_identity_key: str
            - start_identity, end_identity: Any
            - type: str
            - properties: dict (optional)

        Returns a stats dict: ``nodes_created``, ``relationships_created``, ``properties_set``,
        ``records``, ``keys``. Backends that do not support writing keep this default implementation.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support write_graph.")

    def supports_write_graph(self) -> bool:
        """Whether this backend implements :meth:`write_graph`."""
        return type(self).write_graph is not BaseGraphStore.write_graph

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
