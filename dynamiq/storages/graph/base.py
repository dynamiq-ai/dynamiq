from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable


class BaseGraphStore(ABC):
    """Base interface for graph backends that support Cypher-like queries.

    The graph-upsert path (:meth:`write_graph`) is shared here: it builds provider-neutral openCypher
    MERGE/SET statements and runs them through each backend's :meth:`run_cypher`. Backends customize only
    the genuinely-divergent bits via hooks (:meth:`_tally_counts`, the statement builders) and opt in by
    setting :attr:`_writes_graph`.
    """

    # Backends flip this to True once their write_graph path is implemented AND validated.
    _writes_graph: ClassVar[bool] = False

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

    def supports_write_graph(self) -> bool:
        """Whether this backend implements (and has enabled) :meth:`write_graph`."""
        return type(self)._writes_graph

    def write_graph(
        self,
        *,
        nodes: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        database: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Upsert nodes and relationships described in the provider-neutral graph payload format.

        No default implementation — each writing backend defines its OWN write path (Neo4j and Neptune
        build ``MERGE`` statements and run them; the relational Postgres store writes its own tables).
        Backends that do not support writing (e.g. AGE, ``_writes_graph = False``) inherit this and raise.

        Implementations accept these payloads and return a stats dict (``nodes_created``,
        ``relationships_created``, ``properties_set``, ``records``, ``keys``):
            Nodes — ``labels: list[str]``, ``properties: dict`` (must contain the identity key),
                ``identity_key: str`` (defaults to 'id'). Written ``MERGE ... ON CREATE SET`` (write-once).
            Relationships — ``start_label/end_label``, ``start_identity_key/end_identity_key``,
                ``start_identity/end_identity``, ``type``, optional ``properties`` and ``identity_keys``
                (folded into the MERGE pattern so edges differing only on those keys stay distinct).
                Written ``MERGE ... SET`` (props refreshed each write).
        """
        raise NotImplementedError(
            f"{type(self).__name__}: write_graph is not implemented for this backend."
        )

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
