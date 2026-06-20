from typing import Any, ClassVar, Iterable

from neo4j import RoutingControl
from neo4j.exceptions import Neo4jError

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.utils.logger import logger


class Neo4jGraphStore(BaseGraphStore):
    """
    Lightweight wrapper around the Neo4j Python driver.

    Provides helpers to run Cypher and to upsert simple node/relationship payloads. The graph-upsert
    path itself lives in :class:`BaseGraphStore`; this store only flips on writes and reads back native
    Neo4j counters.
    """

    _writes_graph: ClassVar[bool] = True

    def __init__(
        self,
        connection: Neo4jConnection | None = None,
        client: Any | None = None,
        database: str | None = None,
    ):
        if client is None and connection is None:
            raise ValueError("Either 'connection' or 'client' must be provided.")

        self.connection = connection
        self.client = client or connection.connect()
        self.database = database or (connection.database if connection else None)

    def run_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        database: str | None = None,
        routing: str | RoutingControl | None = None,
        result_transformer: Any | None = None,
    ) -> tuple[Any, Any, list[str]]:
        """
        Execute a Cypher query with optional parameters and transformers.

        Args:
            query: Cypher string.
            parameters: Query parameters passed as a dict.
            database: Target database; defaults to connection/database on store.
            routing: Optional routing flag ('r' or 'w').
            result_transformer: Optional transformer passed to `result_transformer_`.
        """
        params = parameters or {}
        execute_kwargs = {}
        target_db = database or self.database
        if target_db:
            execute_kwargs["database_"] = target_db
        if routing:
            execute_kwargs["routing_"] = self._normalize_routing(routing)
        if result_transformer:

            def _transform_with_metadata(result: Any) -> tuple[Any, Any, list[str]]:
                if callable(result_transformer):
                    transformed = result_transformer(result)
                elif hasattr(result_transformer, "__get__"):
                    transformed = result_transformer.__get__(result, type(result))
                else:
                    transformed = result_transformer
                summary = result.consume()
                keys = result.keys()
                return transformed, summary, keys

            execute_kwargs["result_transformer_"] = _transform_with_metadata

        try:
            return self.client.execute_query(query, parameters_=params, **execute_kwargs)
        except Neo4jError as exc:
            logger.error(f"Neo4j query failed: {exc.code} - {exc.message}")
            raise

    def update_client(self, client: Any) -> None:
        self.client = client

    @staticmethod
    def format_records(records: Iterable[Any]) -> list[dict[str, Any]]:
        """Convert Neo4j Record objects to plain dicts."""
        return [record.data() for record in records]

    def introspect_schema(self, *, include_properties: bool, **kwargs: Any) -> dict[str, Any]:
        database = kwargs.get("database")
        labels_records, _, _ = self.run_cypher(
            "CALL db.labels() YIELD label RETURN label ORDER BY label",
            database=database,
        )
        reltype_records, _, _ = self.run_cypher(
            "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType",
            database=database,
        )
        labels = [r["label"] for r in labels_records]
        rel_types = [r["relationshipType"] for r in reltype_records]

        node_props: list[dict[str, Any]] = []
        rel_props: list[dict[str, Any]] = []

        if include_properties:
            try:
                node_props_records, _, _ = self.run_cypher(
                    "CALL db.schema.nodeTypeProperties()",
                    database=database,
                )
                node_props = [r.data() for r in node_props_records]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Node property introspection failed: {exc}")

            try:
                rel_props_records, _, _ = self.run_cypher(
                    "CALL db.schema.relTypeProperties()",
                    database=database,
                )
                rel_props = [r.data() for r in rel_props_records]
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Relationship property introspection failed: {exc}")

        return {
            "labels": labels,
            "relationship_types": rel_types,
            "node_properties": node_props,
            "relationship_properties": rel_props,
        }

    def supports_graph_result(self) -> bool:
        return True

    def _tally_counts(self, totals: dict[str, int], summary: Any) -> None:
        """Read Neo4j's native write counters off each statement's summary into ``totals``.

        Adding all three on every statement equals the per-phase tally: the node statement reports
        ``relationships_created = 0`` (no edge pattern), and the edge statement reports
        ``nodes_created = 0`` (endpoints are MATCHed, only the edge is MERGEd).
        """
        counters = summary.counters
        totals["nodes_created"] += counters.nodes_created
        totals["relationships_created"] += counters.relationships_created
        totals["properties_set"] += counters.properties_set

    def close(self: "Neo4jGraphStore") -> None:
        if self.client:
            self.client.close()

    @staticmethod
    def _normalize_routing(routing: str | RoutingControl) -> RoutingControl:
        if isinstance(routing, RoutingControl):
            return routing
        routing_value = routing.strip().lower()
        if routing_value in {"r", "read"}:
            return RoutingControl.READ
        if routing_value in {"w", "write"}:
            return RoutingControl.WRITE
        raise ValueError("routing must be 'r'/'read' or 'w'/'write' when provided.")
