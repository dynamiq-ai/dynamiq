import re
from typing import Any, ClassVar, Iterable

from neo4j import RoutingControl
from neo4j.exceptions import Neo4jError

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.utils.logger import logger

# Labels / relationship types / property keys are spliced into Cypher text (structure can't be
# parameterized), so they must match this safe-identifier pattern; anything else is rejected.
LABEL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROPERTY_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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

    def write_graph(
        self,
        *,
        nodes: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        database: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Neo4j-owned upsert path: build the (bulk ``UNWIND``) statements and run them in order.

        Self-contained on the Neo4j store (rather than relying on ``BaseGraphStore.write_graph``) so the
        whole Neo4j write path — node/edge builders, counter tally, and this orchestration — lives here.
        The base ``write_graph`` remains the default for the other openCypher backends (Neptune, AGE).

        Nodes are written with ``MERGE ... ON CREATE SET`` (properties write-once, identity matched on
        ``identity_key``); relationships with ``MERGE ... SET`` (edge props refreshed each write). Returns
        a stats dict: ``nodes_created``, ``relationships_created``, ``properties_set``, ``records``, ``keys``.
        """
        if not self.supports_write_graph():
            raise NotImplementedError(
                f"{type(self).__name__}: write_graph is not supported for this backend."
            )
        if not nodes and not relationships:
            raise ValueError("At least one node or relationship must be provided.")

        totals = {"nodes_created": 0, "relationships_created": 0, "properties_set": 0}
        all_records: list[dict[str, Any]] = []
        last_keys: list[str] = []

        statements: list[tuple[str, dict[str, Any]]] = []
        if nodes:
            statements.extend(self._build_node_statements(nodes))
        if relationships:
            statements.extend(self._build_relationship_statements(relationships))

        for query, params in statements:
            records, summary, keys = self.run_cypher(query, params, database=database)
            self._tally_counts(totals, summary)
            all_records.extend(self.format_records(records))
            if keys:
                last_keys = keys

        return {**totals, "records": all_records, "keys": last_keys}

    def delete_documents(
        self,
        document_ids: list[str],
        *,
        doc_scoped_labels: list[str] | None = None,
        provenance_key: str = "source_doc_id",
        database: str | None = None,
        **kwargs: Any,
    ) -> dict[str, int]:
        """Delete the given documents' edges (and their document-scoped nodes), leaving entities intact.

        ``provenance_key`` is the edge property the ids match against — ``source_doc_id`` (default, the
        ingestion chunk id) or any other flattened document-metadata key, e.g. ``file_id`` to delete
        everything extracted from any chunk of a knowledgebase file. Validated as an identifier (it is
        spliced into the query text; the id VALUES are always bound parameters).

        For each label in ``doc_scoped_labels`` (e.g. ``AttributeValue``), the node is the end of an edge
        stamped with one of these documents' provenance and belongs to exactly that document, so a
        ``DETACH DELETE`` removes the node and that edge together. Any remaining per-document edges
        (entity-to-entity) are then deleted; identity (entity) nodes are deliberately never matched.
        """
        if not document_ids:
            return {"nodes_deleted": 0, "relationships_deleted": 0}
        if not PROPERTY_KEY_PATTERN.match(provenance_key):
            raise ValueError(f"Invalid provenance key for delete: {provenance_key!r}")

        totals = {"nodes_deleted": 0, "relationships_deleted": 0}
        params = {"doc_ids": list(document_ids)}

        for label in doc_scoped_labels or []:
            if not LABEL_PATTERN.match(label):
                raise ValueError(f"Invalid document-scoped label for delete: {label!r}")
            query = "MATCH ()-[r]->(v:`" + label + "`)\n" f"WHERE r.{provenance_key} IN $doc_ids\n" "DETACH DELETE v"
            _, summary, _ = self.run_cypher(query, params, database=database)
            totals["nodes_deleted"] += summary.counters.nodes_deleted
            totals["relationships_deleted"] += summary.counters.relationships_deleted

        _, summary, _ = self.run_cypher(
            f"MATCH ()-[r]->() WHERE r.{provenance_key} IN $doc_ids DELETE r",
            params,
            database=database,
        )
        totals["relationships_deleted"] += summary.counters.relationships_deleted
        return totals

    def _build_node_statements(self, nodes: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        """Neo4j override: BULK node upsert via ``UNWIND``, grouped by label-set.

        The base builder inlines one ``MERGE`` clause per node into a single statement; that gives the
        planner an N-operator plan that does not scale (≈19 ms/node measured, and large payloads overrun
        the connection timeout). Here nodes sharing the same labels + identity key are written by one
        ``UNWIND $rows`` statement — Neo4j compiles a single plan and streams the rows (≈0.1 ms/node).
        Per-node semantics are identical: ``MERGE (n:Labels {id}) ON CREATE SET n += props`` (identity on
        ``identity_key`` only, properties write-once). One ``(query, params)`` per label-group.
        """
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        order: list[tuple[str, str]] = []
        for idx, node in enumerate(nodes):
            labels = node.get("labels") or []
            identity_key = self._format_property_key(node.get("identity_key") or "id")
            properties = node.get("properties") or {}
            if identity_key not in properties:
                raise ValueError(f"Node {idx} is missing identity key '{identity_key}' in properties.")

            signature = (self._format_labels(labels), identity_key)
            if signature not in groups:
                groups[signature] = []
                order.append(signature)
            groups[signature].append({"id": properties[identity_key], "props": properties})

        statements: list[tuple[str, dict[str, Any]]] = []
        for signature in order:
            label_string, identity_key = signature
            query = (
                "UNWIND $rows AS row\n"
                f"MERGE (n{label_string} {{{identity_key}: row.id}})\n"
                "ON CREATE SET n += row.props\n"
                "RETURN n"
            )
            statements.append((query, {"rows": groups[signature]}))
        return statements

    def _build_relationship_statements(
        self, relationships: list[dict[str, Any]]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Neo4j override: BULK edge upsert via ``UNWIND``, grouped by structure.

        The base builder emits one statement per edge (one network round-trip each). Here edges sharing
        the same shape — relationship type, endpoint labels, identity keys, and the set of
        ``identity_keys`` folded into the MERGE pattern — are written by one ``UNWIND $rows`` statement.
        Per-edge semantics are identical: ``MATCH`` both endpoints by id, ``MERGE`` the edge (with
        ``identity_keys`` in the pattern so otherwise-identical edges stay distinct), ``SET r += props``.
        Each row is independent, so a missing endpoint drops only that edge. One ``(query, params)`` per
        structural group.
        """
        groups: dict[tuple, list[dict[str, Any]]] = {}
        order: list[tuple] = []
        for idx, rel in enumerate(relationships):
            rel_type = self._format_relationship_type(rel.get("type") or "")
            start_label = self._format_single_label(rel.get("start_label") or "")
            end_label = self._format_single_label(rel.get("end_label") or "")
            start_identity_key = self._format_property_key(rel.get("start_identity_key") or "id")
            end_identity_key = self._format_property_key(rel.get("end_identity_key") or "id")
            start_identity = rel.get("start_identity")
            end_identity = rel.get("end_identity")
            properties = rel.get("properties") or {}

            if start_identity is None or end_identity is None:
                raise ValueError(f"Relationship {idx} missing start or end identity value.")

            present_raw = [k for k in (rel.get("identity_keys") or []) if k in properties]
            present_keys = tuple(self._format_property_key(k) for k in present_raw)
            signature = (
                rel_type, start_label, end_label, start_identity_key, end_identity_key, present_keys
            )
            row: dict[str, Any] = {
                "start_id": start_identity, "end_id": end_identity, "props": properties
            }
            for fkey, raw in zip(present_keys, present_raw):
                row[fkey] = properties[raw]

            if signature not in groups:
                groups[signature] = []
                order.append(signature)
            groups[signature].append(row)

        statements: list[tuple[str, dict[str, Any]]] = []
        for signature in order:
            rel_type, start_label, end_label, start_identity_key, end_identity_key, present_keys = signature
            if present_keys:
                key_fragments = ", ".join(f"{key}: row.{key}" for key in present_keys)
                merge_pattern = f"MERGE (s)-[r:{rel_type} {{{key_fragments}}}]->(e)"
            else:
                merge_pattern = f"MERGE (s)-[r:{rel_type}]->(e)"

            query = (
                "UNWIND $rows AS row\n"
                f"MATCH (s:{start_label} {{{start_identity_key}: row.start_id}})\n"
                f"MATCH (e:{end_label} {{{end_identity_key}: row.end_id}})\n"
                f"{merge_pattern}\n"
                "SET r += row.props\n"
                "RETURN r"
            )
            statements.append((query, {"rows": groups[signature]}))
        return statements

    @staticmethod
    def _format_labels(labels: list[str]) -> str:
        if not labels:
            raise ValueError("At least one label is required for a node.")
        cleaned = [Neo4jGraphStore._format_single_label(label) for label in labels]
        return ":" + ":".join(cleaned)

    @staticmethod
    def _format_single_label(label: str) -> str:
        if not LABEL_PATTERN.match(label):
            raise ValueError(f"Invalid graph label: '{label}'")
        return label

    @staticmethod
    def _format_relationship_type(rel_type: str) -> str:
        if not LABEL_PATTERN.match(rel_type):
            raise ValueError(f"Invalid graph relationship type: '{rel_type}'")
        return rel_type

    @staticmethod
    def _format_property_key(key: str) -> str:
        if not PROPERTY_KEY_PATTERN.match(key):
            raise ValueError(f"Invalid graph property key: '{key}'")
        return key

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
