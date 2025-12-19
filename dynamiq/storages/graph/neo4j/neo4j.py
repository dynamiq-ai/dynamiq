import re
from typing import Any, Iterable

from neo4j import RoutingControl
from neo4j.exceptions import Neo4jError

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.utils.logger import logger

LABEL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROPERTY_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class Neo4jGraphStore:
    """
    Lightweight wrapper around the Neo4j Python driver.

    Provides helpers to run Cypher and to upsert simple node/relationship payloads.
    """

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
    ) -> tuple[list[Any], Any, list[str]]:
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
            execute_kwargs["result_transformer_"] = result_transformer

        try:
            return self.client.execute_query(query, parameters_=params, **execute_kwargs)
        except Neo4jError as exc:
            logger.error(f"Neo4j query failed: {exc.code} - {exc.message}")
            raise

    @staticmethod
    def format_records(records: Iterable[Any]) -> list[dict[str, Any]]:
        """Convert Neo4j Record objects to plain dicts."""
        return [record.data() for record in records]

    def write_graph(
        self,
        *,
        nodes: list[dict[str, Any]],
        relationships: list[dict[str, Any]],
        database: str | None = None,
    ) -> dict[str, Any]:
        """
        Upsert nodes and relationships using MERGE + SET.

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
        """
        if not nodes and not relationships:
            raise ValueError("At least one node or relationship must be provided.")

        total_nodes_created = 0
        total_properties_set = 0
        total_relationships_created = 0
        last_records: list[dict[str, Any]] = []
        last_keys: list[str] = []

        if nodes:
            node_lines: list[str] = []
            node_params: dict[str, Any] = {}
            for idx, node in enumerate(nodes):
                labels = node.get("labels") or []
                identity_key = self._format_property_key(node.get("identity_key") or "id")
                properties = node.get("properties") or {}
                if identity_key not in properties:
                    raise ValueError(f"Node {idx} is missing identity key '{identity_key}' in properties.")

                label_string = self._format_labels(labels)
                param_props = f"node_{idx}_props"
                param_id = f"node_{idx}_id"
                node_params[param_props] = properties
                node_params[param_id] = properties[identity_key]

                node_lines.append(
                    f"MERGE (n{idx}{label_string} {{{identity_key}: ${param_id}}})\n" f"SET n{idx} += ${param_props}"
                )

            node_query = "\n".join(node_lines)
            node_records, node_summary, node_keys = self.run_cypher(node_query, node_params, database=database)
            total_nodes_created += node_summary.counters.nodes_created
            total_properties_set += node_summary.counters.properties_set
            last_records = self.format_records(node_records)
            last_keys = node_keys

        if relationships:
            last_records = []
            last_keys = []
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

                rel_query = (
                    f"MATCH (s:{start_label} {{{start_identity_key}: $start_id}})\n"
                    f"MATCH (e:{end_label} {{{end_identity_key}: $end_id}})\n"
                    f"MERGE (s)-[r:{rel_type}]->(e)\n"
                    f"SET r += $props"
                )
                rel_params = {
                    "start_id": start_identity,
                    "end_id": end_identity,
                    "props": properties,
                }
                rel_records, rel_summary, rel_keys = self.run_cypher(rel_query, rel_params, database=database)
                total_relationships_created += rel_summary.counters.relationships_created
                total_properties_set += rel_summary.counters.properties_set
                last_records = self.format_records(rel_records)
                last_keys = rel_keys

        return {
            "nodes_created": total_nodes_created,
            "properties_set": total_properties_set,
            "relationships_created": total_relationships_created,
            "records": last_records,
            "keys": last_keys,
        }

    @staticmethod
    def _format_labels(labels: list[str]) -> str:
        if not labels:
            raise ValueError("At least one label is required for a node.")
        cleaned = [Neo4jGraphStore._format_single_label(label) for label in labels]
        return ":" + ":".join(cleaned)

    @staticmethod
    def _format_single_label(label: str) -> str:
        if not LABEL_PATTERN.match(label):
            raise ValueError(f"Invalid Neo4j label: '{label}'")
        return label

    @staticmethod
    def _format_relationship_type(rel_type: str) -> str:
        if not LABEL_PATTERN.match(rel_type):
            raise ValueError(f"Invalid Neo4j relationship type: '{rel_type}'")
        return rel_type

    @staticmethod
    def _format_property_key(key: str) -> str:
        if not PROPERTY_KEY_PATTERN.match(key):
            raise ValueError(f"Invalid Neo4j property key: '{key}'")
        return key

    def close(self):
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
