import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Iterable

# Labels / relationship types / property keys must match this pattern — the safe identifier subset
# shared by openCypher backends (Neo4j, Apache AGE, Neptune); stores reject anything else. These are
# spliced into query text (Cypher cannot parameterize structure), so they are validated, never escaped.
LABEL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
PROPERTY_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


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

        Nodes are written with ``MERGE ... ON CREATE SET``: a node's properties are fixed when it is first
        created and are NOT overwritten on re-merge, so an entity keeps its ORIGINAL name across
        re-ingestion (identity is matched on ``identity_key`` alone). Relationships use ``MERGE ... SET``,
        so an edge's properties are refreshed on every write (each edge is already scoped by its merge keys).

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
            - identity_keys: list[str] (optional) — property names to include in the MERGE pattern, so
              relationships that differ only on these keys are kept as distinct edges (e.g. per-document
              edges that each retain their own ACL/provenance). Default: merge on endpoints + type only.

        Returns a stats dict: ``nodes_created``, ``relationships_created``, ``properties_set``,
        ``records``, ``keys``. Counts are best-effort — backends whose driver does not report write
        counters (AGE, Neptune) leave them at 0; the upsert still happens.
        """
        if not self.supports_write_graph():
            raise NotImplementedError(
                f"{type(self).__name__}: write_graph is not supported for this backend. "
                "Only Neo4j (and the relational Postgres store) currently implement it."
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

    def _build_node_statements(self, nodes: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
        """Default: all nodes in ONE batched query (valid on Neo4j / Neptune; AGE overrides for single-column).

        Per node: ``MERGE (nK:Labels {id: $node_K_id}) ON CREATE SET nK += $node_K_props``, then a single
        ``RETURN n0, n1, ...``. Identity matches on ``identity_key`` only; ON CREATE keeps properties
        write-once. Returns a single ``(query, params)`` tuple.
        """
        node_lines: list[str] = []
        node_params: dict[str, Any] = {}
        return_nodes: list[str] = []
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
                f"MERGE (n{idx}{label_string} {{{identity_key}: ${param_id}}})\n"
                f"ON CREATE SET n{idx} += ${param_props}"
            )
            return_nodes.append(f"n{idx}")

        query = "\n".join(node_lines) + "\nRETURN " + ", ".join(return_nodes)
        return [(query, node_params)]

    def _build_relationship_statements(
        self, relationships: list[dict[str, Any]]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Default: one ``(query, params)`` per edge — MATCH both endpoints, MERGE the edge, SET its props.

        ``identity_keys`` are folded into the MERGE pattern so edges differing only on those keys stay
        distinct (e.g. per-source-document edges, each keeping its own ACL/provenance). ``RETURN r`` is a
        single column, so this builder is portable to AGE as-is.
        """
        statements: list[tuple[str, dict[str, Any]]] = []
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

            rel_params: dict[str, Any] = {
                "start_id": start_identity,
                "end_id": end_identity,
                "props": properties,
            }

            # Optional relationship merge-key properties: when present, include them in the MERGE pattern
            # so otherwise-identical relationships that differ on these keys stay DISTINCT edges (e.g.
            # per-source-document edges that each keep their own ACL/provenance).
            key_fragments: list[str] = []
            for kidx, key in enumerate(rel.get("identity_keys") or []):
                if key not in properties:
                    continue
                param_name = f"rkey_{idx}_{kidx}"
                key_fragments.append(f"{self._format_property_key(key)}: ${param_name}")
                rel_params[param_name] = properties[key]
            merge_pattern = (
                f"MERGE (s)-[r:{rel_type} {{{', '.join(key_fragments)}}}]->(e)"
                if key_fragments
                else f"MERGE (s)-[r:{rel_type}]->(e)"
            )

            query = (
                f"MATCH (s:{start_label} {{{start_identity_key}: $start_id}})\n"
                f"MATCH (e:{end_label} {{{end_identity_key}: $end_id}})\n"
                f"{merge_pattern}\n"
                f"SET r += $props\n"
                "RETURN r"
            )
            statements.append((query, rel_params))
        return statements

    def _tally_counts(self, totals: dict[str, int], summary: Any) -> None:
        """Accumulate write counters from a per-statement summary into ``totals``.

        Default is a no-op: backends whose driver does not surface write counters (AGE, Neptune) leave
        the counts at 0. Neo4j overrides this to read its rich ``summary.counters``.
        """
        return

    @staticmethod
    def _format_labels(labels: list[str]) -> str:
        if not labels:
            raise ValueError("At least one label is required for a node.")
        cleaned = [BaseGraphStore._format_single_label(label) for label in labels]
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
