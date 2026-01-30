import json
import re
from typing import Any

from psycopg import sql

from dynamiq.connections import ApacheAGE
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.utils.logger import logger

LABEL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ApacheAgeGraphStore(BaseGraphStore):
    """Wrapper for Apache AGE openCypher execution via PostgreSQL."""

    def __init__(
        self,
        connection: ApacheAGE | None = None,
        client: Any | None = None,
        graph_name: str | None = None,
        create_graph_if_not_exists: bool = False,
    ):
        if client is None and connection is None:
            raise ValueError("Either 'connection' or 'client' must be provided.")

        self.connection = connection
        self.client = client or connection.connect()
        self.graph_name = graph_name
        self.create_graph_if_not_exists = create_graph_if_not_exists

        if not self.graph_name:
            raise ValueError("graph_name must be provided for Apache AGE.")

        self._age_loaded = False
        self._ensure_age_loaded()
        if self.create_graph_if_not_exists:
            self._ensure_graph_exists()

    def run_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], dict[str, Any], list[str]]:
        self._ensure_age_loaded()
        params = parameters or {}
        safe_graph = self._validate_label(self.graph_name)
        sql = "SELECT ag_catalog.agtype_to_json(result) AS result " "FROM cypher(%s, %s, %s::agtype) AS (result agtype)"
        with self.client.cursor() as cursor:
            cursor.execute(sql, (safe_graph, query, json.dumps(params)))
            rows = cursor.fetchall()

        records: list[Any] = []
        for row in rows:
            value = row["result"] if isinstance(row, dict) else row[0]
            records.append(value)

        summary = {"query": query, "counters": {}, "result_available_after": None}
        return records, summary, []

    def update_client(self, client: Any) -> None:
        self.client = client
        self._age_loaded = False
        self._ensure_age_loaded()
        if self.create_graph_if_not_exists:
            self._ensure_graph_exists()

    def introspect_schema(self, *, include_properties: bool, **kwargs: Any) -> dict[str, Any]:
        labels = self._filter_internal(self._get_labels(kind="v"))
        rel_types = self._filter_internal(self._get_labels(kind="e"))

        node_props: list[dict[str, Any]] = []
        rel_props: list[dict[str, Any]] = []
        if include_properties:
            node_props = self._sample_node_properties(labels)
            rel_props = self._sample_edge_properties(rel_types)

        return {
            "labels": labels,
            "relationship_types": rel_types,
            "node_properties": node_props,
            "relationship_properties": rel_props,
            "relationships": self._sample_triples(rel_types),
        }

    def _ensure_age_loaded(self) -> None:
        if self._age_loaded:
            return
        try:
            with self.client.cursor() as cursor:
                cursor.execute("LOAD 'age';")
                cursor.execute('SET search_path = ag_catalog, "$user", public;')
            self._age_loaded = True
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Failed to load Apache AGE extension: {exc}")
            raise

    def _ensure_graph_exists(self) -> None:
        with self.client.cursor() as cursor:
            cursor.execute(
                "SELECT 1 FROM ag_catalog.ag_graph WHERE name = %s",
                (self.graph_name,),
            )
            exists = cursor.fetchone()
            if exists:
                return
            cursor.execute("SELECT * FROM ag_catalog.create_graph(%s)", (self.graph_name,))

    def _get_labels(self, *, kind: str) -> list[str]:
        graph_id_column = self._get_graph_id_column()
        with self.client.cursor() as cursor:
            query = sql.SQL(
                """
                SELECT l.name AS label
                FROM ag_catalog.ag_label l
                JOIN ag_catalog.ag_graph g ON g.{graph_id_column} = l.graph
                WHERE g.name = %s AND l.kind = %s
                ORDER BY l.name
                """
            ).format(graph_id_column=sql.Identifier(graph_id_column))
            cursor.execute(query, (self.graph_name, kind))
            rows = cursor.fetchall()
        labels = [row["label"] if isinstance(row, dict) else row[0] for row in rows]
        return labels

    def _sample_node_properties(self, labels: list[str]) -> list[dict[str, Any]]:
        node_properties: list[dict[str, Any]] = []
        type_mapping = {
            "str": "STRING",
            "float": "DOUBLE",
            "int": "INTEGER",
            "list": "LIST",
            "dict": "MAP",
            "bool": "BOOLEAN",
        }
        for label in labels:
            safe_label = self._validate_label(label)
            query = f"MATCH (n:`{safe_label}`) RETURN properties(n) AS result LIMIT 100"
            records, _, _ = self.run_cypher(query)
            seen: set[tuple[str, str]] = set()
            for record in records:
                props = record or {}
                for key, value in props.items():
                    seen.add((key, type_mapping.get(type(value).__name__, "STRING")))
            node_properties.append({"labels": label, "properties": [{"property": k, "type": v} for k, v in seen]})
        return node_properties

    def _sample_edge_properties(self, labels: list[str]) -> list[dict[str, Any]]:
        edge_properties: list[dict[str, Any]] = []
        type_mapping = {
            "str": "STRING",
            "float": "DOUBLE",
            "int": "INTEGER",
            "list": "LIST",
            "dict": "MAP",
            "bool": "BOOLEAN",
        }
        for label in labels:
            safe_label = self._validate_label(label)
            query = f"MATCH ()-[e:`{safe_label}`]->() RETURN properties(e) AS result LIMIT 100"
            records, _, _ = self.run_cypher(query)
            seen: set[tuple[str, str]] = set()
            for record in records:
                props = record or {}
                for key, value in props.items():
                    seen.add((key, type_mapping.get(type(value).__name__, "STRING")))
            edge_properties.append({"type": label, "properties": [{"property": k, "type": v} for k, v in seen]})
        return edge_properties

    def _sample_triples(self, edge_labels: list[str]) -> list[str]:
        triple_template = "(:`{a}`)-[:`{e}`]->(:`{b}`)"
        triples: list[str] = []
        for label in edge_labels:
            safe_label = self._validate_label(label)
            query = (
                f"MATCH (a)-[e:`{safe_label}`]->(b) "
                "RETURN {from: labels(a), edge: type(e), to: labels(b)} AS result LIMIT 10"
            )
            records, _, _ = self.run_cypher(query)
            for record in records:
                if not isinstance(record, dict):
                    continue
                from_labels = record.get("from", [])
                to_labels = record.get("to", [])
                if not from_labels or not to_labels:
                    continue
                triples.append(triple_template.format(a=from_labels[0], e=record.get("edge"), b=to_labels[0]))
        return triples

    @staticmethod
    def _filter_internal(items: list[str]) -> list[str]:
        return [item for item in items if not item.startswith("_ag_")]

    @staticmethod
    def _validate_label(label: str) -> str:
        if not LABEL_PATTERN.match(label):
            raise ValueError(f"Invalid Apache AGE label: '{label}'")
        return label

    @staticmethod
    def _to_dollar_quoted(query: str) -> str:
        tag_base = "age"
        tag = tag_base
        counter = 0
        while f"${tag}$" in query:
            counter += 1
            tag = f"{tag_base}{counter}"
        return f"${tag}$\n{query}\n${tag}$"

    def _get_graph_id_column(self) -> str:
        with self.client.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = 'ag_catalog' AND table_name = 'ag_graph'
                """
            )
            rows = cursor.fetchall()
        columns = {row["column_name"] if isinstance(row, dict) else row[0] for row in rows}
        for candidate in ("graphid", "id", "oid"):
            if candidate in columns:
                return candidate
        if columns:
            fallback = sorted(columns)[0]
            logger.warning(
                "Falling back to graph id column '%s' from ag_catalog.ag_graph; available columns: %s",
                fallback,
                sorted(columns),
            )
            return fallback
        raise RuntimeError("Unable to introspect ag_catalog.ag_graph columns for Apache AGE.")
