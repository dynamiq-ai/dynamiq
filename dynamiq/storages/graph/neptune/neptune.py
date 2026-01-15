import json
import re
from typing import Any

from dynamiq.connections import AWSNeptune as AWSNeptuneConnection
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.utils.logger import logger

LABEL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class NeptuneGraphStore(BaseGraphStore):
    """Wrapper for Amazon Neptune openCypher execution over HTTP."""

    def __init__(
        self,
        connection: AWSNeptuneConnection | None = None,
        client: Any | None = None,
        endpoint: str | None = None,
        verify_ssl: bool | None = None,
        timeout: int | None = None,
    ) -> None:
        if client is None and connection is None:
            raise ValueError("Either 'connection' or 'client' must be provided.")

        self.connection = connection
        self.client = client or connection.connect()
        self.endpoint = endpoint or (connection.endpoint if connection else None)
        self.verify_ssl = verify_ssl if verify_ssl is not None else (connection.verify_ssl if connection else True)
        self.timeout = timeout if timeout is not None else (connection.timeout if connection else 30)

        if not self.endpoint:
            raise ValueError("endpoint must be provided for Neptune.")

    def run_cypher(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], dict[str, Any], list[str]]:
        params = parameters or {}
        payload = {"query": query}
        if params:
            payload["parameters"] = params
        try:
            response = self.client.post(
                self.endpoint,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                verify=self.verify_ssl,
                timeout=self.timeout,
            )
            response.raise_for_status()
            records = response.json().get("results", [])
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Neptune query failed: {exc}")
            raise

        summary = {"query": query, "counters": {}, "result_available_after": None}
        return records, summary, []

    def introspect_schema(self, *, include_properties: bool, **kwargs: Any) -> dict[str, Any]:
        node_labels = self._sample_labels()
        edge_labels = self._sample_relationship_types()

        node_properties: list[dict[str, Any]] = []
        edge_properties: list[dict[str, Any]] = []
        relationships: list[str] = []

        if include_properties:
            node_properties = self._sample_node_properties(node_labels)
            edge_properties = self._sample_edge_properties(edge_labels)

        relationships = self._sample_triples(edge_labels)

        return {
            "labels": node_labels,
            "relationship_types": edge_labels,
            "node_properties": node_properties,
            "relationship_properties": edge_properties,
            "relationships": relationships,
        }

    def _sample_labels(self) -> list[str]:
        query = "MATCH (n) RETURN DISTINCT labels(n) AS labels LIMIT 200"
        records, _, _ = self.run_cypher(query)
        labels: set[str] = set()
        for record in records:
            row = record.get("labels", []) if isinstance(record, dict) else []
            for label in row:
                if isinstance(label, str):
                    labels.add(label)
        return sorted(labels)

    def _sample_relationship_types(self) -> list[str]:
        query = "MATCH ()-[r]->() RETURN DISTINCT type(r) AS type LIMIT 200"
        records, _, _ = self.run_cypher(query)
        rel_types = sorted(
            {record.get("type") for record in records if isinstance(record, dict) and record.get("type")}
        )
        return rel_types

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
            query = f"MATCH (a:`{safe_label}`) RETURN properties(a) AS props LIMIT 100"
            records, _, _ = self.run_cypher(query)
            seen: set[tuple[str, str]] = set()
            for record in records:
                props = record.get("props", {}) if isinstance(record, dict) else {}
                for key, value in props.items():
                    seen.add((key, type_mapping.get(type(value).__name__, "STRING")))
            node_properties.append(
                {
                    "labels": label,
                    "properties": [{"property": key, "type": value} for key, value in seen],
                }
            )
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
            query = f"MATCH ()-[e:`{safe_label}`]->() RETURN properties(e) AS props LIMIT 100"
            records, _, _ = self.run_cypher(query)
            seen: set[tuple[str, str]] = set()
            for record in records:
                props = record.get("props", {}) if isinstance(record, dict) else {}
                for key, value in props.items():
                    seen.add((key, type_mapping.get(type(value).__name__, "STRING")))
            edge_properties.append(
                {
                    "type": label,
                    "properties": [{"property": key, "type": value} for key, value in seen],
                }
            )
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
                payload = record.get("result", record)
                from_labels = payload.get("from", [])
                to_labels = payload.get("to", [])
                if not from_labels or not to_labels:
                    continue
                triples.append(
                    triple_template.format(
                        a=from_labels[0],
                        e=payload.get("edge"),
                        b=to_labels[0],
                    )
                )
        return triples

    @staticmethod
    def _validate_label(label: str) -> str:
        """Validate label to prevent Cypher injection.

        Args:
            label: Label name to validate.

        Returns:
            The validated label.

        Raises:
            ValueError: If label contains invalid characters.
        """
        if not LABEL_PATTERN.match(label):
            raise ValueError(f"Invalid Neptune label: '{label}'")
        return label
