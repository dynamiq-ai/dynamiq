import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.connections import ApacheAge, Neo4j
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.graph.age import ApacheAgeGraphStore
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.utils.logger import logger

BASE_CYPHER_DESCRIPTION = """Executes parameterized Cypher against Neo4j or Apache AGE, or introspects schema

Inputs:
- mode: execute | introspect
- query: Cypher text or list of Cypher texts (execute mode)
- parameters: dict for $params, or list aligned with query list (additionalProperties allowed)
- database: optional override
- routing: 'r' for read / 'w' for write (clusters)
- return_graph: if true, return nodes/relationships instead of rows
- include_properties: when introspecting, include node/rel property metadata
- allow_writes: if false, block write queries by regex guardrails

Outputs:
- execute mode: records or graph, keys, summary (query, counters, latency), content summary with preview.
  For multi-query, results is a list of per-query payloads and content summarizes each query.
  If return_graph is true, graph.nodes/graph.relationships are returned instead of tabular rows.
- introspect mode: labels, relationship_types, node_properties, relationship_properties, and content summary.
If return_graph is true, graph.nodes/graph.relationships are returned instead of tabular rows.

Key capabilities:
- Safe, parameterized Cypher execution (MATCH/OPTIONAL MATCH/RETURN, MERGE/SET when allowed)
- Supports database selection and read routing
- Returns tabular records or graph objects
- Fetches schema labels/relationship types/property metadata

Usage tips:
- Always use parameters (e.g., $name) instead of string concatenation
- Set routing='r' for read-only queries in clusters
- Provide database explicitly when available
- Use mode="introspect" to fetch schema hints for query generation
- For writes, avoid comma-separated MATCH/MERGE patterns (cartesian products)
- Prefer safe write pattern:
  MATCH (p:Person {id: $person_id})
  WITH p
  MATCH (c:Company {id: $company_id})
  MERGE (p)-[r:WORKS_AT]->(c)
  SET r.role = $role
- For write-then-read flows, prefer a multi-query call with a list of queries and parameters."""

NEO4J_BACKEND_NOTES = """
Neo4j notes:
- return_graph is supported to return nodes/relationships instead of rows.
- routing can be 'r' or 'w' when using Neo4j clusters.
- database can target a named Neo4j database.
"""

AGE_BACKEND_NOTES = """
Apache AGE notes:
- AGE requires queries to RETURN a single column; alias it as `result` (e.g., RETURN n AS result).
- return_graph is not supported for AGE backends.
- Provide graph_name via the ApacheAge connection or the tool's graph_name field.
- Avoid passing a list of queries; use a single query string for AGE.
"""


class CypherInputSchema(BaseModel):
    """Schema for Cypher tool inputs.

    Args:
        mode: Execution mode.
        query: Cypher query or list of queries in execute mode.
        parameters: Parameters for Cypher execution.
        database: Optional database name override.
        routing: Routing preference for clustered deployments.
        return_graph: Whether to return graph results instead of rows.
        include_properties: Whether to include node and relationship property metadata.
        allow_writes: Whether to allow write queries.

    Raises:
        ValueError: If required fields are missing or incompatible with the selected mode.
    """

    model_config = ConfigDict(extra="forbid")

    mode: Literal["execute", "introspect"] = Field(default="execute", description="Execution mode.")
    query: str | list[str] | None = Field(
        default=None, description="Cypher query or list of queries (execute mode only)."
    )
    parameters: dict[str, Any] | list[dict[str, Any]] = Field(
        default_factory=dict,
        description="Parameters for the Cypher query (use $param syntax in query).",
        json_schema_extra={"type": "object", "properties": {}, "additionalProperties": True},
    )
    database: str | None = Field(default=None, description="Optional database name override.")
    routing: str | None = Field(default=None, description="Routing preference ('r' for read, 'w' for write).")
    return_graph: bool = Field(
        default=False,
        description="If true, returns the Neo4j graph result (nodes/relationships) instead of rows.",
    )
    include_properties: bool = Field(
        default=True,
        description="If true, include node and relationship property metadata (introspect mode).",
    )
    allow_writes: bool = Field(default=True, description="If false, reject write queries by regex guardrails.")

    @model_validator(mode="after")
    def validate_mode_inputs(self):
        if self.mode == "execute":
            if isinstance(self.query, list):
                if not self.query or any(not str(item).strip() for item in self.query):
                    raise ValueError("query is required in execute mode.")
            elif not (self.query or "").strip():
                raise ValueError("query is required in execute mode.")
        else:
            if self.return_graph:
                raise ValueError("return_graph is only supported in execute mode.")
            if self.query is not None:
                raise ValueError("query is not supported in introspect mode.")
            if self.parameters:
                raise ValueError("parameters are not supported in introspect mode.")
        if isinstance(self.query, list):
            if isinstance(self.parameters, list):
                if len(self.query) != len(self.parameters):
                    raise ValueError("parameters list must match query list length.")
        elif isinstance(self.parameters, list):
            raise ValueError("parameters list is only supported when query is a list.")
        return self


class CypherExecutor(ConnectionNode):
    """Tool for executing Cypher queries against Neo4j or Apache AGE."""

    input_schema: ClassVar[type[CypherInputSchema]] = CypherInputSchema

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Cypher Executor"
    description: str = BASE_CYPHER_DESCRIPTION
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    connection: Neo4j | ApacheAge
    database: str | None = None
    graph_name: str | None = None
    create_graph_if_missing: bool | None = None
    _graph_store: BaseGraphStore | None = PrivateAttr(default=None)
    _backend_name: str | None = PrivateAttr(default=None)

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Initialize graph store and backend metadata.

        Args:
            connection_manager: Optional connection manager instance.
        """
        super().init_components(connection_manager)
        if isinstance(self.connection, ApacheAge):
            self._backend_name = "age"
            self._graph_store = ApacheAgeGraphStore(
                connection=self.connection,
                client=self.client,
                graph_name=self.graph_name,
                create_graph_if_missing=self.create_graph_if_missing,
            )
        else:
            self._backend_name = "neo4j"
            self._graph_store = Neo4jGraphStore(connection=self.connection, client=self.client, database=self.database)
        self.description = self._build_description()

    def ensure_client(self) -> None:
        previous_client = self.client
        super().ensure_client()
        if self.client is previous_client:
            return
        if not self._graph_store:
            return
        graph_client = getattr(self._graph_store, "client", None)
        if graph_client is not self.client:
            self._graph_store.update_client(self.client)

    def _build_description(self) -> str:
        if self._backend_name == "age":
            return BASE_CYPHER_DESCRIPTION + AGE_BACKEND_NOTES
        if self._backend_name == "neo4j":
            return BASE_CYPHER_DESCRIPTION + NEO4J_BACKEND_NOTES
        return BASE_CYPHER_DESCRIPTION

    def execute(self, input_data: CypherInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Run Cypher queries or introspect schema via the configured backend.

        Args:
            input_data: Validated Cypher input payload.
            config: Optional runnable configuration.
            **kwargs: Extra execution context forwarded to callbacks.

        Returns:
            Dictionary payload containing records or graph output, plus metadata.

        Raises:
            ToolExecutionException: If execution fails or the graph store is not initialized.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not self._graph_store:
            raise ToolExecutionException("Graph store is not initialized.", recoverable=True)

        database = input_data.database or self.database
        routing = input_data.routing
        result_payload: dict[str, Any] = {}

        try:
            if input_data.mode == "introspect":
                result_payload = self._graph_store.introspect_schema(
                    include_properties=input_data.include_properties,
                    database=database,
                )
                result_payload["mode"] = input_data.mode
                result_payload["content"] = self._build_schema_content(result_payload)
                logger.info(
                    f"Tool {self.name} - {self.id}: finished successfully. Content: {result_payload['content']}"
                )
                return result_payload

            if isinstance(input_data.query, list):
                results = self._execute_batch(
                    queries=input_data.query,
                    parameters=input_data.parameters,
                    database=database,
                    routing=routing,
                    return_graph=input_data.return_graph,
                    allow_writes=input_data.allow_writes,
                )
                result_payload = {
                    "mode": input_data.mode,
                    "queries": [self._clean_query(query) for query in input_data.query],
                    "results": results,
                }
                result_payload["content"] = self._build_batch_content(results, input_data.return_graph)
                logger.info(
                    f"Tool {self.name} - {self.id}: finished successfully. Content: {result_payload['content']}"
                )
                return result_payload

            result_payload = self._execute_single(
                query=input_data.query or "",
                parameters=input_data.parameters,
                database=database,
                routing=routing,
                return_graph=input_data.return_graph,
                allow_writes=input_data.allow_writes,
            )
            result_payload["mode"] = input_data.mode
            logger.info(f"Tool {self.name} - {self.id}: finished successfully. Content: {result_payload['content']}")
            return result_payload
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Tool {self.name} - {self.id}: failed to execute Cypher. Error: {exc}")
            raise ToolExecutionException(str(exc), recoverable=True) from exc

    def _execute_batch(
        self,
        *,
        queries: list[str],
        parameters: dict[str, Any] | list[dict[str, Any]],
        database: str | None,
        routing: str | None,
        return_graph: bool,
        allow_writes: bool,
    ) -> list[dict[str, Any]]:
        if isinstance(parameters, list):
            params_list = parameters
        else:
            params_list = [parameters for _ in queries]
        results: list[dict[str, Any]] = []
        for query, query_params in zip(queries, params_list, strict=True):
            results.append(
                self._execute_single(
                    query=query,
                    parameters=query_params,
                    database=database,
                    routing=routing,
                    return_graph=return_graph,
                    allow_writes=allow_writes,
                )
            )
        return results

    def _execute_single(
        self,
        *,
        query: str,
        parameters: dict[str, Any],
        database: str | None,
        routing: str | None,
        return_graph: bool,
        allow_writes: bool,
    ) -> dict[str, Any]:
        transformer = None
        cleaned_query = self._clean_query(query or "")
        self._validate_query(cleaned_query, allow_writes=allow_writes)

        if return_graph:
            if not self._graph_store.supports_graph_result():
                raise ToolExecutionException("return_graph is only supported for Neo4j backends.", recoverable=True)
            import neo4j

            transformer = neo4j.Result.graph

        records, summary, keys = self._graph_store.run_cypher(
            query=cleaned_query,
            parameters=parameters,
            database=database,
            routing=routing,
            result_transformer=transformer,
        )

        result_payload: dict[str, Any] = {}
        if return_graph:
            result_payload["graph"] = self._serialize_graph(records)
            result_payload["keys"] = []
        else:
            result_payload["records"] = self._graph_store.format_records(records)
            result_payload["keys"] = keys or []

        result_payload["summary"] = self._build_summary(summary, cleaned_query)
        result_payload["query"] = cleaned_query
        result_payload["parameters_used"] = parameters
        result_payload["content"] = self._build_content(result_payload, return_graph)
        return result_payload

    @staticmethod
    def _build_content(payload: dict[str, Any], is_graph: bool) -> str:
        summary = payload.get("summary", {})
        counters = summary.get("counters")
        counters_text = str(counters) if counters is not None else "None"
        query_text = payload.get("query", "")
        params = payload.get("parameters_used", {})
        if is_graph:
            graph = payload.get("graph", {})
            node_count = len(graph.get("nodes", []))
            rel_count = len(graph.get("relationships", []))
            return (
                f"Query: {query_text}. Params: {params}. "
                f"Executed graph query. Nodes: {node_count}, Relationships: {rel_count}. "
                f"Counters: {counters_text}."
            )
        records = payload.get("records", [])
        preview = records[:3] if records else []
        return (
            f"Query: {query_text}. Params: {params}. "
            f"Returned {len(records)} records. Preview: {preview}. "
            f"Counters: {counters_text}."
        )

    @staticmethod
    def _build_schema_content(payload: dict[str, Any]) -> str:
        def _first_label(value: Any) -> str:
            if isinstance(value, list):
                return value[0] if value else "?"
            return value or "?"

        labels = payload.get("labels") or []
        rels = payload.get("relationship_types") or []
        node_props = payload.get("node_properties") or []
        rel_props = payload.get("relationship_properties") or []
        node_samples = []
        for entry in node_props[:5]:
            if "nodeLabels" in entry:
                node_samples.append(
                    f"{_first_label(entry.get('nodeLabels'))}.{entry.get('propertyName')}:{entry.get('propertyTypes')}"
                )
            else:
                props = entry.get("properties") or []
                if props:
                    node_samples.append(f"{entry.get('labels')}.{props[0].get('property')}:{props[0].get('type')}")
        rel_samples = []
        for entry in rel_props[:5]:
            if "relType" in entry:
                rel_samples.append(
                    f"{_first_label(entry.get('relType'))}.{entry.get('propertyName')}:{entry.get('propertyTypes')}"
                )
            else:
                props = entry.get("properties") or []
                if props:
                    rel_samples.append(f"{entry.get('type')}.{props[0].get('property')}:{props[0].get('type')}")
        return (
            f"Labels: {labels}. "
            f"Relationship types: {rels}. "
            f"Node properties entries: {len(node_props)} (samples: {node_samples}). "
            f"Relationship properties entries: {len(rel_props)} (samples: {rel_samples})."
        )

    @staticmethod
    def _build_batch_content(results: list[dict[str, Any]], is_graph: bool) -> str:
        if not results:
            return "No queries executed."
        snippets = []
        for index, payload in enumerate(results, start=1):
            summary = payload.get("summary", {})
            counters = summary.get("counters")
            counters_text = str(counters) if counters is not None else "None"
            query_text = payload.get("query", "")
            params = payload.get("parameters_used", {})
            if is_graph:
                graph = payload.get("graph", {})
                node_count = len(graph.get("nodes", []))
                rel_count = len(graph.get("relationships", []))
                snippets.append(
                    f"[{index}] Query: {query_text}. Params: {params}. "
                    f"Nodes: {node_count}, Relationships: {rel_count}. Counters: {counters_text}."
                )
            else:
                records = payload.get("records", [])
                preview = records[:3] if records else []
                snippets.append(
                    f"[{index}] Query: {query_text}. Params: {params}. "
                    f"Returned {len(records)} records. Preview: {preview}. Counters: {counters_text}."
                )
        return " ".join(snippets)

    @classmethod
    def _build_summary(cls, summary: Any, fallback_query: str) -> dict[str, Any]:
        payload = {"query": fallback_query, "counters": {}, "result_available_after": None}
        if summary is None:
            return payload
        if isinstance(summary, dict):
            payload["query"] = summary.get("query", payload["query"])
            payload["counters"] = summary.get("counters", payload["counters"])
            payload["result_available_after"] = summary.get(
                "result_available_after",
                payload["result_available_after"],
            )
            return payload

        payload["query"] = getattr(summary, "query", payload["query"])
        counters = getattr(summary, "counters", None)
        payload["counters"] = cls._serialize_counters(counters) if counters is not None else {}
        payload["result_available_after"] = getattr(summary, "result_available_after", None)
        return payload

    @staticmethod
    def _clean_query(query: str) -> str:
        cleaned = (query or "").strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").strip()
            if cleaned.lower().startswith("cypher"):
                cleaned = cleaned[len("cypher") :].strip()
        return cleaned

    @classmethod
    def _validate_query(cls, query: str, *, allow_writes: bool) -> None:
        if not query:
            raise ToolExecutionException("Cypher query cannot be empty.", recoverable=True)
        if not allow_writes and cls._contains_write(query):
            raise ToolExecutionException("Cypher contains write operations but allow_writes is false.")
        if allow_writes and cls._contains_write(query) and cls._contains_cartesian_match(query):
            raise ToolExecutionException(
                "Cypher contains comma-separated MATCH/MERGE patterns that may create cartesian products. "
                "Use chained MATCH with WITH, or a single MATCH with relationship patterns.",
                recoverable=True,
            )

    @staticmethod
    def _contains_write(cypher: str) -> bool:
        pattern = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|DROP|REMOVE)\b", re.IGNORECASE)
        return bool(pattern.search(cypher or ""))

    @staticmethod
    def _contains_cartesian_match(cypher: str) -> bool:
        if not cypher:
            return False
        pattern = re.compile(r"\b(MATCH|MERGE)\b[\s\S]*?,\s*\(", re.IGNORECASE)
        return bool(pattern.search(cypher))

    @staticmethod
    def _serialize_graph(graph) -> dict[str, Any]:
        """Convert Neo4j Graph result into JSON-serializable structures."""
        if graph is None:
            return {"nodes": [], "relationships": []}

        def _node_to_dict(node):
            return {
                "id": getattr(node, "id", None),
                "element_id": getattr(node, "element_id", None),
                "labels": list(getattr(node, "labels", [])),
                "properties": dict(node),
            }

        def _relationship_to_dict(rel):
            start_node = getattr(rel, "start_node", None)
            end_node = getattr(rel, "end_node", None)
            start_node_id = (
                getattr(start_node, "id", None) if start_node is not None else getattr(rel, "start_node_id", None)
            )
            end_node_id = getattr(end_node, "id", None) if end_node is not None else getattr(rel, "end_node_id", None)
            start_node_element_id = (
                getattr(start_node, "element_id", None)
                if start_node is not None
                else getattr(rel, "start_node_element_id", None)
            )
            end_node_element_id = (
                getattr(end_node, "element_id", None)
                if end_node is not None
                else getattr(rel, "end_node_element_id", None)
            )
            return {
                "id": getattr(rel, "id", None),
                "element_id": getattr(rel, "element_id", None),
                "type": getattr(rel, "type", None),
                "start_node_id": start_node_id,
                "end_node_id": end_node_id,
                "start_node_element_id": start_node_element_id,
                "end_node_element_id": end_node_element_id,
                "properties": dict(rel),
            }

        nodes = [_node_to_dict(node) for node in getattr(graph, "nodes", [])]
        relationships = [_relationship_to_dict(rel) for rel in getattr(graph, "relationships", [])]

        return {"nodes": nodes, "relationships": relationships}

    @staticmethod
    def _serialize_counters(counters) -> dict[str, Any]:
        """Convert Neo4j SummaryCounters to a JSON-serializable dict."""
        if counters is None:
            return {}

        counter_fields = [
            "nodes_created",
            "nodes_deleted",
            "relationships_created",
            "relationships_deleted",
            "properties_set",
            "labels_added",
            "labels_removed",
            "indexes_added",
            "indexes_removed",
            "constraints_added",
            "constraints_removed",
            "system_updates",
        ]

        counters_dict = {field: getattr(counters, field, 0) for field in counter_fields}
        if hasattr(counters, "contains_updates"):
            value = counters.contains_updates
            counters_dict["contains_updates"] = value() if callable(value) else value
        if hasattr(counters, "contains_system_updates"):
            value = counters.contains_system_updates
            counters_dict["contains_system_updates"] = value() if callable(value) else value

        return counters_dict
