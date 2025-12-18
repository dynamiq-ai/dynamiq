from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections import Neo4j
from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.utils.logger import logger

DESCRIPTION_NEO4J_CYPHER = """Executes parameterized Cypher against Neo4j

Inputs:
- query: Cypher text
- parameters: dict for $params (additionalProperties disabled)
- database: optional override
- routing: 'r' for read / 'w' for write (clusters)
- return_graph: if true, return nodes/relationships instead of rows

Outputs:
- records or graph, keys, summary (query, counters, latency), content summary with preview.
If return_graph is true, graph.nodes/graph.relationships are returned instead of tabular rows.

Key capabilities:
- Safe, parameterized Cypher execution (MATCH/OPTIONAL MATCH/RETURN, MERGE/SET when allowed)
- Supports database selection and read routing
- Returns tabular records or graph objects

Usage tips:
- Always use parameters (e.g., $name) instead of string concatenation
- Set routing='r' for read-only queries in clusters
- Provide database explicitly when available"""


class Neo4jCypherInputSchema(BaseModel):
    query: str = Field(..., description="Cypher query to execute.")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters for the Cypher query (use $param syntax in query).",
        json_schema_extra={"type": "object", "properties": {}, "additionalProperties": False},
    )
    database: str | None = Field(default=None, description="Optional database name override.")
    routing: str | None = Field(default=None, description="Routing preference ('r' for read, 'w' for write).")
    return_graph: bool = Field(
        default=False,
        description="If true, returns the Neo4j graph result (nodes/relationships) instead of rows.",
    )

    model_config = ConfigDict(extra="forbid")


class Neo4jCypherExecutor(ConnectionNode):
    """Tool for executing Cypher queries against Neo4j."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Neo4j Cypher Executor"
    description: str = DESCRIPTION_NEO4J_CYPHER
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    connection: Neo4j
    database: str | None = None

    input_schema: ClassVar[type[Neo4jCypherInputSchema]] = Neo4jCypherInputSchema
    _graph_store: Neo4jGraphStore | None = PrivateAttr(default=None)

    def init_components(self, connection_manager=None):
        super().init_components(connection_manager)
        self._graph_store = Neo4jGraphStore(connection=self.connection, client=self.client, database=self.database)

    def execute(self, input_data: Neo4jCypherInputSchema, config: RunnableConfig = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not self._graph_store:
            raise ToolExecutionException("Neo4j graph store is not initialized.", recoverable=True)

        database = input_data.database or self.database
        routing = input_data.routing
        transformer = None
        result_payload: dict[str, Any] = {}

        try:
            if input_data.return_graph:
                import neo4j

                transformer = neo4j.Result.graph

            records, summary, keys = self._graph_store.run_cypher(
                query=input_data.query,
                parameters=input_data.parameters,
                database=database,
                routing=routing,
                result_transformer=transformer,
            )

            if input_data.return_graph:
                result_payload["graph"] = {
                    "nodes": list(records.nodes),
                    "relationships": list(records.relationships),
                }
                result_payload["keys"] = []
            else:
                result_payload["records"] = Neo4jGraphStore.format_records(records)
                result_payload["keys"] = keys

            result_payload["summary"] = {
                "query": summary.query,
                "counters": summary.counters,
                "result_available_after": summary.result_available_after,
            }
            result_payload["query"] = input_data.query
            result_payload["parameters_used"] = input_data.parameters
            result_payload["content"] = self._build_content(result_payload, input_data.return_graph)

            logger.info(f"Tool {self.name} - {self.id}: finished successfully. Content: {result_payload['content']}")
            return result_payload
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Tool {self.name} - {self.id}: failed to execute Cypher. Error: {exc}")
            raise ToolExecutionException(str(exc), recoverable=True) from exc

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
