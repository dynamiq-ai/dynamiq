from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from dynamiq.connections import Neo4j
from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.utils.logger import logger

DESCRIPTION_NEO4J_GRAPH_WRITER = """Upserts nodes and relationships into Neo4j using MERGE + SET.

Inputs:
- nodes: list of {labels, identity_key, properties (must include identity_key)}
- relationships: list of {type, start_label/end_label, start_identity/end_identity, identity_keys, properties}
- database: optional override

Outputs:
- records/keys, counters, input_preview, content summary.

Rules:
- Each node requires labels and an identity_key present in properties (default: id)
- Relationships reference existing or newly merged nodes by label + identity
- Uses parameterized Cypher; avoids string concatenation"""


class Neo4jNodeInput(BaseModel):
    labels: list[str] = Field(..., description="Labels to apply to the node.")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Node properties.",
        json_schema_extra={"type": "object", "properties": {}, "additionalProperties": True},
    )
    identity_key: str = Field(default="id", description="Property key used to identify/merge the node.")

    @model_validator(mode="after")
    def ensure_identity(self):
        if self.identity_key not in self.properties:
            raise ValueError(f"Node properties must include the identity key '{self.identity_key}'.")
        return self


class Neo4jRelationshipInput(BaseModel):
    type: str = Field(..., description="Relationship type (e.g., KNOWS).")
    start_label: str = Field(..., description="Label of the start node.")
    end_label: str = Field(..., description="Label of the end node.")
    start_identity: Any = Field(..., description="Identity value of the start node.")
    end_identity: Any = Field(..., description="Identity value of the end node.")
    start_identity_key: str = Field(default="id", description="Identity key on the start node.")
    end_identity_key: str = Field(default="id", description="Identity key on the end node.")
    properties: dict[str, Any] = Field(
        default_factory=dict,
        description="Relationship properties.",
        json_schema_extra={"type": "object", "properties": {}, "additionalProperties": True},
    )


class Neo4jGraphWriterInputSchema(BaseModel):
    nodes: list[Neo4jNodeInput] = Field(
        default_factory=list,
        description="Nodes to upsert. Each entry must include labels, identity_key, and properties with that key.",
        json_schema_extra={"additionalProperties": False},
    )
    relationships: list[Neo4jRelationshipInput] = Field(
        default_factory=list,
        description="Relationships to merge. Nodes are matched by label + identity.",
        json_schema_extra={"additionalProperties": False},
    )
    database: str | None = Field(
        default=None,
        description="Optional database override.",
        json_schema_extra={"nullable": True, "additionalProperties": False},
    )

    model_config = ConfigDict(extra="forbid")


class Neo4jGraphWriter(ConnectionNode):
    """Tool for upserting nodes/relationships into Neo4j."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Neo4j Graph Writer"
    description: str = DESCRIPTION_NEO4J_GRAPH_WRITER
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    connection: Neo4j
    database: str | None = None

    input_schema: ClassVar[type[Neo4jGraphWriterInputSchema]] = Neo4jGraphWriterInputSchema
    _graph_store: Neo4jGraphStore | None = PrivateAttr(default=None)

    def init_components(self, connection_manager=None):
        super().init_components(connection_manager)
        self._graph_store = Neo4jGraphStore(connection=self.connection, client=self.client, database=self.database)

    def execute(self, input_data: Neo4jGraphWriterInputSchema, config: RunnableConfig = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not self._graph_store:
            raise ToolExecutionException("Neo4j graph store is not initialized.", recoverable=True)

        database = input_data.database or self.database
        nodes_payload = [node.model_dump() for node in input_data.nodes]
        relationships_payload = [relationship.model_dump() for relationship in input_data.relationships]

        try:
            result = self._graph_store.write_graph(
                nodes=nodes_payload,
                relationships=relationships_payload,
                database=database,
            )
            result["input_preview"] = {
                "nodes": nodes_payload[:3],
                "relationships": relationships_payload[:3],
            }
            result["content"] = (
                "Upserted graph. "
                f"Nodes created: {result.get('nodes_created', 0)}, "
                f"Relationships created: {result.get('relationships_created', 0)}, "
                f"Properties set: {result.get('properties_set', 0)}. "
                f"Input preview: {result['input_preview']}."
            )
            logger.info(f"Tool {self.name} - {self.id}: finished successfully. Content: {result['content']}")
            return result
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Tool {self.name} - {self.id}: failed to upsert graph. Error: {exc}")
            raise ToolExecutionException(str(exc), recoverable=True) from exc
