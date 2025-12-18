from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from dynamiq.connections import Neo4j
from dynamiq.nodes import ErrorHandling, NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.utils.logger import logger


class Neo4jSchemaInputSchema(BaseModel):
    database: str | None = Field(
        default=None,
        description="Optional database override.",
        json_schema_extra={"nullable": True, "additionalProperties": False},
    )
    include_properties: bool = Field(
        default=True,
        description="If true, include node and relationship property metadata.",
    )

    model_config = ConfigDict(extra="forbid")


class Neo4jSchemaIntrospector(ConnectionNode):
    """Tool to fetch labels, relationship types, and optional property metadata from Neo4j."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Neo4j Schema Introspector"
    description: str = (
        "Retrieves Neo4j schema for grounding Cypher. "
        "Inputs: database (optional home-db override), "
        "include_properties (bool; if true, fetch db.schema node/rel properties). "
        "Outputs: labels, relationship_types, node_properties, "
        "relationship_properties, and a content summary with samples."
    )
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    connection: Neo4j
    database: str | None = None

    input_schema: ClassVar[type[Neo4jSchemaInputSchema]] = Neo4jSchemaInputSchema
    _graph_store: Neo4jGraphStore | None = PrivateAttr(default=None)

    def init_components(self, connection_manager=None):
        super().init_components(connection_manager)
        self._graph_store = Neo4jGraphStore(connection=self.connection, client=self.client, database=self.database)

    def execute(self, input_data: Neo4jSchemaInputSchema, config: RunnableConfig = None, **kwargs):
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not self._graph_store:
            raise ToolExecutionException("Neo4j graph store is not initialized.", recoverable=True)

        database = input_data.database or self.database

        try:
            labels_records, _, _ = self._graph_store.run_cypher(
                "CALL db.labels() YIELD label RETURN label ORDER BY label",
                database=database,
            )
            reltype_records, _, _ = self._graph_store.run_cypher(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType ORDER BY relationshipType",
                database=database,
            )
            labels = [r["label"] for r in labels_records]
            rel_types = [r["relationshipType"] for r in reltype_records]

            node_props: list[dict[str, Any]] = []
            rel_props: list[dict[str, Any]] = []

            if input_data.include_properties:
                try:
                    node_props_records, _, _ = self._graph_store.run_cypher(
                        "CALL db.schema.nodeTypeProperties()",
                        database=database,
                    )
                    node_props = [r.data() for r in node_props_records]
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Node property introspection failed: {exc}")

                try:
                    rel_props_records, _, _ = self._graph_store.run_cypher(
                        "CALL db.schema.relTypeProperties()",
                        database=database,
                    )
                    rel_props = [r.data() for r in rel_props_records]
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Relationship property introspection failed: {exc}")

            payload = {
                "labels": labels,
                "relationship_types": rel_types,
                "node_properties": node_props,
                "relationship_properties": rel_props,
            }
            payload["content"] = self._build_content(payload)

            logger.info(f"Tool {self.name} - {self.id}: finished successfully. Output: {payload['content']}")
            return payload
        except Exception as exc:  # noqa: BLE001
            logger.error(f"Tool {self.name} - {self.id}: failed to introspect schema. Error: {exc}")
            raise ToolExecutionException(str(exc), recoverable=True) from exc

    @staticmethod
    def _build_content(payload: dict[str, Any]) -> str:
        labels = payload.get("labels") or []
        rels = payload.get("relationship_types") or []
        node_props = payload.get("node_properties") or []
        rel_props = payload.get("relationship_properties") or []
        node_samples = [
            f"{p.get('nodeLabels', ['?'])[0]}.{p.get('propertyName')}:{p.get('propertyTypes')}" for p in node_props[:5]
        ]
        rel_samples = [
            f"{p.get('relType', '?')}.{p.get('propertyName')}:{p.get('propertyTypes')}" for p in rel_props[:5]
        ]
        return (
            f"Labels: {labels}. "
            f"Relationship types: {rels}. "
            f"Node properties entries: {len(node_props)} (samples: {node_samples}). "
            f"Relationship properties entries: {len(rel_props)} (samples: {rel_samples})."
        )
