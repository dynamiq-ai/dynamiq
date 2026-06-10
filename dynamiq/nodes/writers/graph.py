from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, PrivateAttr

from dynamiq.connections import Neo4j as Neo4jConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger


class GraphWriterInputSchema(BaseModel):
    nodes: list[dict] = Field(default_factory=list, description="Nodes to upsert (write_graph node payloads).")
    relationships: list[dict] = Field(
        default_factory=list, description="Relationships to upsert (write_graph relationship payloads)."
    )


class Neo4jGraphWriter(ConnectionNode):
    """Writes entities and relationships into a Neo4j knowledge graph.

    Thin writer node around :meth:`Neo4jGraphStore.write_graph`. It expects payloads already shaped
    for ``write_graph`` (e.g. produced by
    :class:`~dynamiq.nodes.extractors.entity_extractor.EntityExtractor`):

    - node: ``{"labels": [...], "identity_key": "id", "properties": {"id": ..., ...}}``
    - relationship: ``{"type": ..., "start_label": ..., "end_label": ..., "start_identity_key": "id",
      "end_identity_key": "id", "start_identity": ..., "end_identity": ..., "properties": {...}}``

    Attributes:
        group (Literal[NodeGroup.WRITERS]): Node group. Defaults to NodeGroup.WRITERS.
        name (str): Node name. Defaults to "neo4j-graph-writer".
        connection (Neo4j): The Neo4j connection.
        database (str | None): Optional target database; overrides the connection default.
    """

    group: Literal[NodeGroup.WRITERS] = NodeGroup.WRITERS
    action_type: ActionType = ActionType.DATABASE_QUERY
    name: str = "neo4j-graph-writer"
    connection: Neo4jConnection
    database: str | None = None
    input_schema: ClassVar[type[GraphWriterInputSchema]] = GraphWriterInputSchema

    _graph_store: Neo4jGraphStore | None = PrivateAttr(default=None)

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Initialize the Neo4j graph store from the connection/client."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        self._graph_store = Neo4jGraphStore(connection=self.connection, client=self.client, database=self.database)

    def ensure_client(self) -> None:
        """Reconnect if the client was closed and keep the graph store's client in sync."""
        previous_client = self.client
        super().ensure_client()
        if self.client is previous_client:
            return
        if not self._graph_store:
            return
        if getattr(self._graph_store, "client", None) is not self.client:
            self._graph_store.update_client(self.client)

    def execute(self, input_data: GraphWriterInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Upsert the provided nodes and relationships into Neo4j.

        Returns:
            dict: The ``write_graph`` stats (``nodes_created``, ``relationships_created``,
            ``properties_set``, ``records``, ``keys``). Returns zeroed stats if there is nothing to write.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        nodes = input_data.nodes
        relationships = input_data.relationships

        if not nodes and not relationships:
            logger.debug(f"Node {self.name} - {self.id}: nothing to write (no nodes or relationships).")
            return {
                "nodes_created": 0,
                "relationships_created": 0,
                "properties_set": 0,
                "records": [],
                "keys": [],
            }

        result = self._graph_store.write_graph(nodes=nodes, relationships=relationships, database=self.database)
        logger.debug(
            f"Node {self.name} - {self.id}: wrote {result.get('nodes_created')} nodes, "
            f"{result.get('relationships_created')} relationships to Neo4j."
        )
        return result
