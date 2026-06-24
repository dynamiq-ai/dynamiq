import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, PrivateAttr

from dynamiq.connections import ApacheAGE, AWSNeptune, Neo4j
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.extractors.entity_extractor import (
    ATTRIBUTE_VALUE_LABEL,
    ENTITY_ID_INDEX,
    ENTITY_LABEL,
    ENTITY_NAME_FULLTEXT_INDEX,
    HAS_ATTRIBUTE_TYPE,
    KG_ENTITY_IDS_KEY,
    build_attribute_value_id,
)
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.graph.age import ApacheAgeGraphStore
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.storages.graph.neptune import NeptuneGraphStore
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger

# Cypher identifier guard (Neo4j/openCypher label syntax used by the resolution read query).
_LABEL_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _trigrams(text: str) -> set[str]:
    """Padded character trigrams of a normalized name (pg_trgm-style)."""
    norm = " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", text.lower())).strip() + " "
    return {norm[i : i + 3] for i in range(len(norm) - 2)} if len(norm) >= 3 else set()


def _trigram_similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)  # Jaccard


def _entity_ids_by_doc(relationships: list[dict]) -> dict[str, list[str]]:
    """Map each source document id -> the RESOLVED entity ids it mentions, recovered from relationships.

    Relationships carry ``source_doc_id`` (stamped by EntityExtractor) and, after resolution, durable
    endpoint ids. The entity ids a chunk mentions are therefore the endpoints of the relationships sourced
    from that chunk. ``HAS_ATTRIBUTE`` edges point at an ``AttributeValue`` holder, not an entity, so their
    end id is excluded (the owner entity on the start side is still kept).
    """
    by_doc: dict[str, set[str]] = {}
    for rel in relationships:
        doc_id = (rel.get("properties") or {}).get("source_doc_id")
        if not doc_id:
            continue
        ids = by_doc.setdefault(str(doc_id), set())
        if rel.get("start_identity"):
            ids.add(rel["start_identity"])
        if rel.get("type") != HAS_ATTRIBUTE_TYPE and rel.get("end_identity"):
            ids.add(rel["end_identity"])
    return {doc_id: sorted(ids) for doc_id, ids in by_doc.items()}


class KnowledgeGraphWriterInputSchema(BaseModel):
    """Input for the writer: an ``EntityExtractor``'s output payload.

    ``nodes`` / ``relationships`` are the provider-neutral graph payload; ``documents`` are the source
    chunks the payload was extracted from, tagged on output with the resolved entity ids they mention.
    """

    nodes: list[dict] = Field(default_factory=list, description="Extracted graph nodes (EntityExtractor output).")
    relationships: list[dict] = Field(default_factory=list, description="Extracted graph relationships.")
    documents: list[Document] = Field(
        default_factory=list, description="Source chunks to tag with the resolved entity ids each mentions."
    )


class KnowledgeGraphWriter(Node):
    """Writes an extracted knowledge graph to a graph store, assigning durable entity identity first.

    Consumes an :class:`~dynamiq.nodes.extractors.entity_extractor.EntityExtractor`'s output
    (``{nodes, relationships, documents}``) and:

      1. Write-time entity resolution: node identity is decided by NAME similarity only — the
         LLM-produced ids are just wiring that links edges to entities within one extraction and
         never participate in identity. An entity whose name is trigram-similar
         (>= ``similarity_threshold``) to an existing same-label node adopts that node's id;
         otherwise it is written as a new node under a fresh UUID. Re-running ingestion therefore
         converges onto existing entities instead of duplicating them.
      2. Upsert into the graph backend via ``BaseGraphStore.write_graph``.
      3. Tag each source chunk with the resolved entity ids it mentions (``kg_entity_ids``) so the
         chunk can go to a vector store and a hybrid retriever can seed graph traversal by unique id.

    Split from extraction so extraction can be parallelized in a flow: many ``EntityExtractor`` nodes
    (each on a shard of documents) can fan into a SINGLE ``KnowledgeGraphWriter``. The writer must stay
    one node — resolution reads the current graph plus a per-call candidate cache, so two writers running
    concurrently against the same graph would race and create duplicate entities.

    Provider-agnostic: the concrete store is selected from the connection type
    (``Neo4j`` / ``ApacheAGE`` / ``AWSNeptune``), exactly like ``CypherExecutor``. The write path
    requires a store that implements ``write_graph`` — currently Neo4j; other backends raise a
    clear error until their stores add it.

    Input: ``{"nodes": [...], "relationships": [...], "documents": [Document, ...]}``.
    Output: the resolved payload plus ``write_stats`` and ``nodes_created`` / ``relationships_created``.

    Attributes:
        connection (Neo4j | ApacheAGE | AWSNeptune): The graph backend connection.
        database (str | None): Optional target database (Neo4j).
        graph_name (str | None): Graph name (Apache AGE).
        create_graph_if_not_exists (bool): Create the AGE graph if missing.
        similarity_threshold (float): Trigram similarity above which two names are the same entity.
    """

    group: Literal[NodeGroup.EXTRACTORS] = NodeGroup.EXTRACTORS
    name: str = "knowledge-graph-writer"
    connection: Neo4j | ApacheAGE | AWSNeptune
    database: str | None = None
    graph_name: str | None = None
    create_graph_if_not_exists: bool = False
    similarity_threshold: float = 0.6

    input_schema: ClassVar[type[KnowledgeGraphWriterInputSchema]] = KnowledgeGraphWriterInputSchema
    _graph_store: BaseGraphStore | None = PrivateAttr(default=None)
    _existing_cache: dict[str, list[tuple[str, str]]] = PrivateAttr(default_factory=dict)

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Select the graph store for the connection and ensure the entity indexes exist."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self._graph_store is None:
            self._graph_store = self._build_graph_store()
        self._ensure_entity_index()

    def _ensure_entity_index(self) -> None:
        """Create the entity name full-text index AND the entity id range index (Neo4j) so GraphRetriever
        can seek instead of scan — by the question's words (full-text) or by resolved id (range index).

        Idempotent (``IF NOT EXISTS``) and best-effort — a failure (e.g. missing privileges) only means
        retrieval falls back to a scan, so it is logged, not raised. Neo4j-only; other backends skip it.
        """
        if not isinstance(self._graph_store, Neo4jGraphStore):
            return
        for statement, what in (
            (
                f"CREATE FULLTEXT INDEX {ENTITY_NAME_FULLTEXT_INDEX} IF NOT EXISTS "
                f"FOR (n:{ENTITY_LABEL}) ON EACH [n.name]",
                "full-text",
            ),
            (
                f"CREATE INDEX {ENTITY_ID_INDEX} IF NOT EXISTS FOR (n:{ENTITY_LABEL}) ON (n.id)",
                "id",
            ),
        ):
            try:
                self._graph_store.run_cypher(statement, database=self.database)
            except Exception as e:
                logger.warning(f"Node {self.name} - {self.id}: could not create entity {what} index: {e}")

    def _build_graph_store(self) -> BaseGraphStore:
        """Pick the concrete store from the connection type (same dispatch as CypherExecutor)."""
        client = self.connection.connect()
        if isinstance(self.connection, ApacheAGE):
            return ApacheAgeGraphStore(
                connection=self.connection,
                client=client,
                graph_name=self.graph_name,
                create_graph_if_not_exists=self.create_graph_if_not_exists,
            )
        if isinstance(self.connection, AWSNeptune):
            return NeptuneGraphStore(
                connection=self.connection,
                client=client,
                endpoint=self.connection.endpoint,
                verify_ssl=self.connection.verify_ssl,
                timeout=self.connection.timeout,
            )
        return Neo4jGraphStore(connection=self.connection, client=client, database=self.database)

    def execute(
        self, input_data: KnowledgeGraphWriterInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """Resolve extracted entities to durable identity and write them to the graph store."""
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        check_cancellation(config)

        if not self._graph_store.supports_write_graph():
            raise NotImplementedError(
                f"{type(self).__name__}: write is not supported for backend "
                f"{type(self._graph_store).__name__}. Only Neo4j currently implements write_graph."
            )

        self._existing_cache = {}
        nodes, relationships = input_data.nodes, input_data.relationships

        # Relationship endpoints reference node wiring ids; resolution (below) only rewrites endpoints found
        # among `nodes`. An endpoint absent from `nodes` can never be resolved -- it would be written with an
        # ephemeral id that MATCHes no graph node (creating no edge) and would mis-tag chunks with that id.
        # Reject such payloads instead of silently writing nothing useful.
        if relationships:
            node_ids = {n["properties"]["id"] for n in nodes}
            dangling = [
                r for r in relationships if r["start_identity"] not in node_ids or r["end_identity"] not in node_ids
            ]
            if dangling:
                raise ValueError(
                    f"KnowledgeGraphWriter: {len(dangling)} relationship(s) reference endpoints absent from "
                    f"`nodes` ({len(nodes)} node(s) given); relationships must be written together with their "
                    "endpoint nodes."
                )

        if nodes:
            nodes, relationships = self._resolve_against_graph(nodes, relationships)

        if nodes or relationships:
            write_result = self._graph_store.write_graph(
                nodes=nodes, relationships=relationships, database=self.database
            )
        else:
            write_result = {
                "nodes_created": 0,
                "relationships_created": 0,
                "properties_set": 0,
                "records": [],
                "keys": [],
            }

        logger.info(
            f"Node {self.name} - {self.id}: wrote {write_result.get('nodes_created')} new node(s) and "
            f"{write_result.get('relationships_created')} new relationship(s)."
        )

        # Return the input chunks tagged with the RESOLVED entity ids each mentions, so they can go to a
        # vector store and a hybrid retriever can seed graph traversal by unique id (not by ambiguous name).
        documents = self._attach_entity_ids(input_data.documents, relationships)

        return {
            "nodes": nodes,
            "relationships": relationships,
            "documents": documents,
            "write_stats": write_result,
            "nodes_created": write_result.get("nodes_created"),
            "relationships_created": write_result.get("relationships_created"),
        }

    @staticmethod
    def _attach_entity_ids(documents: list, relationships: list[dict]) -> list:
        """Return COPIES of the chunks, each with ``kg_entity_ids`` = the resolved entity ids it mentions.

        Recovered from the resolved relationships by ``source_doc_id`` (see :func:`_entity_ids_by_doc`).
        The caller's documents are not mutated. A chunk whose entities appear in no relationship gets an
        empty list — it still flows to the vector store, just with no graph seeds.
        """
        by_doc = _entity_ids_by_doc(relationships)
        enriched = []
        for document in documents:
            ids = by_doc.get(str(document.id), [])
            metadata = {**(document.metadata or {}), KG_ENTITY_IDS_KEY: ids}
            enriched.append(document.model_copy(update={"metadata": metadata}))
        return enriched

    def _existing_nodes(self, label: str) -> list[tuple[str, str]]:
        """All (id, name) of existing nodes with the given label, cached per execute() call."""
        if label in self._existing_cache:
            return self._existing_cache[label]
        rows: list[tuple[str, str]] = []
        if _LABEL_PATTERN.match(label):
            records, _, _ = self._graph_store.run_cypher(
                f"MATCH (n:`{label}`) WHERE n.name IS NOT NULL RETURN n.id AS id, n.name AS name",
                database=self.database,
            )
            rows = [(r.get("id"), r.get("name")) for r in self._graph_store.format_records(records)]
        self._existing_cache[label] = rows
        return rows

    def _resolve_against_graph(self, nodes: list[dict], relationships: list[dict]) -> tuple[list[dict], list[dict]]:
        """Assign graph identity to every extracted entity by name similarity ONLY.

        LLM-produced ids are ephemeral wiring: they link edges to entities within one extraction
        and never participate in identity. Every entity either adopts the id of the best
        trigram-matching existing node (same label, score >= ``similarity_threshold``) or gets a
        fresh UUID. Newly assigned entities are added to the per-call candidate cache so later
        entities in the same batch converge onto them too. ``AttributeValue`` node ids are
        re-derived from their owner's resolved id (``{owner_id}::{attr_key}::{doc_id}``).
        """
        id_remap: dict[str, str] = {}
        for node in nodes:
            label = node["labels"][0]
            if label == ATTRIBUTE_VALUE_LABEL:
                continue
            old_id = node["properties"]["id"]
            if old_id in id_remap:
                continue  # wiring ids are doc-scoped, so a repeat means the same in-document entity
            name = node["properties"].get("name")

            best_id, best_score = None, 0.0
            for ex_id, ex_name in self._existing_nodes(label) if name else []:
                score = _trigram_similarity(name, ex_name)
                if score >= self.similarity_threshold and score > best_score:
                    best_id, best_score = ex_id, score

            if best_id:
                id_remap[old_id] = best_id
                logger.info(
                    f"Node {self.name} - {self.id}: linking {name!r} -> existing node "
                    f"{best_id!r} (trigram={best_score:.2f})"
                )
            else:
                id_remap[old_id] = generate_uuid()
                if name:
                    # Make the new node a match candidate for the rest of this batch.
                    self._existing_cache.setdefault(label, []).append((id_remap[old_id], name))

        # AttributeValue ids derive from their owner entity so re-ingestion updates the same value
        # node. The structured (owner, key, doc) parts ride on the node, so we rebuild the resolved
        # id directly via build_attribute_value_id — no string parsing, so an LLM id containing "::"
        # can't corrupt the split. Pop the transient ref so it never reaches the store.
        for node in nodes:
            if node["labels"][0] != ATTRIBUTE_VALUE_LABEL:
                continue
            ref = node.pop("attr_ref", None)
            old_id = node["properties"]["id"]
            if ref and ref["owner"] in id_remap:
                id_remap[old_id] = build_attribute_value_id(id_remap[ref["owner"]], ref["key"], ref["doc"])

        for node in nodes:
            nid = node["properties"]["id"]
            if nid in id_remap:
                node["properties"]["id"] = id_remap[nid]
        for rel in relationships:
            if rel["start_identity"] in id_remap:
                rel["start_identity"] = id_remap[rel["start_identity"]]
            if rel["end_identity"] in id_remap:
                rel["end_identity"] = id_remap[rel["end_identity"]]

        # Two extracted entities can resolve to the same node -> dedupe by (label, id).
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for node in nodes:
            key = (node["labels"][0], node["properties"]["id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(node)
        return deduped, relationships
