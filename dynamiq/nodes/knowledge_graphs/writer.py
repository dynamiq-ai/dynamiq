import re
from typing import Any, ClassVar, Literal
from uuid import NAMESPACE_DNS, uuid5

from pydantic import BaseModel, Field, PrivateAttr

from dynamiq.connections import ApacheAGE, AWSNeptune, Neo4j
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes.embedders.base import DocumentEmbedder
from dynamiq.nodes.knowledge_graphs.entity_extractor import (
    ATTRIBUTE_VALUE_LABEL,
    ENTITY_EMBEDDING_VECTOR_INDEX,
    ENTITY_ID_INDEX,
    ENTITY_LABEL,
    ENTITY_NAME_FULLTEXT_INDEX,
    HAS_ATTRIBUTE_TYPE,
    build_attribute_value_id,
    build_fact_text,
    normalize_name,
)
from dynamiq.nodes.node import Node, NodeGroup, ensure_config
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.graph.age import ApacheAgeGraphStore
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.storages.graph.neptune import NeptuneGraphStore
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils import generate_uuid
from dynamiq.utils.logger import logger

# Fixed namespace for deterministic entity ids: uuid5(_KG_NAMESPACE, f"{label}:{normalize_name(name)}").
# A stable constant so the same (type, name) always hashes to the same id, across machines and runs.
_KG_NAMESPACE = uuid5(NAMESPACE_DNS, "dynamiq.knowledge-graph.entity")


def _trigrams(text: str) -> set[str]:
    """Padded character trigrams of a normalized name (pg_trgm-style)."""
    norm = " " + normalize_name(text) + " "
    return {norm[i : i + 3] for i in range(len(norm) - 2)} if len(norm) >= 3 else set()


def _lucene_or_query(text: str) -> str:
    """Fuzzy Lucene OR-query over a name's word tokens (``tok~ OR tok~``) for full-text blocking.

    Keeps only word tokens so Lucene special characters can't break the parser; ``~`` adds edit-distance
    fuzziness. Returns "" when there are no usable tokens.
    """
    terms = re.findall(r"[A-Za-z0-9]+", text or "")
    return " OR ".join(f"{t}~" for t in terms)


def _trigram_similarity(a: str, b: str) -> float:
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)  # Jaccard


class KnowledgeGraphWriterInputSchema(BaseModel):
    """Input for the writer: an ``EntityExtractor``'s output payload (the provider-neutral graph)."""

    nodes: list[dict] = Field(default_factory=list, description="Extracted graph nodes (EntityExtractor output).")
    relationships: list[dict] = Field(default_factory=list, description="Extracted graph relationships.")


class KnowledgeGraphWriter(Node):
    """Writes an extracted knowledge graph to a graph store, assigning durable entity identity first.

    Consumes an :class:`~dynamiq.nodes.extractors.entity_extractor.EntityExtractor`'s output
    (``{nodes, relationships}``) and:

      1. Write-time entity resolution: node identity is decided by NAME similarity only — the
         LLM-produced ids are just wiring that links edges to entities within one extraction and
         never participate in identity. An entity whose name is trigram-similar
         (>= ``similarity_threshold``) to an existing same-label node adopts that node's id;
         otherwise it is written as a new node under a fresh UUID. Re-running ingestion therefore
         converges onto existing entities instead of duplicating them.
      2. Upsert into the graph backend via ``BaseGraphStore.write_graph``.

    Split from extraction so extraction can be parallelized in a flow: many ``EntityExtractor`` nodes
    (each on a shard of documents) can fan into a SINGLE ``KnowledgeGraphWriter``. The writer must stay
    one node — resolution reads the current graph plus a per-call candidate cache, so two writers running
    concurrently against the same graph would race and create duplicate entities.

    Provider-agnostic: the concrete store is selected from the connection type
    (``Neo4j`` / ``ApacheAGE`` / ``AWSNeptune``), exactly like ``CypherExecutor``. The write path
    requires a store that implements ``write_graph`` — currently Neo4j; other backends raise a
    clear error until their stores add it.

    Input: ``{"nodes": [...], "relationships": [...]}``.
    Output: ``{"nodes_created": int, "relationships_created": int}`` — the write counts.

    Attributes:
        connection (Neo4j | ApacheAGE | AWSNeptune): The graph backend connection.
        database (str | None): Optional target database (Neo4j).
        graph_name (str | None): Graph name (Apache AGE).
        create_graph_if_not_exists (bool): Create the AGE graph if missing.
        similarity_threshold (float): Trigram similarity above which two names are the same entity.
        fuzzy_matching (bool): Identity always uses the deterministic id ``uuid5(label:normalize_name(name))``
            so identical (normalized) names collapse for free with no graph read (idempotent). This flag
            only toggles the OPTIONAL fuzzy tier on top: when ``True`` (default), spelling variants
            ("Acme" ≈ "Acme LLC") are additionally merged by pulling a bounded set of near candidates from
            an index (entity vector index if embeddings are on, else the entity-name full-text index) and
            confirming with trigram — embeddings only *find* candidates, trigram *decides*, so distinct
            same-category names ("John Smith" vs "John Doe") are never fused. ``False`` = deterministic only.
        resolution_top_k (int): Max candidates pulled from the index per name during fuzzy matching.
        entity_embedder (DocumentEmbedder | None): Optional embedder (Neo4j only). When set: (1) each
            entity's name is embedded onto the node with a vector index, so ``GraphRetriever`` can seed
            traversal by semantic similarity; and (2) each relationship's triplet ("src rel dst: desc") is
            embedded onto the EDGE as ``embedding`` (no new node, ACL stays on the edge), so the retriever
            can rerank retrieved facts by relevance. Off by default. Use the SAME embedding model on the
            retriever so vector dimensions match.
    """

    group: Literal[NodeGroup.EXTRACTORS] = NodeGroup.EXTRACTORS
    name: str = "knowledge-graph-writer"
    connection: Neo4j | ApacheAGE | AWSNeptune
    database: str | None = None
    graph_name: str | None = None
    create_graph_if_not_exists: bool = False
    similarity_threshold: float = 0.6
    fuzzy_matching: bool = True
    resolution_top_k: int = 10
    entity_embedder: DocumentEmbedder | None = None

    input_schema: ClassVar[type[KnowledgeGraphWriterInputSchema]] = KnowledgeGraphWriterInputSchema
    _graph_store: BaseGraphStore | None = PrivateAttr(default=None)
    # Vector indexes ensured this process (by name) — created lazily on first embed from the real embedding
    # length so a dimension can never mismatch the embedder's model.
    _vector_indexes_ready: set[str] = PrivateAttr(default_factory=set)

    @property
    def to_dict_exclude_params(self) -> dict:
        return super().to_dict_exclude_params | {"entity_embedder": True}

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        if self.entity_embedder:
            data["entity_embedder"] = self.entity_embedder.to_dict(**kwargs)
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Select the graph store for the connection and ensure the entity indexes exist."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.entity_embedder and self.entity_embedder.is_postponed_component_init:
            self.entity_embedder.init_components(connection_manager)
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

    def _embed_texts(self, texts: list[str], config: RunnableConfig, **kwargs) -> dict[str, list[float]]:
        """Embed a list of texts in ONE embedder call → ``{text: vector}``; ``{}`` when disabled or on failure.

        Neo4j + ``entity_embedder`` gated (embeddings back a Neo4j vector index, so a raw vector on any other
        backend is unqueryable dead weight). Best-effort: any embedder failure is logged and returns ``{}`` so
        the write proceeds without embeddings.
        """
        if not self.entity_embedder or not isinstance(self._graph_store, Neo4jGraphStore):
            return {}
        texts = [t for t in dict.fromkeys(texts) if (t or "").strip()]
        if not texts:
            return {}
        try:
            run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            run_kwargs.pop("run_depends", None)
            result = self.entity_embedder.run(
                input_data={"documents": [Document(content=t) for t in texts]}, config=config, **run_kwargs
            )
            if result.status != RunnableStatus.SUCCESS:
                logger.warning(f"Node {self.name} - {self.id}: embedder failed ({result.error}); skipping embeddings.")
                return {}
            return {
                doc.content: doc.embedding
                for doc in (result.output.get("documents") or [])
                if doc.embedding is not None
            }
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Node {self.name} - {self.id}: embedding error: {e}; skipping embeddings.")
            return {}

    def _embed_entity_names(self, nodes: list[dict], config: RunnableConfig, **kwargs) -> dict[str, list[float]]:
        """Embed every entity node's NAME (one call) → ``{name: vector}``. Reused for the fuzzy candidate
        lookup (Tier 2 of resolution) and node storage. ``AttributeValue`` holders are doc-scoped values, not
        entities that get seeded/matched on, so they're skipped."""
        names = [
            n["name"]
            for n in nodes
            if n["labels"][0] != ATTRIBUTE_VALUE_LABEL and (n.get("name") or "").strip()
        ]
        return self._embed_texts(names, config, **kwargs)

    @staticmethod
    def _attach_embeddings(nodes: list[dict], name_vectors: dict[str, list[float]]) -> list[dict]:
        """Attach each entity node's precomputed name embedding (from ``_embed_entity_names``) as a property.

        Runs after resolution on the deduped node set. No-op when there are no embeddings.
        """
        if not name_vectors:
            return nodes
        for node in nodes:
            if node["labels"][0] == ATTRIBUTE_VALUE_LABEL:
                continue
            embedding = name_vectors.get(node.get("name"))
            if embedding is not None:
                node["properties"]["embedding"] = embedding
        return nodes

    @staticmethod
    def _edge_triplet_text(rel: dict) -> str | None:
        """The triplet text to embed for an edge, or ``None`` when it can't be rendered (missing names).

        Mirrors ``GraphRetriever``'s render: the attribute KEY is the relation for ``HAS_ATTRIBUTE`` edges,
        else the relationship type; the endpoint nodes' ``name`` snapshots are the fact's endpoints.
        """
        props = rel.get("properties") or {}
        rel_type = rel.get("type") or ""
        src_name = (rel.get("start_node") or {}).get("name")
        dst_name = (rel.get("end_node") or {}).get("name")
        if not (rel_type and src_name and dst_name):
            return None
        rel_label = props["key"] if rel_type == HAS_ATTRIBUTE_TYPE and props.get("key") else rel_type
        return build_fact_text(src_name, rel_label, dst_name, rel.get("description"))

    def _maybe_embed_edges(self, relationships: list[dict], config: RunnableConfig, **kwargs) -> None:
        """Embed each relationship's triplet and store it on the EDGE (``properties['embedding']``), in place.

        No-op without an embedder / non-Neo4j store (``_embed_texts`` returns ``{}``). The embedding rides
        through the existing edge write path (``SET r += props``); the retriever uses it to rerank facts and
        strips it from output. ACL stays on the edge — no new node, no new access-control surface.
        """
        if not relationships:
            return
        texts = {id(rel): self._edge_triplet_text(rel) for rel in relationships}
        vectors = self._embed_texts([t for t in texts.values() if t], config, **kwargs)
        if not vectors:
            return
        for rel in relationships:
            text = texts[id(rel)]
            vector = vectors.get(text) if text else None
            if vector is not None:
                # New props dict (not in-place): `_resolve_against_graph` shallow-copies rels, so the
                # properties dict is shared with the caller's input — replacing the ref keeps input pristine.
                rel["properties"] = {**(rel.get("properties") or {}), "embedding": vector}

    def _ensure_vector_index(self, index_name: str, label: str, dimension: int) -> None:
        """Create a Neo4j (>= 5.11) vector index ``index_name`` on ``(:label).embedding``, once per process.

        Lazy from the REAL embedding length so the dimension can never mismatch the model. Idempotent
        (``IF NOT EXISTS``), best-effort (logged not raised), Neo4j-only.
        """
        if index_name in self._vector_indexes_ready or not isinstance(self._graph_store, Neo4jGraphStore):
            return
        statement = (
            f"CREATE VECTOR INDEX {index_name} IF NOT EXISTS "
            f"FOR (n:{label}) ON n.embedding "
            f"OPTIONS {{indexConfig: {{`vector.dimensions`: {int(dimension)}, "
            "`vector.similarity_function`: 'cosine'}}"
        )
        try:
            self._graph_store.run_cypher(statement, database=self.database)
            self._vector_indexes_ready.add(index_name)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Node {self.name} - {self.id}: could not create vector index {index_name!r}: {e}")

    def _ensure_entity_vector_index(self, dimension: int) -> None:
        """Create the entity-name vector index so ``GraphRetriever`` can seed by similarity."""
        self._ensure_vector_index(ENTITY_EMBEDDING_VECTOR_INDEX, ENTITY_LABEL, dimension)

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

        nodes, relationships = input_data.nodes, input_data.relationships

        # Resolution only rewrites endpoints found among `nodes`. An endpoint absent from `nodes` can't be
        # resolved -- it would get an ephemeral id that matches no graph node, leaving a dangling edge.
        # Reject such payloads instead of silently writing nothing useful.
        if relationships:
            node_ids = {n["id"] for n in nodes}
            dangling = [
                r for r in relationships if r["start_node"]["id"] not in node_ids or r["end_node"]["id"] not in node_ids
            ]
            if dangling:
                raise ValueError(
                    f"KnowledgeGraphWriter: {len(dangling)} relationship(s) reference endpoints absent from "
                    f"`nodes` ({len(nodes)} node(s) given); relationships must be written together with their "
                    "endpoint nodes."
                )

        # Bare nodes (referenced by no relationship) are never persisted -- see _drop_bare_nodes.
        nodes = self._drop_bare_nodes(nodes, relationships)

        if nodes:
            # Embed BEFORE resolving so one embed call serves both: the fuzzy candidate lookup (Tier 2)
            # and node storage. Ensure the vector index up front so resolution can query it this write.
            name_vectors = self._embed_entity_names(nodes, config, **kwargs)
            if name_vectors:
                self._ensure_entity_vector_index(len(next(iter(name_vectors.values()))))
            nodes, relationships = self._resolve_against_graph(nodes, relationships, name_vectors)
            nodes = self._attach_embeddings(nodes, name_vectors)
            # Embed each relationship's triplet onto the edge (for fact reranking at retrieval time).
            self._maybe_embed_edges(relationships, config, **kwargs)

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

        return {
            "nodes_created": write_result.get("nodes_created"),
            "relationships_created": write_result.get("relationships_created"),
        }

    def delete_documents(self, document_ids: list[str], key: str = "source_doc_id") -> dict[str, int]:
        """Remove everything the given documents contributed to the graph (delegates to the store).

        Deletion is clean by construction: every edge carries the provenance of the document that
        asserted it, and the same fact from two documents is two separate edges — so removing one
        document's edges never erases another document's identical claim. Doc-scoped ``AttributeValue``
        holders are removed with their edges. A shared entity (identity) node is kept while another
        document still cites it, and swept once this delete removes its last edge — an entity with no
        remaining edge is invisible to retrieval anyway, so it is cleaned up rather than left orphaned.

        ``key`` is the edge property the ids match against: ``source_doc_id`` (default) deletes by the
        ingestion chunk id; any other flattened document-metadata key works too — e.g. ``file_id`` deletes
        every edge extracted from any chunk of a knowledgebase file (the store validates it).

        The documents' chunks in the vector store are NOT touched — delete those separately with the
        vector store's own delete-by-ids (this node does not own that store). Backends without graph
        writes raise ``NotImplementedError`` (only Neo4j implements deletion today).

        Returns ``{"relationships_deleted": int, "nodes_deleted": int}``.
        """
        if not document_ids:
            return {"relationships_deleted": 0, "nodes_deleted": 0}
        totals = self._graph_store.delete_documents(
            [str(document_id) for document_id in document_ids],
            doc_scoped_labels=[ATTRIBUTE_VALUE_LABEL],
            orphan_labels=[ENTITY_LABEL],
            provenance_key=key,
            database=self.database,
        )
        logger.info(
            f"Node {self.name} - {self.id}: deleted {totals.get('relationships_deleted')} relationship(s) "
            f"and {totals.get('nodes_deleted')} node(s) (doc-scoped + newly-orphaned entities) for "
            f"{len(document_ids)} document(s)."
        )
        return totals

    def _drop_bare_nodes(self, nodes: list[dict], relationships: list[dict]) -> list[dict]:
        """Drop nodes referenced by NO relationship (bare mentions) instead of writing them.

        All provenance and ACL live on edges, so a bare node could never be attributed to any document:
        retrieval (edge-driven) cannot reach it, and ``delete_documents`` could never remove it — the
        orphan sweep only examines endpoints of removed edges. Skipping the write keeps the invariant that everything
        persisted is reachable through at least one provenance-stamped edge, so deleting a document
        provably removes all it contributed. Nothing is lost: entity ids are content-addressed
        (``uuid5`` of label + normalized name), so a future document asserting a fact about the same
        name re-creates the identical node.
        """
        if not nodes:
            return nodes
        referenced = {r["start_node"]["id"] for r in relationships} | {r["end_node"]["id"] for r in relationships}
        kept = [node for node in nodes if node["id"] in referenced]
        if len(kept) != len(nodes):
            logger.info(
                f"Node {self.name} - {self.id}: skipping {len(nodes) - len(kept)} bare node(s) "
                "referenced by no relationship."
            )
        return kept

    @staticmethod
    def _deterministic_id(label: str, name: str) -> str:
        """Content-addressed node id: ``uuid5(_KG_NAMESPACE, f"{label}:{normalize_name(name)}")``.

        Same (type, normalized name) → same id, on every machine and run — so identical names collapse
        under ``MERGE`` with no read (Tier 1), and re-ingestion is idempotent. The label is in the hashed
        string, so an "Apple" Organization and an "Apple" Product never collide.
        """
        return str(uuid5(_KG_NAMESPACE, f"{label}:{normalize_name(name)}"))

    def _existing_ids(self, ids: list[str]) -> set[str]:
        """Which of ``ids`` are already saved as ``:Entity`` nodes — one batched, index-backed lookup.

        Uses the entity-id range index (``entity_id``), so it's a b-tree seek per id, not a scan. Lets
        resolution skip the fuzzy tier for entities that already exist exactly. Best-effort and Neo4j-only:
        on any other backend or any failure it returns an empty set, so fuzzy simply runs (never blocks
        the write).
        """
        if not ids or not isinstance(self._graph_store, Neo4jGraphStore):
            return set()
        try:
            records, _, _ = self._graph_store.run_cypher(
                f"UNWIND $ids AS id MATCH (n:{ENTITY_LABEL} {{id: id}}) RETURN n.id AS id",
                parameters={"ids": list(dict.fromkeys(ids))},
                database=self.database,
            )
            return {r.get("id") for r in self._graph_store.format_records(records)}
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Node {self.name} - {self.id}: existing-id lookup failed ({e}); running fuzzy.")
            return set()

    def _blocking_candidates(self, label: str, name: str, vector: list[float] | None) -> list[tuple[str, str]]:
        """Bounded set of existing same-label entities near ``name`` for fuzzy resolution (Neo4j only).

        Prefers the entity vector index when a name embedding is available (catches synonyms). If that
        query FAILS (missing / dim-mismatched vector index, permissions), it degrades to the entity-name
        full-text index rather than silently returning nothing — otherwise a transient vector-index fault
        would blind resolution to existing nodes and write duplicate entities. Best-effort throughout: any
        remaining failure yields no candidates, so resolution keeps the deterministic id. Returns
        ``(id, name)`` pairs.
        """
        if vector is not None:
            candidates = self._vector_candidates(label, vector)
            if candidates is not None:  # vector query ran (may legitimately be empty) — trust it
                return candidates
            # vector index errored -> degrade to the lexical full-text path instead of returning []
        return self._fulltext_candidates(label, name)

    def _vector_candidates(self, label: str, vector: list[float]) -> list[tuple[str, str]] | None:
        """Nearest same-label entities by name embedding (semantic near-dups). Returns ``None`` on query
        failure so the caller can fall back to full-text; an empty list means the query ran but matched
        nothing.
        """
        try:
            # Over-fetch (queryNodes returns nearest across ALL types) then keep this label's.
            records, _, _ = self._graph_store.run_cypher(
                "CALL db.index.vector.queryNodes($index, $k, $vec) YIELD node "
                "WHERE $label IN labels(node) AND node.name IS NOT NULL "
                "RETURN node.id AS id, node.name AS name",
                parameters={
                    "index": ENTITY_EMBEDDING_VECTOR_INDEX,
                    "k": max(self.resolution_top_k * 5, self.resolution_top_k),
                    "vec": vector,
                    "label": label,
                },
                database=self.database,
            )
            return [(r.get("id"), r.get("name")) for r in self._graph_store.format_records(records)]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Node {self.name} - {self.id}: vector candidate lookup failed ({e}); " "falling back to full-text."
            )
            return None

    def _fulltext_candidates(self, label: str, name: str) -> list[tuple[str, str]]:
        """Same-label entities lexically near ``name`` via the entity-name full-text index. Best-effort:
        any failure (e.g. the index doesn't exist yet on the first write) yields no candidates, so
        resolution keeps the deterministic id.
        """
        lucene = _lucene_or_query(name)
        if not lucene:
            return []
        try:
            records, _, _ = self._graph_store.run_cypher(
                "CALL db.index.fulltext.queryNodes($index, $q) YIELD node "
                "WHERE $label IN labels(node) AND node.name IS NOT NULL "
                "RETURN node.id AS id, node.name AS name LIMIT $k",
                parameters={
                    "index": ENTITY_NAME_FULLTEXT_INDEX,
                    "q": lucene,
                    "label": label,
                    "k": self.resolution_top_k,
                },
                database=self.database,
            )
            return [(r.get("id"), r.get("name")) for r in self._graph_store.format_records(records)]
        except Exception as e:  # noqa: BLE001
            logger.warning(
                f"Node {self.name} - {self.id}: full-text candidate lookup failed ({e}); " "using deterministic id."
            )
            return []

    def _find_canonical(
        self,
        label: str,
        name: str,
        det_id: str,
        vector: list[float] | None,
        batch_candidates: list[tuple[str, str]],
        use_db: bool,
    ) -> str | None:
        """Tier 2: return the id of an existing/earlier entity that is a trigram near-dup of ``name``.

        Candidates come from the index (``_blocking_candidates``, Neo4j) plus this-write entities already
        resolved (``batch_candidates``). The merge decision is ALWAYS trigram ≥ ``similarity_threshold`` —
        embeddings/full-text only propose candidates, so distinct same-category names ("John Smith" vs
        "John Doe") are never fused. Returns ``None`` when nothing clears the threshold (keep ``det_id``).
        """
        candidates = list(batch_candidates)
        if use_db:
            candidates += self._blocking_candidates(label, name, vector)

        best_id, best_score = None, 0.0
        for cand_id, cand_name in candidates:
            if not cand_id or not cand_name or cand_id == det_id:
                continue  # skip self: an exact-normalized dup already shares det_id
            score = _trigram_similarity(name, cand_name)
            if score >= self.similarity_threshold and score > best_score:
                best_id, best_score = cand_id, score
        if best_id:
            logger.info(
                f"Node {self.name} - {self.id}: linking {name!r} -> existing node {best_id!r} "
                f"(trigram={best_score:.2f})"
            )
        return best_id

    def _resolve_against_graph(
        self, nodes: list[dict], relationships: list[dict], name_vectors: dict[str, list[float]] | None = None
    ) -> tuple[list[dict], list[dict]]:
        """Assign durable identity to every extracted entity — deterministic base + optional fuzzy tier.

        LLM-produced ids are ephemeral wiring. Each entity ALWAYS gets a **deterministic** id
        ``uuid5(label:normalize_name(name))`` (Tier 1) — identical names collapse under ``MERGE`` with no
        graph read, and re-ingestion is idempotent. When ``fuzzy_matching`` is on (default), spelling
        variants are additionally merged (Tier 2, optional): for entities whose deterministic id is not
        already saved, a bounded index-backed candidate lookup + trigram confirm may adopt an existing
        entity's id instead. ``AttributeValue`` ids are re-derived from their owner's resolved id.

        Works on copies of the caller's dicts (rewrites the top-level ``id``, strips the transient
        ``attr_ref``), so the input stays writable a second time.
        """
        name_vectors = name_vectors or {}
        nodes = [{**node, "properties": {**node["properties"]}} for node in nodes]
        # Copy the endpoint nodes too: their `id` is rewritten below, and a bare `{**rel}` would share the
        # nested start_node/end_node dicts with the caller's input (which must stay writable a second time).
        relationships = [
            {**rel, "start_node": {**rel["start_node"]}, "end_node": {**rel["end_node"]}} for rel in relationships
        ]

        fuzzy = self.fuzzy_matching
        use_db = fuzzy and isinstance(self._graph_store, Neo4jGraphStore)

        # Tier 1: deterministic id per named entity (nameless ones get a fresh uuid). One pass, so the
        # existence check below can be batched over the whole set.
        det_by_old: dict[str, str] = {}
        for node in nodes:
            if node["labels"][0] == ATTRIBUTE_VALUE_LABEL:
                continue
            old_id = node["id"]
            if old_id in det_by_old:
                continue  # wiring ids are doc-scoped, so a repeat means the same in-document entity
            name = node.get("name")
            # Gate on the NORMALIZED name: a name that is only whitespace/apostrophes normalizes to "" and
            # must stay nameless (fresh uuid), else every such entity would collide on the "{label}:" id.
            det_by_old[old_id] = (
                self._deterministic_id(node["labels"][0], name) if name and normalize_name(name) else generate_uuid()
            )

        # Tier 2 GATE: an entity whose deterministic id is ALREADY in the graph is an established exact
        # node — keep it and skip fuzzy, so re-ingestion is a no-op and fuzzy can never re-merge it into a
        # similar-but-different node. One batched, index-backed existence lookup answers this for all ids.
        already_saved = self._existing_ids(list(det_by_old.values())) if fuzzy else set()
        # This-write candidates so variants within one batch converge (brand-new ids aren't indexed yet).
        batch_candidates: dict[str, list[tuple[str, str]]] = {}

        id_remap: dict[str, str] = {}
        for node in nodes:
            label = node["labels"][0]
            if label == ATTRIBUTE_VALUE_LABEL:
                continue
            old_id = node["id"]
            if old_id in id_remap:
                continue
            name = node.get("name")
            det_id = det_by_old[old_id]
            resolved = det_id
            if fuzzy and name and det_id not in already_saved:
                match = self._find_canonical(
                    label, name, det_id, name_vectors.get(name), batch_candidates.get(label, []), use_db
                )
                if match:
                    resolved = match
            id_remap[old_id] = resolved
            if name:
                batch_candidates.setdefault(label, []).append((resolved, name))

        # AttributeValue ids derive from their owner so re-ingestion updates the same node. Rebuild the id
        # from the (owner, key, doc) parts on the node via build_attribute_value_id (no string parsing, so
        # an LLM id containing "::" can't corrupt it). Pop the transient ref so it never reaches the store.
        for node in nodes:
            if node["labels"][0] != ATTRIBUTE_VALUE_LABEL:
                continue
            ref = node.pop("attr_ref", None)
            old_id = node["id"]
            if ref and ref["owner"] in id_remap:
                id_remap[old_id] = build_attribute_value_id(id_remap[ref["owner"]], ref["key"], ref["doc"])

        for node in nodes:
            nid = node["id"]
            if nid in id_remap:
                node["id"] = id_remap[nid]
        for rel in relationships:
            if rel["start_node"]["id"] in id_remap:
                rel["start_node"]["id"] = id_remap[rel["start_node"]["id"]]
            if rel["end_node"]["id"] in id_remap:
                rel["end_node"]["id"] = id_remap[rel["end_node"]["id"]]

        # Two extracted entities can resolve to the same node -> dedupe by (label, id).
        seen: set[tuple[str, str]] = set()
        deduped: list[dict] = []
        for node in nodes:
            key = (node["labels"][0], node["id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(node)
        return deduped, relationships
