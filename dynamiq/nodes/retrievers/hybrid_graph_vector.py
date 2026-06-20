"""Hybrid retriever that fuses a vector store with a knowledge graph into one structured context.

It composes two existing nodes — any :class:`VectorStoreRetriever` (dense + BM25 hybrid over chunks) and
any :class:`GraphRetriever` (bounded, ACL-filtered graph traversal) — and combines their results the way
Onyx does: vector search finds the relevant passages, the resolved entity ids those passages mention (the
``kg_entity_ids`` metadata that ``KnowledgeGraphWriter`` attaches per chunk) seed a graph expansion, and
the two are returned as one sectioned context (``## Passages`` + ``## Facts``).

Seeding by id (not name) means the graph is expanded from the exact, unique entities the chunks were
linked to at write time — variant-proof. Seeding from the retrieved passages (rather than the raw query)
keeps the facts relevant to what vector search surfaced; when no passage carries entity ids it falls back
to the query.

This is the backend-agnostic counterpart to the single-Postgres ``GraphVectorRetriever``: here the vector
store and the graph can be different systems (e.g. pgvector + Neo4j), fused at the node level.
"""

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, PrivateAttr

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.extractors.entity_extractor import KG_ENTITY_IDS_KEY
from dynamiq.nodes.node import Node, NodeDependency, NodeGroup, ensure_config
from dynamiq.nodes.retrievers.graph import GraphRetriever
from dynamiq.nodes.retrievers.retriever import VectorStoreRetriever
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger


class HybridGraphVectorRetrieverInputSchema(BaseModel):
    query: str = Field(..., description="Natural-language question to retrieve context for.")
    top_k: int | None = Field(default=None, description="Override the number of vector passages.")
    alpha: float | None = Field(
        default=None, ge=0, le=1, description="Hybrid alpha for the vector retriever (0=keyword, 1=semantic)."
    )
    filters: dict[str, Any] = Field(default_factory=dict, description="Metadata filters for the vector retriever.")


class HybridGraphVectorRetriever(Node):
    """Fuses a vector retriever and a graph retriever into one sectioned context.

    Composes (does not subclass) a :class:`VectorStoreRetriever` and a :class:`GraphRetriever`, so each can
    point at a different backend and keep its own config (the vector retriever's hybrid ``alpha`` /
    reranker, the graph retriever's LOCKED ACL ``filters``). Flow:

      1. Vector-retrieve passages for the query.
      2. Collect the resolved entity ids those passages mention (their ``kg_entity_ids`` metadata) as graph
         seeds — falling back to the query when no passage carries ids.
      3. Graph-retrieve facts anchored on those seed ids.
      4. Render ``## Passages`` + ``## Facts`` as one ``content`` string, and return all documents tagged
         ``metadata["origin"]`` = ``"passage"`` | ``"fact"``.

    Output: ``{"content": <sectioned text>, "documents": [Document, ...]}`` (same shape as the other
    retrievers, so it drops into agents/tools unchanged).

    Attributes:
        vector_retriever (VectorStoreRetriever): The dense+keyword retriever over the enriched chunks.
        graph_retriever (GraphRetriever): The knowledge-graph retriever to expand passage entities.
        seed_graph_from_passages (bool): Seed the graph with the entity ids the retrieved passages mention
            (Onyx-style). When False, the graph is queried with the raw query only.
        max_seed_entities (int): Cap on how many passage entity ids seed the graph traversal.
        ground_facts_with_source_chunks (bool): Cite each fact with its source chunk (``source_doc_id``).
            Facts whose source chunk was retrieved are cited to that passage; for facts from chunks vector
            search did NOT return, the chunk is fetched by id (when the vector retriever supports it) and
            added as evidence. Needs ``get_documents_by_ids`` on the vector retriever (pgvector only today;
            otherwise facts are still cited to retrieved passages, just not back-filled).
    """

    group: Literal[NodeGroup.RETRIEVERS] = NodeGroup.RETRIEVERS
    action_type: ActionType = ActionType.SEMANTIC_SEARCH
    name: str = "hybrid-graph-vector-retriever"
    description: str = (
        "Retrieves supporting passages from a vector store AND the knowledge-graph facts about the "
        "entities those passages mention, merged into one sectioned context with each fact cited to its "
        "source chunk. Use for questions that need both source text and how things are connected."
    )
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    vector_retriever: VectorStoreRetriever
    graph_retriever: GraphRetriever
    reranker: Node | None = None  # optional reranker over the facts (e.g. a cross-encoder reranker node)
    seed_graph_from_passages: bool = True
    max_seed_entities: int = 10
    # Graph expansion depth. 1 = single hop. >1 iterates hop-by-hop (a beam search), pruning the frontier
    # each hop — NOT one exploding *1..N traversal. The inner GraphRetriever is single-hop by design; this
    # node drives the multi-hop expansion by re-seeding it with each hop's neighbor ids.
    max_hops: int = 1
    beam_width: int = 10  # frontier cap per hop — bounds the iterative expansion
    fact_limit: int = 50  # max facts kept after ranking (guards against hub-entity edge floods)
    ground_facts_with_source_chunks: bool = True

    input_schema: ClassVar[type[HybridGraphVectorRetrieverInputSchema]] = HybridGraphVectorRetrieverInputSchema
    _run_depends: list = PrivateAttr(default_factory=list)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._run_depends = []

    def reset_run_state(self):
        self._run_depends = []

    @property
    def to_dict_exclude_params(self):
        return super().to_dict_exclude_params | {
            "vector_retriever": True,
            "graph_retriever": True,
            "reranker": True,
        }

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        data["vector_retriever"] = self.vector_retriever.to_dict(**kwargs)
        data["graph_retriever"] = self.graph_retriever.to_dict(**kwargs)
        if self.reranker:
            data["reranker"] = self.reranker.to_dict(**kwargs)
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.vector_retriever.is_postponed_component_init:
            self.vector_retriever.init_components(connection_manager)
        if self.graph_retriever.is_postponed_component_init:
            self.graph_retriever.init_components(connection_manager)
        if self.reranker and self.reranker.is_postponed_component_init:
            self.reranker.init_components(connection_manager)

    @staticmethod
    def _seed_entity_ids(passages: list[Document], max_seed_entities: int) -> list[str]:
        """Distinct resolved entity ids mentioned across the retrieved passages, in order, capped."""
        seeds: list[str] = []
        seen: set[str] = set()
        for doc in passages:
            for entity_id in (doc.metadata or {}).get(KG_ENTITY_IDS_KEY) or []:
                if entity_id and entity_id not in seen:
                    seen.add(entity_id)
                    seeds.append(entity_id)
        return seeds[:max_seed_entities]

    @staticmethod
    def _fact_source_ids(fact: Document) -> list[str]:
        """The chunk ids a fact was extracted from — its ``source_doc_ids`` (multi-hop) / ``source_doc_id``."""
        metadata = fact.metadata or {}
        ids = [str(i) for i in (metadata.get("source_doc_ids") or [])]
        scalar = metadata.get("source_doc_id")
        if scalar and str(scalar) not in ids:
            ids.append(str(scalar))
        return ids

    @staticmethod
    def _fact_neighbor_ids(fact: Document) -> list[str]:
        """The endpoint entity ids of a fact (``source_id``/``target_id``, set by GraphRetriever) — the
        candidate seeds for the next hop when iterating over the graph."""
        metadata = fact.metadata or {}
        return [str(metadata[key]) for key in ("source_id", "target_id") if metadata.get(key)]

    @classmethod
    def _next_frontier(
        cls, facts: list[Document], passage_ids: set[str], visited: set[str], beam_width: int
    ) -> list[str]:
        """Entity ids to expand on the next hop: neighbor ids of this hop's top (passage-grounded) facts
        that haven't been visited yet, capped to ``beam_width``. Mutates ``visited`` with the chosen ids."""
        frontier: list[str] = []
        for fact in cls._passage_grounded_first(facts, passage_ids)[:beam_width]:
            for neighbor_id in cls._fact_neighbor_ids(fact):
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    frontier.append(neighbor_id)
        return frontier[:beam_width]

    def _expand_graph(
        self, seeds: list[str], query: str, passages: list[Document], config: RunnableConfig, **kwargs
    ) -> list[Document]:
        """Expand the graph one hop at a time (beam search), instead of one exploding ``*1..N`` traversal.

        Each hop is a cheap 1-hop GraphRetriever call seeded by the previous hop's kept neighbor ids; the
        frontier is pruned (passage-grounded, ``beam_width``) so reaching ``max_hops`` stays bounded. Hop 0
        falls back to query-based entry when there are no seeds. Returns the deduped facts across all hops.
        """
        passage_ids = {str(p.id) for p in passages}
        visited: set[str] = set(seeds)
        frontier: list[str] = list(seeds)
        collected: list[Document] = []
        seen_facts: set[str] = set()

        for hop in range(max(1, self.max_hops)):
            if hop > 0 and not frontier:
                break  # nothing new to expand
            graph_input: dict[str, Any] = {"query": query}
            if frontier:
                graph_input["entity_ids"] = frontier
            check_cancellation(config)
            graph_out = self.graph_retriever.run(
                input_data=graph_input, run_depends=self._run_depends, config=config, **kwargs
            )
            if graph_out.status != RunnableStatus.SUCCESS:
                error = graph_out.error.message if graph_out.error else "unknown error"
                raise RuntimeError(f"Graph retriever failed: {error}")
            self._run_depends = [NodeDependency(node=self.graph_retriever).to_dict(for_tracing=True)]

            fresh = [f for f in graph_out.output.get("documents", []) if f.content not in seen_facts]
            for fact in fresh:
                seen_facts.add(fact.content)
            collected.extend(fresh)

            frontier = self._next_frontier(fresh, passage_ids, visited, self.beam_width)
        return collected

    @classmethod
    def _passage_grounded_first(cls, facts: list[Document], passage_ids: set[str]) -> list[Document]:
        """Stable-sort facts so those asserted by a RETRIEVED passage come first — a cheap relevance prior
        that needs no model: a fact the query-relevant chunks stated outranks a random hub-entity spur."""
        return sorted(facts, key=lambda f: 0 if (set(cls._fact_source_ids(f)) & passage_ids) else 1)

    def _rank_facts(
        self, facts: list[Document], query: str, passages: list[Document], config: RunnableConfig, **kwargs
    ) -> list[Document]:
        """Rank facts by relevance and cap to ``fact_limit`` (so a hub entity can't flood the result).

        With a ``reranker`` configured, it owns the ordering (e.g. a cross-encoder over the query + facts);
        otherwise facts asserted by a retrieved passage are preferred. Either way the list is truncated.
        """
        if not facts:
            return facts
        passage_ids = {str(p.id) for p in passages}
        if self.reranker is not None:
            rerank_out = self.reranker.run(
                input_data={"query": query, "documents": facts},
                run_depends=self._run_depends,
                config=config,
                **kwargs,
            )
            if rerank_out.status == RunnableStatus.SUCCESS:
                self._run_depends = [NodeDependency(node=self.reranker).to_dict(for_tracing=True)]
                facts = rerank_out.output.get("documents", facts)
            else:
                logger.warning(f"Tool {self.name} - {self.id}: reranker failed; using passage-grounded order.")
                facts = self._passage_grounded_first(facts, passage_ids)
        else:
            facts = self._passage_grounded_first(facts, passage_ids)
        return facts[: self.fact_limit]

    @classmethod
    def _missing_source_ids(cls, facts: list[Document], have_ids: set[str]) -> list[str]:
        """Distinct fact source-chunk ids that are NOT already among ``have_ids`` (the retrieved passages)."""
        missing: list[str] = []
        seen: set[str] = set()
        for fact in facts:
            for source_id in cls._fact_source_ids(fact):
                if source_id not in have_ids and source_id not in seen:
                    seen.add(source_id)
                    missing.append(source_id)
        return missing

    def _fetch_source_chunks(self, facts: list[Document], passages: list[Document]) -> list[Document]:
        """Fetch the source chunks for facts whose chunk wasn't already retrieved (pgvector-only capability).

        Returns [] when nothing to fetch or the vector retriever doesn't support ``get_documents_by_ids``
        (other backends): facts are then cited only to the retrieved passages.
        """
        missing = self._missing_source_ids(facts, {str(p.id) for p in passages})
        if not missing:
            return []

        fetch = getattr(self.vector_retriever.document_retriever, "get_documents_by_ids", None)
        if not callable(fetch):
            logger.info(
                f"Tool {self.name} - {self.id}: vector retriever has no get_documents_by_ids; "
                "facts cited to retrieved passages only."
            )
            return []
        try:
            return fetch(missing) or []
        except NotImplementedError:
            return []

    def execute(
        self, input_data: HybridGraphVectorRetrieverInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.reset_run_state()
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        try:
            kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            kwargs.pop("run_depends", None)

            # 1. Vector retrieval (dense + keyword hybrid over the enriched chunks).
            check_cancellation(config)
            vec_input: dict[str, Any] = {"query": input_data.query, "filters": input_data.filters}
            if input_data.top_k is not None:
                vec_input["top_k"] = input_data.top_k
            if input_data.alpha is not None:
                vec_input["alpha"] = input_data.alpha
            vec_out = self.vector_retriever.run(
                input_data=vec_input, run_depends=self._run_depends, config=config, **kwargs
            )
            if vec_out.status != RunnableStatus.SUCCESS:
                error = vec_out.error.message if vec_out.error else "unknown error"
                raise RuntimeError(f"Vector retriever failed: {error}")
            self._run_depends = [NodeDependency(node=self.vector_retriever).to_dict(for_tracing=True)]
            passages = vec_out.output.get("documents", [])

            # 2. Seed the graph with the entity ids the passages mention (Onyx-style); query fallback.
            seeds = self._seed_entity_ids(passages, self.max_seed_entities) if self.seed_graph_from_passages else []

            # 3. Expand the graph from those seeds, iterating hop-by-hop with a pruned frontier (beam) so a
            #    multi-hop reach never becomes an exploding variable-length traversal.
            facts = self._expand_graph(seeds, input_data.query, passages, config, **kwargs)

            # 4. Rank facts by relevance and cap to fact_limit — keeps the relevant facts, not an arbitrary
            #    slice of a hub entity's edges.
            facts = self._rank_facts(facts, input_data.query, passages, config, **kwargs)

            # 5. Ground each kept fact with its source chunk: cite to a retrieved passage, or fetch the chunk
            #    by id when vector search didn't surface it (the multi-hop case).
            evidence = self._fetch_source_chunks(facts, passages) if self.ground_facts_with_source_chunks else []

            logger.info(
                f"Tool {self.name} - {self.id}: fused {len(passages)} passage(s), {len(facts)} fact(s), "
                f"{len(evidence)} fetched source chunk(s)."
            )
            return self._render(passages, facts, evidence)
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: execution error: {e}", exc_info=True)
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve hybrid graph+vector context. Error: {e}.",
                recoverable=True,
            )

    @classmethod
    def _render(
        cls, passages: list[Document], facts: list[Document], evidence: list[Document] | None = None
    ) -> dict[str, Any]:
        """Sectioned output: passages (+ fetched evidence chunks) and facts CITED to their source chunk.

        Passages and evidence chunks share one ``[n]`` index space; each fact is annotated with the indices
        of the chunks it was extracted from (via ``source_doc_id``). Evidence chunks are graph-discovered
        source chunks that vector search did not return.
        """
        evidence = evidence or []
        seen_content: set[str] = set()

        unique_passages = cls._dedupe_by_content(passages, seen_content)
        unique_evidence = cls._dedupe_by_content(evidence, seen_content)  # drop any already shown as a passage
        chunks = unique_passages + unique_evidence
        id_to_index = {str(d.id): i + 1 for i, d in enumerate(chunks)}

        def cite(fact: Document) -> list[int]:
            return sorted({id_to_index[s] for s in cls._fact_source_ids(fact) if s in id_to_index})

        sections: list[str] = []
        if chunks:
            lines = []
            for i, doc in enumerate(chunks):
                tag = "" if i < len(unique_passages) else " (graph evidence)"
                lines.append(f"[{i + 1}]{tag} {doc.content}")
            sections.append("## Passages\n" + "\n".join(lines))
        if facts:
            lines = []
            for doc in facts:
                idxs = cite(doc)
                suffix = f"   (source: {', '.join(f'[{i}]' for i in idxs)})" if idxs else ""
                lines.append(f"- {doc.content}{suffix}")
            sections.append("## Facts\n" + "\n".join(lines))
        content = "\n\n".join(sections) or "No matching context found."

        documents: list[Document] = []
        for i, doc in enumerate(chunks):
            doc.metadata = {**(doc.metadata or {}), "origin": "passage" if i < len(unique_passages) else "evidence"}
            documents.append(doc)
        for doc in facts:
            doc.metadata = {**(doc.metadata or {}), "origin": "fact", "source_passages": cite(doc)}
            documents.append(doc)
        return {"content": content, "documents": documents}

    @staticmethod
    def _dedupe_by_content(documents: list[Document], seen_content: set[str]) -> list[Document]:
        """Documents with distinct, not-yet-seen content (mutates ``seen_content`` to track across calls)."""
        unique: list[Document] = []
        for doc in documents:
            if doc.content and doc.content not in seen_content:
                seen_content.add(doc.content)
                unique.append(doc)
        return unique
