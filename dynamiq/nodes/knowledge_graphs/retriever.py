import itertools
import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from dynamiq.connections import ApacheAGE, AWSNeptune, Neo4j
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.embedders.base import TextEmbedder
from dynamiq.nodes.knowledge_graphs.entity_extractor import (
    ENTITY_EMBEDDING_VECTOR_INDEX,
    ENTITY_LABEL,
    ENTITY_NAME_FULLTEXT_INDEX,
    HAS_ATTRIBUTE_TYPE,
    Ontology,
)
from dynamiq.nodes.node import ConnectionNode, Node, NodeDependency, NodeGroup, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.storages.graph.age import ApacheAgeGraphStore
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.storages.graph.neptune import NeptuneGraphStore
from dynamiq.storages.vector.utils import normalize_filters
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.json_parser import parse_llm_json_output
from dynamiq.utils.logger import logger

# Only filter KEYS are validated against this (VALUES are always bound params), so a key can't inject Cypher.
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# LLM pre-step: extract the entities a QUESTION is about (constrained to ontology types) so the graph
# seeds on those names rather than every word. Structured output is just a list of names.
_QUERY_ENTITY_PROMPT = (
    "You identify which named entities a QUESTION is about, so they can be looked up in a knowledge graph. "
    'Return ONLY a JSON object of the form {"names": ["..."]} -- the entity names exactly as they appear '
    "in the question. Only include entities whose type is one of: {{entity_types}}. Omit question words, "
    "verbs and generic nouns. Return an empty list if the question names no such entity.\n\nQuestion:\n{{query}}"
)

_QUERY_ENTITY_RESPONSE_FORMAT: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "query_entities",
        "schema": {
            "type": "object",
            "properties": {"names": {"type": "array", "items": {"type": "string"}}},
            "required": ["names"],
        },
    },
}

# Prompt for the optional `summarize` step: compose the retrieved context into a written answer.
_SUMMARIZE_PROMPT = (
    "Answer the QUESTION using ONLY the CONTEXT below (facts and/or passages retrieved from a knowledge "
    "graph). If the context is insufficient, say what is missing. Be concise.\n\n"
    "CONTEXT:\n{{context}}\n\nQUESTION:\n{{query}}"
)

# Supported comparison operators -> Cypher predicate builders. This is the SAME filter DSL the
# vector-store retrievers expose (see ``dynamiq/storages/vector/*/filters.py``), so filters are written
# identically for both retriever families. `c` is the qualified edge property (e.g. ``r.source_url``);
# `p` is the bound-parameter name.
_COMPARISON_OPERATORS: dict[str, Any] = {
    "==": lambda c, p: f"{c} = ${p}",
    "!=": lambda c, p: f"{c} <> ${p}",
    ">": lambda c, p: f"{c} > ${p}",
    ">=": lambda c, p: f"{c} >= ${p}",
    "<": lambda c, p: f"{c} < ${p}",
    "<=": lambda c, p: f"{c} <= ${p}",
    "in": lambda c, p: f"{c} IN ${p}",  # scalar property equals one of the value list
    "not in": lambda c, p: f"NOT {c} IN ${p}",
    # list-list intersection (default-deny): keep the edge only if its list property shares >=1 element
    # with the value list. Null property -> [] -> excluded. How ACL is expressed:
    # {"field": "allowed_principals", "operator": "contains_any", "value": [...]}.
    "contains_any": lambda c, p: f"size([x IN coalesce({c}, []) WHERE x IN ${p}]) > 0",
}

# Logical operators for nesting conditions (mirrors the vector-store ``LOGICAL_OPERATORS``).
_LOGICAL_OPERATORS = {"AND", "OR"}


def _validate_identifier(name: str) -> str:
    """Guard a property key that will be interpolated into Cypher text."""
    if not _IDENTIFIER_PATTERN.match(name):
        raise ValueError(
            f"KnowledgeGraphRetriever: unsafe property identifier {name!r} "
            f"(must match {_IDENTIFIER_PATTERN.pattern})."
        )
    return name


def _lucene_query(text: str) -> str:
    """Turn free text into a fuzzy Lucene OR-query (the no-extraction fallback over the raw question).

    Keeps only word tokens (so Lucene special characters can't break the parser) and appends ``~`` to
    each for edit-distance fuzzy matching. OR across tokens maximizes recall. Returns "" when there are
    no usable tokens.
    """
    terms = re.findall(r"[A-Za-z0-9]+", text or "")
    return " OR ".join(f"{t}~" for t in terms)


def _grouped_lucene_query(names: list[str]) -> str:
    """Fuzzy Lucene query: AND tokens WITHIN a name, OR ACROSS names.

    ``"Alice Smith"`` -> ``(Alice~ AND Smith~)`` (matches the full name and typos, not a one-token overlap);
    multiple names are OR-ed. Returns "" when no name yields a usable token.
    """
    groups: list[str] = []
    for name in names:
        tokens = re.findall(r"[A-Za-z0-9]+", name or "")
        if tokens:
            groups.append("(" + " AND ".join(f"{t}~" for t in tokens) + ")")
    return " OR ".join(groups)


def _compile_edge_filters(
    rel_var: str, filters: dict[str, Any] | None, *, param_prefix: str = "f"
) -> tuple[str, dict[str, Any]]:
    """Compile a structured metadata filter into a parameterized Cypher predicate on edge properties.

    Uses the SAME filter grammar as the vector-store retrievers (see ``dynamiq/storages/vector/*/filters.py``),
    so filters are written identically for both retriever families:

      - comparison: ``{"field": <name>, "operator": <op>, "value": <v>}`` where ``<op>`` is one of
        ``_COMPARISON_OPERATORS``;
      - logical:    ``{"operator": "AND"|"OR", "conditions": [ <comparison-or-logical>, ... ]}`` (nestable).

    The shared ``normalize_filters`` helper first accepts the ``{"field": value}`` shorthand as well (again,
    the same front-door the vector stores use). Field names are validated as identifiers; values are always
    bound parameters (never interpolated), so this cannot be an injection vector. Returns ``("", {})`` for an
    empty filter.
    """
    filters = normalize_filters(filters)
    if not filters:
        return "", {}

    counter = itertools.count()
    params: dict[str, Any] = {}

    def compile_node(condition: Any) -> str:
        if not isinstance(condition, dict):
            raise ValueError(f"KnowledgeGraphRetriever: filter condition must be a dict, got {type(condition)}.")
        # 'field' is only present in comparison conditions; otherwise treat it as logical (AND/OR).
        return compile_comparison(condition) if "field" in condition else compile_logical(condition)

    def compile_logical(condition: dict[str, Any]) -> str:
        if "operator" not in condition:
            raise ValueError(f"KnowledgeGraphRetriever: 'operator' key missing in {condition!r}.")
        if "conditions" not in condition:
            raise ValueError(f"KnowledgeGraphRetriever: 'conditions' key missing in {condition!r}.")
        operator = condition["operator"]
        if operator not in _LOGICAL_OPERATORS:
            raise ValueError(f"KnowledgeGraphRetriever: unsupported logical operator {operator!r}.")
        parts = [p for p in (compile_node(c) for c in condition["conditions"]) if p]
        if not parts:
            return ""
        joiner = " AND " if operator == "AND" else " OR "
        return "(" + joiner.join(parts) + ")"

    def compile_comparison(condition: dict[str, Any]) -> str:
        if "operator" not in condition:
            raise ValueError(f"KnowledgeGraphRetriever: 'operator' key missing in {condition!r}.")
        if "value" not in condition:
            raise ValueError(f"KnowledgeGraphRetriever: 'value' key missing in {condition!r}.")
        # A dict value is never meaningful (graph properties can't be maps) — it is the signature of the
        # LEGACY nested-operator grammar ({"key": {"$op": value}}), which the {"field": value} shorthand
        # would otherwise normalize into a silently-never-matching equality. Fail loudly instead.
        if isinstance(condition["value"], dict):
            raise ValueError(
                "KnowledgeGraphRetriever: dict filter values are not valid — this looks like the legacy "
                '{"key": {"$op": ...}} filter grammar; use {"field": ..., "operator": ..., "value": ...} '
                "instead (e.g. operator 'contains_any' for ACL list intersection)."
            )
        operator = condition["operator"]
        if operator not in _COMPARISON_OPERATORS:
            raise ValueError(f"KnowledgeGraphRetriever: unsupported filter operator {operator!r}.")
        column = f"{rel_var}.{_validate_identifier(condition['field'])}"
        pname = f"{param_prefix}{next(counter)}"
        params[pname] = condition["value"]
        return _COMPARISON_OPERATORS[operator](column, pname)

    return compile_node(filters), params


class GraphRetrieverInputSchema(BaseModel):
    query: str = Field(..., description="Natural-language question to retrieve graph context for.")
    top_k: int | None = Field(default=None, description="Override the node-level maximum number of facts.")
    max_hops: int | None = Field(
        default=None,
        ge=1,
        le=4,
        description="Override the node-level traversal depth: how many hops to expand from the seed "
        "entities (beam search — each hop keeps only the most relevant edges and expands from those). "
        "Use 2 for chain questions whose answer is about a neighbor of the named entity.",
    )
    entities: list[str] | None = Field(
        default=None,
        description="Optional explicit entity names to start from. When omitted, entry entities are those "
        "whose name is mentioned in the query.",
    )
    entity_ids: list[str] | None = Field(
        default=None,
        description="Optional explicit, resolved entity ids to start from (unique, variant-proof). Takes "
        "precedence over 'entities' and the query — used by hybrid retrieval to seed traversal by id.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional edge-property filters to narrow results, in the same structured format as the "
        'vector-store retrievers ({"field","operator","value"} or {"operator","conditions"}). '
        "AND-ed on top of the node's locked filters (can only further restrict, never widen).",
    )


class KnowledgeGraphRetriever(ConnectionNode):
    """Retrieves bounded, ACL-filtered context from a knowledge graph for a natural-language query.

    The graph sibling of :class:`~dynamiq.nodes.retrievers.retriever.VectorStoreRetriever`, and the
    controlled alternative to ``CypherExecutor`` for read access. An LLM first extracts the entities the
    question is about (constrained to the ontology's types), the retriever finds those entry-point entities
    by name, expands ``max_hops`` hops through **visible** edges (beam search: each hop keeps the most
    relevant edges and expands only from those), and renders the resulting facts as ``Document`` objects an
    agent can consume directly. Deeper exploration is also possible by iteration: an agent feeds a fact's
    neighbor name back as the next call's ``entities`` seed.

    Entry-point selection is backend-agnostic: on Neo4j with the entity-name full-text index present
    (created by ``KnowledgeGraphWriter``) it uses an index seek and expands only the seed's neighborhood;
    otherwise (non-Neo4j, or index missing / retrieve-before-write) it transparently falls back to a
    portable ``CONTAINS`` scan. Either way, a seed with no ACL-visible edge produces no rows, so only
    entities reachable through a visible edge are surfaced.

    Why a node (not LLM-written Cypher): filters and result bounds are compiled server-side into a single
    parameterized query. Filters use the SAME structured grammar as the vector-store retrievers — a
    comparison ``{"field": <name>, "operator": <op>, "value": <v>}`` or a logical
    ``{"operator": "AND"|"OR", "conditions": [...]}`` (nestable) — applied to edge properties. Two filter
    tiers (mirroring how ``VectorStoreRetriever`` exposes filters, but with a locked layer added):

      - ``filters`` (node config) are LOCKED — always AND-applied and NOT overridable from the input. This
        is the access-control point: express ACL here, e.g.
        ``filters={"field": "allowed_principals", "operator": "contains_any", "value": ["group:a"]}`` keeps
        only edges whose ``allowed_principals`` list shares a principal with the caller's (default-deny — a
        missing list is excluded). Because it is not on the input schema, an agent cannot drop or widen it.
      - input ``filters`` (caller/agent-supplied) are AND-ed on top — they can only further narrow.

    ACL model: this graph keeps all access metadata on EDGES; a node is visible exactly when reachable
    through a visible edge. With no locked filters, all edges are visible.

    Output: ``{"content": <newline-separated facts>, "documents": [Document, ...]}``.

    Attributes:
        connection (Neo4j | ApacheAGE | AWSNeptune): The graph backend connection.
        llm (Node): LLM used to extract the query's entities before seeding the search (required).
        ontology (Ontology): Schema whose entity types constrain query-entity extraction — pass the SAME
            ontology used at ingestion so the question is parsed for the entity kinds the graph contains.
        database (str | None): Optional target database (Neo4j).
        graph_name (str | None): Graph name (Apache AGE).
        filters (dict | None): LOCKED edge-property filters, always applied. Same structured
            ``{"field","operator","value"}`` / ``{"operator","conditions"}`` grammar as the vector-store
            retrievers. ACL lives here via the ``contains_any`` operator.
        top_k (int): Maximum number of facts to fetch from the graph. With a ``document_reranker`` this is
            the CANDIDATE pool that gets reranked; the reranker's own ``top_k`` decides the final count.
        max_hops (int): Beam-search traversal depth (default 1 = single hop, the previous behavior).
            At 2+, chain questions ("what does X's employer use?") resolve in one call: hop 1 finds
            ``X -WORKS_AT-> Acme``, hop 2 expands from Acme to ``Acme -USES-> ...``. Overridable per call.
            Portable: hop queries read endpoint ids off the bound pattern nodes (``a.id``/``b.id``),
            so multi-hop needs no dialect-specific functions on any backend.
        beam_width (int | None): Edges kept per hop when ``max_hops > 1`` (default ``top_k // max_hops``).
        document_reranker (Node | None): Optional reranker (e.g. a cross-encoder ``CohereReranker``) applied
            to the rendered facts. A high-degree (hub) entity can expand to many edges that all match the
            seed equally; the reranker scores each fact against the query and keeps the most relevant, so
            precision on hubs comes from relevance, not the position-only fallback score. Off by default.
            Over-fetch by setting this node's ``top_k`` above the reranker's ``top_k``.
    """

    group: Literal[NodeGroup.RETRIEVERS] = NodeGroup.RETRIEVERS
    action_type: ActionType = ActionType.DATABASE_QUERY
    name: str = "graph-retriever"
    description: str = (
        "Retrieves facts from a knowledge graph for a natural-language question. Input: a 'query' string "
        "(and optionally 'entities'/'entity_ids' to start from, 'top_k', or 'max_hops'). Returns related "
        "facts, one per line. Set 'max_hops': 2 for chain questions about a neighbor of the named "
        "entity (e.g. \"what does X's employer use\"); or call again with a fact's neighbor id."
    )
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    connection: Neo4j | ApacheAGE | AWSNeptune
    # Optional: only used to extract the query's entities (default seeding path) and to `summarize`. When
    # omitted, seeding falls back to the raw query; `summarize=True` then requires an llm (see validator).
    llm: Node | None = None
    # Optional: constrains query-entity extraction to these types. Only used alongside `llm`; unused when
    # seeding by entities/entity_ids/seed_by_query or with no llm.
    ontology: Ontology | None = None
    database: str | None = None
    graph_name: str | None = None
    create_graph_if_not_exists: bool = False
    filters: dict[str, Any] | None = None  # LOCKED: always applied; not overridable from the input schema
    top_k: int = 50
    # Beam-search traversal depth. 1 (default) = today's single hop. At >1, each hop keeps only the
    # `beam_width` most relevant edges and expands ONLY from their endpoints — so the frontier stays
    # bounded (no hub explosion) and hop-N candidates compete against hop-N siblings, never against
    # seed-adjacent facts that trivially sound more like the question. Every hop applies the same locked
    # ACL filters. Reliable ranking across hops needs edge embeddings (writer's `entity_embedder`).
    max_hops: int = 1
    # Edges kept per hop when max_hops > 1. None -> top_k // max_hops (total budget stays ~top_k).
    beam_width: int | None = None
    document_reranker: Node | None = None  # optional rerank of rendered facts; top_k here = candidate pool
    # Optional semantic seeding: when set AND the entity vector index exists, entry entities are found by
    # embedding the query's entity names and vector-searching, instead of the full-text/CONTAINS name match.
    # Use the SAME embedding model as the writer's `entity_embedder` so vector dimensions match.
    text_embedder: TextEmbedder | None = None
    vector_top_k: int = 5  # candidate entities pulled from the vector index per seed name
    # When True (and vector seeding is available), skip LLM entity extraction and seed the top-k entities by
    # the WHOLE question embedding — simpler, no LLM, context-preserving. Facts are then reranked by the same
    # vector. Off by default (keeps entity-anchored extraction). Requires a text_embedder + entity index.
    seed_by_query: bool = False
    # optional grounding source: any object with get_documents_by_ids(ids) -> list[Document]
    document_retriever: Any = None
    # when True, LLM-compose an answer from the retrieved context; when False (default) return raw facts/source
    summarize: bool = False

    input_schema: ClassVar[type[GraphRetrieverInputSchema]] = GraphRetrieverInputSchema

    _graph_store: BaseGraphStore | None = PrivateAttr(default=None)
    _use_fulltext: bool = PrivateAttr(default=False)
    _use_vector: bool = PrivateAttr(default=False)
    _run_depends: list = PrivateAttr(default_factory=list)

    @model_validator(mode="after")
    def _validate_llm_requirements(self):
        # `summarize` composes an answer with the llm, so it can't run without one. The default entity
        # extraction path also needs an llm, but that degrades gracefully to raw-query seeding (see
        # `_seed_entity_names`), so it isn't enforced here.
        if self.summarize and self.llm is None:
            raise ValueError("KnowledgeGraphRetriever: `summarize=True` requires an `llm`.")
        return self

    def reset_run_state(self) -> None:
        self._run_depends = []

    @property
    def to_dict_exclude_params(self) -> dict:
        return super().to_dict_exclude_params | {
            "llm": True,
            "document_reranker": True,
            "document_retriever": True,
            "text_embedder": True,
        }

    def to_dict(self, **kwargs) -> dict:
        data = super().to_dict(**kwargs)
        if self.llm:
            data["llm"] = self.llm.to_dict(**kwargs)
        if self.document_reranker:
            data["document_reranker"] = self.document_reranker.to_dict(**kwargs)
        if self.text_embedder:
            data["text_embedder"] = self.text_embedder.to_dict(**kwargs)
        if self.document_retriever is not None and hasattr(self.document_retriever, "to_dict"):
            data["document_retriever"] = self.document_retriever.to_dict(**kwargs)
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Select the concrete graph store from the connection type (same dispatch as CypherExecutor)."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.llm and self.llm.is_postponed_component_init:
            self.llm.init_components(connection_manager)
        if self.document_reranker and self.document_reranker.is_postponed_component_init:
            self.document_reranker.init_components(connection_manager)
        if self.document_retriever is not None and getattr(
            self.document_retriever, "is_postponed_component_init", False
        ):
            self.document_retriever.init_components(connection_manager)
        if self.text_embedder and self.text_embedder.is_postponed_component_init:
            self.text_embedder.init_components(connection_manager)
        if self._graph_store is None:
            self._graph_store = self._build_graph_store()
        self._use_fulltext = self._probe_fulltext()
        self._use_vector = self._probe_vector()

    def _probe_fulltext(self) -> bool:
        """Whether the entity-name full-text index exists (Neo4j only). Read-only; failure -> scan fallback.

        Backend-agnostic: non-Neo4j stores (and a Neo4j without the index yet, e.g. retrieve-before-write)
        return ``False``, so the retriever transparently uses the portable ``CONTAINS`` scan instead.
        """
        if not isinstance(self._graph_store, Neo4jGraphStore):
            return False
        try:
            records, _, _ = self._graph_store.run_cypher(
                "SHOW INDEXES YIELD name, type WHERE name = $n AND type = 'FULLTEXT' RETURN count(*) AS c",
                parameters={"n": ENTITY_NAME_FULLTEXT_INDEX},
                database=self.database,
            )
            rows = self._graph_store.format_records(records)
            return bool(rows and (rows[0].get("c") or 0) > 0)
        except Exception as e:
            logger.warning(f"Node {self.name} - {self.id}: full-text index probe failed, using scan: {e}")
            return False

    def _probe_vector(self) -> bool:
        """Whether semantic seeding is available: a ``text_embedder`` is set AND the entity vector index
        exists (Neo4j only). Read-only; any failure -> ``False`` so the retriever falls back to full-text/scan.

        Backend-agnostic: without an embedder, or on a non-Neo4j store, or before the writer has embedded any
        entity (so no vector index), this returns ``False`` and the existing name-match path is used.
        """
        if not self.text_embedder or not isinstance(self._graph_store, Neo4jGraphStore):
            return False
        try:
            records, _, _ = self._graph_store.run_cypher(
                "SHOW INDEXES YIELD name, type WHERE name = $n AND type = 'VECTOR' RETURN count(*) AS c",
                parameters={"n": ENTITY_EMBEDDING_VECTOR_INDEX},
                database=self.database,
            )
            rows = self._graph_store.format_records(records)
            return bool(rows and (rows[0].get("c") or 0) > 0)
        except Exception as e:
            logger.warning(f"Node {self.name} - {self.id}: vector index probe failed, using name match: {e}")
            return False

    def _build_graph_store(self) -> BaseGraphStore:
        if isinstance(self.connection, ApacheAGE):
            return ApacheAgeGraphStore(
                connection=self.connection,
                client=self.client,
                graph_name=self.graph_name,
                create_graph_if_not_exists=self.create_graph_if_not_exists,
            )
        if isinstance(self.connection, AWSNeptune):
            return NeptuneGraphStore(
                connection=self.connection,
                client=self.client,
                endpoint=self.connection.endpoint,
                verify_ssl=self.connection.verify_ssl,
                timeout=self.connection.timeout,
            )
        return Neo4jGraphStore(connection=self.connection, client=self.client, database=self.database)

    def ensure_client(self) -> None:
        """Keep the graph store's client in sync if the connection node reconnects."""
        previous_client = self.client
        super().ensure_client()
        if self.client is previous_client or not self._graph_store:
            return
        if getattr(self._graph_store, "client", None) is not self.client:
            self._graph_store.update_client(self.client)

    def _extract_entity_names(self, query: str, config: RunnableConfig, **kwargs) -> list[str]:
        """Extract the entity names a question is about via the LLM, constrained to the ontology's types.

        Returns ``[]`` on any failure (non-SUCCESS status, unparseable output, exception) so the search
        degrades to seeding on the raw query rather than erroring.
        """
        try:
            prompt = Prompt(messages=[Message(role="user", content=_QUERY_ENTITY_PROMPT)])
            run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            run_kwargs.pop("run_depends", None)
            result = self.llm.run(
                input_data={
                    "query": query,
                    "entity_types": ", ".join(self.ontology.entity_types if self.ontology else []),
                },
                prompt=prompt,
                response_format=_QUERY_ENTITY_RESPONSE_FORMAT,
                config=config,
                run_depends=self._run_depends,
                **run_kwargs,
            )
            self._run_depends = [NodeDependency(node=self.llm).to_dict(for_tracing=True)]
            if result.status != RunnableStatus.SUCCESS:
                logger.warning(f"Node {self.name} - {self.id}: query entity extraction failed; using raw query.")
                return []
            parsed = parse_llm_json_output(result.output.get("content") or "")
            names = parsed.get("names") if isinstance(parsed, dict) else None
            return [n.strip() for n in (names or []) if isinstance(n, str) and n.strip()]
        except Exception as e:
            logger.warning(f"Node {self.name} - {self.id}: query entity extraction error: {e}; using raw query.")
            return []

    def _seed_entity_names(self, input_data: GraphRetrieverInputSchema, config: RunnableConfig, **kwargs) -> list[str]:
        """The entity names the search seeds on, matched fuzzily (same path as extracted names).

        - explicit ``entities`` -> used as the seed names directly, **skipping LLM extraction**.
        - explicit ``entity_ids`` -> ``[]`` (anchored exactly by id in ``_entry``, not by name).
        - ``seed_by_query`` -> ``[]`` (no names; seed on the WHOLE question vector, no LLM extraction).
        - no ``llm`` set -> ``[]`` (no extractor; seed on the raw query via ``_entry``'s no-names fallback).
        - otherwise -> extracted from the question by the LLM.

        An empty list means "no seed names" -> ``_entry`` seeds on the whole-query vector (or the raw query).
        """
        if input_data.entity_ids or self.seed_by_query:
            return []
        if input_data.entities:
            return input_data.entities
        if self.llm is None:
            return []
        return self._extract_entity_names(input_data.query, config, **kwargs)

    def _embed_query(self, text: str, config: RunnableConfig, **kwargs) -> list[float] | None:
        """Embed one text via ``text_embedder``; ``None`` on any failure. One call, reused for the whole-query
        rank vector and per-name seed vectors."""
        try:
            embed_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            embed_kwargs.pop("run_depends", None)
            result = self.text_embedder.run(
                input_data={"query": text}, run_depends=self._run_depends, config=config, **embed_kwargs
            )
            if result.status != RunnableStatus.SUCCESS:
                return None
            self._run_depends = [NodeDependency(node=self.text_embedder).to_dict(for_tracing=True)]
            return result.output.get("embedding") or None
        except Exception as e:
            logger.warning(f"Node {self.name} - {self.id}: query embed failed: {e}")
            return None

    def _query_vector(self, query: str, config: RunnableConfig, **kwargs) -> list[float] | None:
        """Embed the WHOLE query ONCE (when ``_use_vector``) → reused for BOTH server-side fact reranking
        (``$qvec``) and the no-entity-names seed vector (incl. ``seed_by_query``). ``None`` disables both.

        ``_use_vector`` is true only when the entity vector index exists, which — created with the 5.15+
        ``CREATE VECTOR INDEX`` syntax — also guarantees the ``vector.similarity.cosine`` function the rank
        uses. Best-effort: any failure returns ``None``."""
        if not self._use_vector or not query:
            return None
        return self._embed_query(query, config, **kwargs)

    def _seed_vectors(
        self,
        input_data: GraphRetrieverInputSchema,
        seed_names: list[str],
        query_vector: list[float] | None,
        config: RunnableConfig,
        **kwargs,
    ) -> list[list[float]] | None:
        """Vectors to seed entity search on, or ``None`` to fall back to the full-text/scan path.

        ``None`` when semantic seeding is off (``_use_vector`` False) or the search is anchored by explicit
        ``entity_ids``. With entity names, each is embedded separately (a multi-entity question seeds on each
        independently); with NO names (extraction found none, or ``seed_by_query``), the already-computed
        ``query_vector`` is reused — the whole query is embedded once and serves both seeding and reranking.
        Any embedder failure → ``None`` (name match).
        """
        if not self._use_vector or input_data.entity_ids:
            return None
        if not seed_names:
            return [query_vector] if query_vector else None
        vectors: list[list[float]] = []
        for name in seed_names:
            vector = self._embed_query(name, config, **kwargs)
            if vector is None:
                logger.warning(f"Node {self.name} - {self.id}: seed-name embed failed; using name match.")
                return None
            vectors.append(vector)
        return vectors or None

    def _entry(
        self,
        input_data: GraphRetrieverInputSchema,
        params: dict[str, Any],
        seed_names: list[str] | None,
        seed_vectors: list[list[float]] | None = None,
    ) -> tuple[list[str], str | None, bool]:
        """Pick the entry-point strategy. Returns (lead_lines, entry_where, anchored).

        - explicit ``entity_ids`` -> anchor on ``:Entity`` nodes by exact id (the precise hybrid/iterative
          seed; takes precedence and skips name matching entirely).
        - else, ``seed_vectors`` present (a ``text_embedder`` is set and the entity vector index exists)
          -> anchor on the nearest entities by embedding similarity (``db.index.vector.queryNodes``), one
          lookup per seed vector. Semantic seeding: matches "car" to an entity named "automobile".
        - else, seed by NAME (``seed_names`` = explicit ``entities`` or LLM-extracted names), matched
          fuzzily: on Neo4j with the index, a full-text seek over ``(tok AND tok) OR (...)``; otherwise a
          portable ``CONTAINS`` scan. With no names, falls back to a recall-y OR over the raw query.

        Anchored modes bind ``a`` to a seed and expand from it (cheap, neighborhood-bounded); the scan
        binds nothing and filters every edge by name (the fallback). Either way an entity with no
        ACL-visible edge yields no rows — so seeds without a visible edge drop out for free.
        """
        if input_data.entity_ids:
            params["entity_ids"] = input_data.entity_ids
            return [f"MATCH (a:{ENTITY_LABEL})"], "a.id IN $entity_ids", True

        if seed_vectors:
            params["qvecs"] = seed_vectors
            params["vk"] = self.vector_top_k
            return (
                [
                    "UNWIND $qvecs AS qv",
                    f"CALL db.index.vector.queryNodes('{ENTITY_EMBEDDING_VECTOR_INDEX}', $vk, qv) YIELD node AS a",
                ],
                None,
                True,
            )

        # Explicit entities and extracted names are matched the same way; `seed_names` carries either.
        # (`seed_names is None` only on direct _build_query calls -> derive from the input's entities.)
        names = seed_names if seed_names is not None else input_data.entities
        if self._use_fulltext:
            # Precise: AND tokens within each name, OR across names. No names -> recall-y OR over the query.
            lucene = _grouped_lucene_query(names) if names else _lucene_query(input_data.query)
            if lucene:
                params["q"] = lucene
                return (
                    [f"CALL db.index.fulltext.queryNodes('{ENTITY_NAME_FULLTEXT_INDEX}', $q) YIELD node AS a"],
                    None,
                    True,
                )

        params["q"] = " ".join(names) if names else input_data.query
        return [], "(toLower($q) CONTAINS toLower(a.name) OR toLower($q) CONTAINS toLower(b.name))", False

    def _build_query(
        self,
        input_data: GraphRetrieverInputSchema,
        limit: int,
        seed_names: list[str] | None = None,
        seed_vectors: list[list[float]] | None = None,
        query_vector: list[float] | None = None,
        include_endpoint_ids: bool = False,
    ) -> tuple[str, dict[str, Any]]:
        """Build the single parameterized one-hop Cypher query and its parameters.

        ``seed_names`` are the extracted entity names computed in ``execute``; ``None`` (the default when
        called directly) means seed on the raw query (the pre-extraction behavior). ``seed_vectors`` are the
        embedded seed names (when semantic seeding is active) — when present they take precedence over names.
        ``query_vector`` is the embedded query — when present the neighbourhood is ranked server-side by the
        cosine of each edge's ``r.embedding`` so only the top ``limit`` MOST RELEVANT facts come back (a hub
        no longer returns an arbitrary slice), and the embeddings never cross the wire.
        ``include_endpoint_ids`` additionally returns each edge's endpoint entity ids (``a_id``/``b_id``,
        same shape as ``_hop_query``) — required when this hop seeds a multi-hop frontier.
        """
        params: dict[str, Any] = {"limit": limit}
        lead, entry_where, anchored = self._entry(input_data, params, seed_names, seed_vectors)
        # Anchored expansion is undirected (catch edges into and out of the seed); each edge's direction is
        # carried by its own r.src_name/r.dst_name snapshot. The scan keeps the directed pattern with
        # a=source, b=target.
        arrow = "-[{rel}]-" if anchored else "-[{rel}]->"

        rel_clause, filter_params = self._edge_predicates("r", input_data.filters)
        params.update(filter_params)
        where = " AND ".join([t for t in [entry_where, rel_clause] if t])
        # Names come from the edge's own ACL-bearing snapshot (r.src_name/r.dst_name), never the shared
        # merged node, so a differently-scoped name can't leak.
        if include_endpoint_ids:
            # Endpoint ids feed _endpoint_ids -> the hop-2 frontier; keep the shape identical to _hop_query.
            # anchor_ids are the SEED endpoint(s) of the row, so the hop-2 frontier can exclude the seeds —
            # otherwise hop 2 re-expands them and their leftover 1-hop edges (the ones hop 1's beam cut)
            # would compete against true chain facts. Anchored modes bind `a` to the seed by construction;
            # the CONTAINS scan matches the seed on EITHER endpoint, so the anchor is recomputed per row
            # with the same predicate the scan matched on (both endpoints can be seeds -> a list).
            anchor_expr = (
                "[a.id]"
                if anchored
                else "[x IN [a, b] WHERE x.name IS NOT NULL AND toLower($q) CONTAINS toLower(x.name) | x.id]"
            )
            # Endpoint ids read straight off the bound pattern nodes — plain property access, portable
            # to every openCypher backend (no startNode()/endNode(), which AGE/Neptune lack). a_id/b_id
            # feed set-based frontier building only, so they need not align with src/dst direction (on
            # anchored entry `a` is the seed side, whichever direction the edge points).
            ret = (
                "RETURN r.src_name AS a_name, a.id AS a_id, type(r) AS rel, "
                "properties(r) AS rprops, r.dst_name AS b_name, b.id AS b_id, "
                f"{anchor_expr} AS anchor_ids"
            )
        else:
            ret = "RETURN r.src_name AS a_name, type(r) AS rel, " "properties(r) AS rprops, r.dst_name AS b_name"
        lines = [*lead, f"MATCH (a){arrow.format(rel='r')}(b)"]
        if where:
            lines.append(f"WHERE {where}")
        lines.append(ret)
        if query_vector:
            # Rank the whole matched neighbourhood by fact relevance, then LIMIT — so the top_k returned are
            # the MOST relevant, not an arbitrary slice. Edges without an embedding sort last (score -1).
            params["qvec"] = query_vector
            lines.append(
                "ORDER BY CASE WHEN r.embedding IS NULL THEN -1.0 "
                "ELSE vector.similarity.cosine(r.embedding, $qvec) END DESC"
            )
        lines.append("LIMIT $limit")
        return "\n".join(lines), params

    def _edge_predicates(self, rel_var: str, user_filters: dict[str, Any] | None) -> tuple[str, dict[str, Any]]:
        """Compile the locked node filters + the caller's filters for an edge variable, AND-ed together.

        Locked (node) filters always apply; caller filters can only narrow further. Distinct parameter
        prefixes ("lf"/"uf") keep the two sets from colliding. Returns a single combined WHERE fragment
        ("" when neither is set) plus its bound parameters.
        """
        locked_clause, locked_params = _compile_edge_filters(rel_var, self.filters, param_prefix="lf")
        user_clause, user_params = _compile_edge_filters(rel_var, user_filters, param_prefix="uf")
        combined = " AND ".join(c for c in (locked_clause, user_clause) if c)
        return combined, {**locked_params, **user_params}

    def _hop_query(
        self,
        frontier_ids: list[str],
        visited_ids: list[str],
        limit: int,
        user_filters: dict[str, Any] | None,
        query_vector: list[float] | None,
    ) -> tuple[str, dict[str, Any]]:
        """One beam-search expansion step: the ACL-filtered edges of the frontier entities, ranked by fact
        relevance and cut to the per-hop beam.

        The ``NOT b.id IN $visited`` guard keeps frontier->NEW edges (the walk) while dropping the edges
        we arrived by. The SAME locked filters as hop 1 apply, so ACL holds on every hop. Ranking uses the
        same edge-embedding cosine as hop 1 — the cut happens among same-hop siblings, so a chain answer
        never competes against seed-adjacent facts that sound more like the question.
        """
        params: dict[str, Any] = {"frontier": frontier_ids, "visited": visited_ids, "limit": limit}
        rel_clause, filter_params = self._edge_predicates("r", user_filters)
        params.update(filter_params)
        # `a` is bound from $frontier and frontier ⊆ visited (execute maintains that invariant), so
        # "neither endpoint may be already-visited twice over" reduces to: b must be NEW. This drops the
        # arrived-by edges AND edges between two frontier nodes, and means no edge can bind from both
        # ends and survive — so no DISTINCT is needed and endpoint ids read straight off the bound
        # pattern nodes (a.id/b.id): plain property access, portable to every openCypher backend
        # (startNode()/endNode() are Neo4j dialect; AGE/Neptune lack a usable equivalent).
        guard = "NOT b.id IN $visited"
        where = " AND ".join([t for t in [guard, rel_clause] if t])
        lines = [
            f"MATCH (a:{ENTITY_LABEL}) WHERE a.id IN $frontier",
            "MATCH (a)-[r]-(b)",
            f"WHERE {where}",
            "RETURN r.src_name AS a_name, a.id AS a_id, type(r) AS rel, "
            "properties(r) AS rprops, r.dst_name AS b_name, b.id AS b_id",
        ]
        if query_vector:
            params["qvec"] = query_vector
            lines.append(
                "ORDER BY CASE WHEN r.embedding IS NULL THEN -1.0 "
                "ELSE vector.similarity.cosine(r.embedding, $qvec) END DESC"
            )
        lines.append("LIMIT $limit")
        return "\n".join(lines), params

    @staticmethod
    def _endpoint_ids(rows: list[dict[str, Any]]) -> set[str]:
        """The distinct entity ids on either end of the returned edges (next hop's frontier candidates)."""
        return {rid for row in rows for rid in (row.get("a_id"), row.get("b_id")) if rid}

    def execute(self, input_data: GraphRetrieverInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """Retrieve filtered graph context for the query and render it as Documents."""
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        self.reset_run_state()
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not self._graph_store:
            raise ToolExecutionException("KnowledgeGraphRetriever: graph store is not initialized.", recoverable=True)

        limit = input_data.top_k or self.top_k
        max_hops = input_data.max_hops or self.max_hops
        # Per-hop beam: with multi-hop the budget is split across hops so the total stays ~top_k, and the
        # LIMIT cut at every hop happens among same-hop siblings (see _hop_query).
        hop_limit = limit if max_hops <= 1 else (self.beam_width or max(1, limit // max_hops))
        try:
            seed_names = self._seed_entity_names(input_data, config, **kwargs)
            # Embed the whole question once (when vectors are on): reused as the fact-rank vector AND, when
            # there are no seed names, the seed vector — so seeding and reranking share one embedding.
            query_vector = self._query_vector(input_data.query, config, **kwargs)
            seed_vectors = self._seed_vectors(input_data, seed_names, query_vector, config, **kwargs)
            query, params = self._build_query(
                input_data, hop_limit, seed_names, seed_vectors, query_vector, include_endpoint_ids=max_hops > 1
            )
            records, _, _ = self._graph_store.run_cypher(query, parameters=params, database=self.database)
            rows = self._graph_store.format_records(records)
            # Beam expansion: hop N+1 expands ONLY from the NEW nodes the previous hop reached, excluding
            # edges already inside the visited set. The seeds (anchors) are excluded from the first
            # frontier — hop 1 already expanded them, and re-expanding would let their leftover 1-hop
            # edges compete against true chain facts. ACL-filtered and relevance-ranked per hop.
            anchor_ids = {aid for row in rows for aid in (row.get("anchor_ids") or []) if aid}
            endpoint_ids = self._endpoint_ids(rows)
            visited = endpoint_ids | anchor_ids
            frontier = endpoint_ids - anchor_ids
            for _ in range(max_hops - 1):
                if not frontier:
                    break
                check_cancellation(config)
                hop_query, hop_params = self._hop_query(
                    sorted(frontier), sorted(visited), hop_limit, input_data.filters, query_vector
                )
                records, _, _ = self._graph_store.run_cypher(hop_query, parameters=hop_params, database=self.database)
                hop_rows = self._graph_store.format_records(records)
                if not hop_rows:
                    break
                rows.extend(hop_rows)
                endpoint_ids = self._endpoint_ids(hop_rows)
                frontier = endpoint_ids - visited
                visited |= endpoint_ids
            documents = self._render_single_hop(rows)
            logger.info(f"Tool {self.name} - {self.id}: retrieved {len(documents)} fact(s).")
            # Rerank BEFORE enforcing top_k: when multi-hop over-fetches (explicit beam_width or the
            # >=1-per-hop floor), the reranker must see the WHOLE pool so a deep chain fact is kept or
            # dropped by RELEVANCE. The cap then trims the (reranked, else hop-ordered) list — so without
            # a reranker the cut still drops deepest facts first.
            documents = self._maybe_rerank(documents, input_data.query, config, **kwargs)
            if len(documents) > limit:
                documents = documents[:limit]
            facts = "\n".join(d.content for d in documents) or "No matching facts found."
            source_documents = self._fetch_source_documents(documents)
            # `facts` (triples) and `source_documents` (passages) are ALWAYS present so consumers never infer.
            output = {"content": facts, "facts": facts, "documents": documents, "source_documents": source_documents}
            if source_documents:
                # `content` prefers verbatim source text. ACL-safe: each source_doc_id came from an ACL-visible edge.
                output["content"] = "\n\n".join(d.content for d in source_documents if d.content) or facts
            if self.summarize and documents:
                # keep the raw retrieval under `context`; `content` becomes the composed answer
                output["context"] = output["content"]
                output["content"] = self._summarize(input_data.query, output["context"], config, **kwargs)
            return output
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: execution error: {e}", exc_info=True)
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve graph context. Error: {e}.", recoverable=True
            )

    def _maybe_rerank(self, documents: list[Document], query: str, config: RunnableConfig, **kwargs) -> list[Document]:
        """Refine the rendered facts with the optional reranker; the candidate facts ARE the documents.

        The reranker scores each fact against the query and returns its own ``top_k`` -- so a hub's many
        equally-seed-matching edges get ordered by relevance instead of the position-only fallback score.
        Reranking is a refinement, not a hard dependency: a reranker failure degrades to the unranked facts
        (mirrors how query-entity extraction degrades), so an optional precision step never fails the read.
        """
        if not self.document_reranker or not documents:
            return documents
        check_cancellation(config)
        before = len(documents)
        result = self.document_reranker.run(
            input_data={"query": query, "documents": documents},
            run_depends=self._run_depends,
            config=config,
            **kwargs,
        )
        self._run_depends = [NodeDependency(node=self.document_reranker).to_dict(for_tracing=True)]
        if result.status != RunnableStatus.SUCCESS:
            logger.warning(f"Tool {self.name} - {self.id}: reranker failed; returning unranked facts.")
            return documents
        reranked = result.output.get("documents", documents)
        logger.info(f"Tool {self.name} - {self.id}: reranked {before} -> {len(reranked)} fact(s).")
        return reranked

    def _fetch_source_documents(self, documents: list[Document]) -> list[Document]:
        """Fetch the verbatim source documents behind the retrieved facts (optional grounding).

        Each fact came from an ACL-visible edge whose ACL equals its source document's, so the caller is
        already entitled to those documents; the distinct ``source_doc_ids`` (every per-document edge behind
        a deduped fact) are pulled via the ``document_retriever``'s ``get_documents_by_id`` with no extra
        ACL check. Returns ``[]`` when no retriever is set, no fact carries provenance, the backend can't
        fetch by id, or the fetch fails.
        """
        if not self.document_retriever or not hasattr(self.document_retriever, "get_documents_by_id"):
            return []
        ids: list[str] = []
        for d in documents:
            md = d.metadata
            ids.extend(md.get("source_doc_ids") or ([md["source_doc_id"]] if md.get("source_doc_id") else []))
        ids = list(dict.fromkeys(ids))  # dedupe across facts, preserve order
        if not ids:
            return []
        try:
            return self.document_retriever.get_documents_by_id(ids)
        except Exception as e:
            logger.warning(f"Tool {self.name} - {self.id}: source-document fetch failed: {e}")
            return []

    def _summarize(self, query: str, context: str, config: RunnableConfig, **kwargs) -> str:
        """LLM-compose an answer from the retrieved context, reusing this node's ``llm``.

        Degrades to the raw context on any failure, so this optional step never fails the read.
        """
        try:
            prompt = Prompt(messages=[Message(role="user", content=_SUMMARIZE_PROMPT)])
            run_kwargs = kwargs | {"parent_run_id": kwargs.get("run_id")}
            run_kwargs.pop("run_depends", None)
            result = self.llm.run(
                input_data={"query": query, "context": context},
                prompt=prompt,
                config=config,
                run_depends=self._run_depends,
                **run_kwargs,
            )
            self._run_depends = [NodeDependency(node=self.llm).to_dict(for_tracing=True)]
            if result.status != RunnableStatus.SUCCESS:
                logger.warning(f"Tool {self.name} - {self.id}: summarization failed; returning raw context.")
                return context
            return result.output.get("content") or context
        except Exception as e:
            logger.warning(f"Tool {self.name} - {self.id}: summarization error: {e}; returning raw context.")
            return context

    @staticmethod
    def _render_single_hop(rows: list[dict[str, Any]]) -> list[Document]:
        documents: list[Document] = []
        # The same fact can arrive via several per-document edges -> show it once, but MERGE every edge's
        # source_doc_id onto the kept fact so grounding fetches all the source chunks, not just the first.
        seen: dict[str, Document] = {}
        for rank, row in enumerate(rows):
            source, rel, target = row.get("a_name"), row.get("rel"), row.get("b_name")
            if source is None or target is None or not rel:
                continue
            rprops = dict(row.get("rprops") or {})
            # already surfaced as source/target below -> drop from raw props to avoid metadata duplication.
            rprops.pop("src_name", None)
            rprops.pop("dst_name", None)
            rprops.pop("embedding", None)  # edge embedding is used server-side for ranking, never surfaced
            # Attribute edges reify a "key -> value" pair; HAS_ATTRIBUTE is just the bookkeeping type, so
            # render the attribute KEY as the relation -- otherwise the fact is a bare value ("$250,000")
            # with no hint of WHICH attribute it is.
            rel_label = rprops["key"] if rel == HAS_ATTRIBUTE_TYPE and rprops.get("key") else rel
            fact = f"{source} -[{rel_label}]-> {target}"
            # An edge description (when the extractor captured one) enriches the bare type with detail it
            # cannot convey -- append it so it reaches the answer, not just the metadata.
            if rprops.get("description"):
                fact = f"{fact}: {rprops['description']}"
            source_doc_id = rprops.get("source_doc_id")
            if fact in seen:
                # duplicate fact from another per-document edge: keep its source doc for grounding
                if source_doc_id:
                    doc_ids = seen[fact].metadata.setdefault("source_doc_ids", [])
                    if source_doc_id not in doc_ids:
                        doc_ids.append(source_doc_id)
                continue
            metadata = {"source": source, "target": target, "rel": rel, **rprops}
            # Normalize provenance to a list so later duplicate facts can accumulate every source doc.
            if source_doc_id:
                metadata["source_doc_ids"] = [source_doc_id]
            document = Document(content=fact, metadata=metadata, score=1.0 / (1.0 + rank))
            seen[fact] = document
            documents.append(document)
        return documents
