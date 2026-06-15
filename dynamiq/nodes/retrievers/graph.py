import re
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, PrivateAttr

from dynamiq.connections import ApacheAGE, AWSNeptune, Neo4j
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import ErrorHandling
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.extractors.entity_extractor import ENTITY_LABEL, ENTITY_NAME_FULLTEXT_INDEX
from dynamiq.nodes.node import ConnectionNode, NodeGroup, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.graph.age import ApacheAgeGraphStore
from dynamiq.storages.graph.base import BaseGraphStore
from dynamiq.storages.graph.neo4j import Neo4jGraphStore
from dynamiq.storages.graph.neptune import NeptuneGraphStore
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

# Property keys interpolated into Cypher text (label/property identifiers) must match this — the safe
# identifier subset shared by openCypher backends. Filter VALUES are always passed as bound parameters;
# only KEYS are validated here, so a malicious filter key cannot smuggle Cypher.
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Hard ceiling on traversal depth — keeps the variable-length expansion (and the result size) bounded.
_MAX_TRAVERSAL_DEPTH = 5

# Supported edge-filter operators -> Cypher predicate builders. `c` is the qualified property
# (e.g. ``r.source_url``); `p` is the bound-parameter name.
_FILTER_OPERATORS: dict[str, Any] = {
    "$eq": lambda c, p: f"{c} = ${p}",
    "$ne": lambda c, p: f"{c} <> ${p}",
    "$in": lambda c, p: f"{c} IN ${p}",
    "$nin": lambda c, p: f"NOT {c} IN ${p}",
    "$gt": lambda c, p: f"{c} > ${p}",
    "$gte": lambda c, p: f"{c} >= ${p}",
    "$lt": lambda c, p: f"{c} < ${p}",
    "$lte": lambda c, p: f"{c} <= ${p}",
    "$contains": lambda c, p: f"{c} CONTAINS ${p}",  # substring (string properties)
    "$any": lambda c, p: f"${p} IN {c}",  # membership (scalar value in a list property)
    # list-list intersection (default-deny): keep the edge only if its list property shares >=1 element
    # with the parameter list. This is how ACL is expressed — `{"allowed_principals": {"$intersects": [...]}}`
    # — a missing/null property coalesces to [] and is therefore excluded.
    "$intersects": lambda c, p: f"size([x IN coalesce({c}, []) WHERE x IN ${p}]) > 0",
}


def _validate_identifier(name: str) -> str:
    """Guard a property key that will be interpolated into Cypher text."""
    if not _IDENTIFIER_PATTERN.match(name):
        raise ValueError(f"GraphRetriever: unsafe property identifier {name!r} (must match {_IDENTIFIER_PATTERN.pattern}).")
    return name


def _lucene_query(text: str) -> str:
    """Turn a free-text question into a fuzzy Lucene OR-query over entity names.

    Keeps only word tokens (so Lucene special characters can't break the parser) and appends ``~`` to
    each for edit-distance fuzzy matching. Returns "" when there are no usable tokens.
    """
    terms = re.findall(r"[A-Za-z0-9]+", text or "")
    return " OR ".join(f"{t}~" for t in terms)


def _compile_edge_filters(
    rel_var: str, filters: dict[str, Any] | None, *, param_prefix: str = "f"
) -> tuple[list[str], dict[str, Any]]:
    """Compile a metadata filter dict into parameterized Cypher predicates on edge properties.

    Each entry is either ``{"key": value}`` (equality) or ``{"key": {"$op": value, ...}}`` using the
    operators in ``_FILTER_OPERATORS``. Keys are validated as identifiers; values are always bound
    parameters (never interpolated), so this cannot be an injection vector.
    """
    clauses: list[str] = []
    params: dict[str, Any] = {}
    for i, (key, condition) in enumerate(sorted((filters or {}).items())):
        column = f"{rel_var}.{_validate_identifier(key)}"
        if isinstance(condition, dict):
            for op, value in condition.items():
                if op not in _FILTER_OPERATORS:
                    raise ValueError(f"GraphRetriever: unsupported filter operator {op!r}.")
                pname = f"{param_prefix}{i}_{op.lstrip('$')}"
                clauses.append(_FILTER_OPERATORS[op](column, pname))
                params[pname] = value
        else:
            pname = f"{param_prefix}{i}"
            clauses.append(f"{column} = ${pname}")
            params[pname] = condition
    return clauses, params


class GraphRetrieverInputSchema(BaseModel):
    query: str = Field(..., description="Natural-language question to retrieve graph context for.")
    top_k: int | None = Field(default=None, description="Override the node-level maximum number of facts.")
    entities: list[str] | None = Field(
        default=None,
        description="Optional explicit entity names to start from. When omitted, entry entities are those "
        "whose name is mentioned in the query.",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Optional edge-property filters to narrow results. AND-ed on top of the node's locked "
        "filters (can only further restrict, never widen).",
    )


class GraphRetriever(ConnectionNode):
    """Retrieves bounded, ACL-filtered context from a knowledge graph for a natural-language query.

    The graph sibling of :class:`~dynamiq.nodes.retrievers.retriever.VectorStoreRetriever`, and the
    controlled alternative to ``CypherExecutor`` for read access: it finds entry-point entities by name,
    expands one or more hops through **visible** edges, and renders the resulting facts as ``Document``
    objects an agent can consume directly.

    Entry-point selection is backend-agnostic: on Neo4j with the entity-name full-text index present
    (created by ``KnowledgeGraphWriter``) it uses an index seek and expands only the seed's neighborhood;
    otherwise (non-Neo4j, or index missing / retrieve-before-write) it transparently falls back to a
    portable ``CONTAINS`` scan. Either way, a seed with no ACL-visible edge produces no rows, so only
    entities reachable through a visible edge are surfaced.

    Why a node (not LLM-written Cypher): filters and result bounds are compiled server-side into a single
    parameterized query. Two filter tiers (mirroring how ``VectorStoreRetriever`` exposes filters, but with
    a locked layer added):

      - ``filters`` (node config) are LOCKED — always AND-applied and NOT overridable from the input. This
        is the access-control point: express ACL here, e.g.
        ``filters={"allowed_principals": {"$intersects": ["group:a"]}}`` keeps only edges whose
        ``allowed_principals`` list shares a principal with the caller's (default-deny — a missing list is
        excluded). Because it is not on the input schema, an agent cannot drop or widen it.
      - input ``filters`` (caller/agent-supplied) are AND-ed on top — they can only further narrow.

    ACL model: this graph keeps all access metadata on EDGES; a node is visible exactly when reachable
    through a visible edge. With no locked filters, all edges are visible.

    Output: ``{"content": <bullet list of facts>, "documents": [Document, ...]}``.

    Attributes:
        connection (Neo4j | ApacheAGE | AWSNeptune): The graph backend connection.
        database (str | None): Optional target database (Neo4j).
        graph_name (str | None): Graph name (Apache AGE).
        filters (dict | None): LOCKED edge-property filters, always applied (operator grammar). ACL lives
            here via the ``$intersects`` operator.
        max_depth (int): Traversal depth from entry entities (1 = direct facts). Capped at 5.
        top_k (int): Maximum number of facts to return.
    """

    group: Literal[NodeGroup.RETRIEVERS] = NodeGroup.RETRIEVERS
    action_type: ActionType = ActionType.DATABASE_QUERY
    name: str = "graph-retriever"
    description: str = (
        "Retrieves facts from a knowledge graph for a natural-language question. Input: a 'query' string "
        "(and optionally 'entities' to start from, or 'top_k'). Returns related entities and relationships "
        "as bullet-point facts. Use for questions about how things are connected."
    )
    error_handling: ErrorHandling = Field(default_factory=lambda: ErrorHandling(timeout_seconds=600))
    connection: Neo4j | ApacheAGE | AWSNeptune
    database: str | None = None
    graph_name: str | None = None
    create_graph_if_not_exists: bool = False
    filters: dict[str, Any] | None = None  # LOCKED: always applied; not overridable from the input schema
    max_depth: int = 1
    top_k: int = 10

    input_schema: ClassVar[type[GraphRetrieverInputSchema]] = GraphRetrieverInputSchema

    _graph_store: BaseGraphStore | None = PrivateAttr(default=None)
    _use_fulltext: bool = PrivateAttr(default=False)

    def init_components(self, connection_manager: ConnectionManager | None = None) -> None:
        """Select the concrete graph store from the connection type (same dispatch as CypherExecutor)."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self._graph_store is None:
            self._graph_store = self._build_graph_store()
        self._use_fulltext = self._probe_fulltext()

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

    def _effective_depth(self) -> int:
        return max(1, min(self.max_depth, _MAX_TRAVERSAL_DEPTH))

    def _entry(self, input_data: GraphRetrieverInputSchema, params: dict[str, Any]) -> tuple[list[str], str | None, bool]:
        """Pick the entry-point strategy. Returns (lead_lines, entry_where, anchored).

        - explicit ``entities``  -> anchor on ``:Entity`` nodes by exact name.
        - full-text index (Neo4j, index present) -> seek seed entities by the question's words.
        - otherwise (non-Neo4j, or index missing) -> portable ``CONTAINS`` scan.

        Anchored modes bind ``a`` to a seed and expand from it (cheap, neighborhood-bounded); the scan
        binds nothing and filters every edge by name (the fallback). Either way an entity with no
        ACL-visible edge yields no rows — so seeds without a visible edge drop out for free.
        """
        if input_data.entities:
            params["entities"] = input_data.entities
            return [f"MATCH (a:{ENTITY_LABEL})"], "a.name IN $entities", True

        if self._use_fulltext:
            lucene = _lucene_query(input_data.query)
            if lucene:
                params["q"] = lucene
                return [f"CALL db.index.fulltext.queryNodes('{ENTITY_NAME_FULLTEXT_INDEX}', $q) YIELD node AS a"], None, True

        params["q"] = input_data.query
        return [], "(toLower($q) CONTAINS toLower(a.name) OR toLower($q) CONTAINS toLower(b.name))", False

    def _build_query(self, input_data: GraphRetrieverInputSchema, limit: int) -> tuple[str, dict[str, Any]]:
        """Build the single parameterized Cypher query and its parameters."""
        params: dict[str, Any] = {"limit": limit}
        depth = self._effective_depth()
        lead, entry_where, anchored = self._entry(input_data, params)
        # Anchored expansion is undirected (catch edges into and out of the seed); direction is recovered
        # in RETURN via startNode/endNode. The scan keeps the directed pattern with a=source, b=target.
        arrow = "-[{rel}]-" if anchored else "-[{rel}]->"

        if depth == 1:
            rel_clauses, filter_params = self._edge_predicates("r", input_data.filters)
            params.update(filter_params)
            where = " AND ".join([t for t in [entry_where, *rel_clauses] if t])
            if anchored:
                ret = (
                    "RETURN coalesce(startNode(r).name, startNode(r).value) AS a_name, type(r) AS rel, "
                    "properties(r) AS rprops, coalesce(endNode(r).name, endNode(r).value) AS b_name"
                )
            else:
                ret = (
                    "RETURN coalesce(a.name, a.value) AS a_name, type(r) AS rel, "
                    "properties(r) AS rprops, coalesce(b.name, b.value) AS b_name"
                )
            lines = [*lead, f"MATCH (a){arrow.format(rel='r')}(b)"]
            if where:
                lines.append(f"WHERE {where}")
            lines += [ret, "LIMIT $limit"]
            return "\n".join(lines), params

        # Multi-hop: every edge on the path must be visible (all(...)). Depth is a validated int, so it is
        # safe to inline — Cypher does not allow a parameter in variable-length bounds.
        rel_clauses, filter_params = self._edge_predicates("rel", input_data.filters)
        params.update(filter_params)
        all_clause = f"all(rel IN rels WHERE {' AND '.join(rel_clauses)})" if rel_clauses else None
        where = " AND ".join([t for t in [entry_where, all_clause] if t])
        lines = [*lead, f"MATCH path = (a){arrow.format(rel=f'rels*1..{depth}')}(b)"]
        if where:
            lines.append(f"WHERE {where}")
        lines += [
            "RETURN [n IN nodes(path) | coalesce(n.name, n.value)] AS node_names, "
            "[r IN relationships(path) | type(r)] AS rel_types, "
            "[r IN relationships(path) | properties(r)] AS rel_props",
            "LIMIT $limit",
        ]
        return "\n".join(lines), params

    def _edge_predicates(self, rel_var: str, user_filters: dict[str, Any] | None) -> tuple[list[str], dict[str, Any]]:
        """Compile the locked node filters + the caller's filters for an edge variable, AND-ed together.

        Locked (node) filters always apply; caller filters can only narrow further. Distinct parameter
        prefixes ("lf"/"uf") keep the two sets from colliding.
        """
        locked_clauses, locked_params = _compile_edge_filters(rel_var, self.filters, param_prefix="lf")
        user_clauses, user_params = _compile_edge_filters(rel_var, user_filters, param_prefix="uf")
        return [*locked_clauses, *user_clauses], {**locked_params, **user_params}

    def execute(
        self, input_data: GraphRetrieverInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        """Retrieve filtered graph context for the query and render it as Documents."""
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        if not self._graph_store:
            raise ToolExecutionException("GraphRetriever: graph store is not initialized.", recoverable=True)

        limit = input_data.top_k or self.top_k
        try:
            query, params = self._build_query(input_data, limit)
            records, _, _ = self._graph_store.run_cypher(query, parameters=params, database=self.database)
            rows = self._graph_store.format_records(records)
            documents = (
                self._render_single_hop(rows) if self._effective_depth() == 1 else self._render_paths(rows)
            )
            content = "\n".join(f"- {d.content}" for d in documents) or "No matching facts found."
            logger.info(f"Tool {self.name} - {self.id}: retrieved {len(documents)} fact(s).")
            return {"content": content, "documents": documents}
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: execution error: {e}", exc_info=True)
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve graph context. Error: {e}.", recoverable=True
            )

    @staticmethod
    def _render_single_hop(rows: list[dict[str, Any]]) -> list[Document]:
        documents: list[Document] = []
        for rank, row in enumerate(rows):
            source, rel, target = row.get("a_name"), row.get("rel"), row.get("b_name")
            if source is None or target is None or not rel:
                continue
            rprops = dict(row.get("rprops") or {})
            documents.append(
                Document(
                    content=f"{source} -[{rel}]-> {target}",
                    metadata={"source": source, "target": target, "rel": rel, **rprops},
                    score=1.0 / (1.0 + rank),
                )
            )
        return documents

    @staticmethod
    def _render_paths(rows: list[dict[str, Any]]) -> list[Document]:
        documents: list[Document] = []
        for rank, row in enumerate(rows):
            names = row.get("node_names") or []
            rels = row.get("rel_types") or []
            if len(names) < 2 or len(rels) != len(names) - 1:
                continue
            fact = names[0]
            for rel, node in zip(rels, names[1:]):
                fact += f" -[{rel}]-> {node}"
            # Merge provenance pointers across the path's edges.
            source_doc_ids: list[str] = []
            for props in row.get("rel_props") or []:
                source_doc_ids.extend((props or {}).get("source_doc_ids") or [])
            documents.append(
                Document(
                    content=fact,
                    metadata={
                        "source": names[0],
                        "target": names[-1],
                        "rels": rels,
                        "source_doc_ids": source_doc_ids,
                    },
                    score=1.0 / (1.0 + rank),
                )
            )
        return documents
