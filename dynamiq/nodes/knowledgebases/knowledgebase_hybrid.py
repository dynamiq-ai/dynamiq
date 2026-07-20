import asyncio
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import Node, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.types import Document
from dynamiq.types.cancellation import CanceledException, check_cancellation
from dynamiq.utils.logger import logger

from .knowledgebase_graph import DynamiqKnowledgebaseGraphSearch
from .knowledgebase_vector import DynamiqKnowledgebaseVectorSearch

DESCRIPTION = (
    "Hybrid retrieval over a Dynamiq knowledgebase: runs vector search and graph search together, "
    "merges the vector chunks with the graph's source documents, and reranks the combined set for the "
    "query. Graph facts are appended to the returned content. Access control is enforced by the "
    "Dynamiq API based on the connection credentials."
)


class DynamiqKnowledgebaseHybridSearchInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query to retrieve documents.")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter to provide filters to apply for retrieving specific documents.",
    )
    limit: int | None = Field(default=None, description="Parameter to provide how many documents to retrieve.")
    similarity_threshold: float | None = Field(
        default=None,
        description="Parameter to provide minimal similarity or maximal distance score for retrieved documents.",
    )
    alpha: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Parameter to provide alpha for hybrid vector retrieval. 0 is keyword-only, 1 is semantic-only.",
    )
    user: str | None = Field(
        default=None,
        description="Parameter to provide the user identity for ACL-enforced retrieval.",
        json_schema_extra={"is_accessible_to_agent": False},
    )


class DynamiqKnowledgebaseHybridSearch(Node):
    """Composes ``DynamiqKnowledgebaseVectorSearch`` and ``DynamiqKnowledgebaseGraphSearch`` into one tool.

    The two sub-searches run concurrently (via ``execute_async``). Vector chunks and the graph's structured
    source documents are merged and deduplicated, then reranked against the query when a reranker is
    configured (otherwise all merged documents are returned). The graph's ``content`` (facts block) is
    always appended to the formatted output.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.SEMANTIC_SEARCH
    name: str = "dynamiq-knowledgebase-hybrid-search"
    description: str = DESCRIPTION

    vector_search: DynamiqKnowledgebaseVectorSearch
    graph_search: DynamiqKnowledgebaseGraphSearch
    # Any RANKERS-group node with a ``{query, documents}`` -> ``{documents}`` contract (e.g. CohereReranker).
    reranker: Node | None = None
    # Per-source fetch size and the final cap on the merged result; overridable per call via input ``limit``.
    limit: int | None = None

    input_schema: ClassVar[type[DynamiqKnowledgebaseHybridSearchInputSchema]] = (
        DynamiqKnowledgebaseHybridSearchInputSchema
    )

    @property
    def to_dict_exclude_params(self):
        # Sub-nodes are serialized via their own ``to_dict`` (below) so their non-serializable runtime
        # state (e.g. HTTP ``client``) is excluded; keep them out of the parent ``model_dump``.
        return super().to_dict_exclude_params | {
            "vector_search": True,
            "graph_search": True,
            "reranker": True,
        }

    def to_dict(self, **kwargs) -> dict:
        """Serialize composed sub-nodes through their own ``to_dict`` so the node roundtrips via YAML."""
        data = super().to_dict(**kwargs)
        data["vector_search"] = self.vector_search.to_dict(**kwargs)
        data["graph_search"] = self.graph_search.to_dict(**kwargs)
        data["reranker"] = self.reranker.to_dict(**kwargs) if self.reranker else None
        return data

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """Initialize the composed sub-nodes so they share the parent's connection manager."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        for component in (self.vector_search, self.graph_search, self.reranker):
            if component is not None and component.is_postponed_component_init:
                component.init_components(connection_manager)

    def _resolve_limit(self, input_data: DynamiqKnowledgebaseHybridSearchInputSchema) -> int | None:
        """Effective limit for this run: per-call input wins over the node default."""
        return input_data.limit if input_data.limit is not None else self.limit

    def _sub_input(self, input_data: DynamiqKnowledgebaseHybridSearchInputSchema, *, graph: bool) -> dict[str, Any]:
        """Build the input dict for a sub-search, dropping keys the graph search does not accept."""
        payload: dict[str, Any] = {"query": input_data.query, "filters": input_data.filters}
        limit = self._resolve_limit(input_data)
        if limit is not None:
            payload["limit"] = limit
        if input_data.user is not None:
            payload["user"] = input_data.user
        if not graph:
            if input_data.similarity_threshold is not None:
                payload["similarity_threshold"] = input_data.similarity_threshold
            if input_data.alpha is not None:
                payload["alpha"] = input_data.alpha
        return payload

    @staticmethod
    def _sub_run_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
        """Tracing kwargs for a sub-run: reparent it under THIS node's run.

        Sets ``parent_run_id`` to the hybrid node's own ``run_id`` so the vector/graph/reranker runs nest
        under it in the trace tree (mirrors ``Agent``). Without it the sub-runs record no parent, the tracer
        treats each as a root, and ``TracingCallbackHandler`` flushes when a sub-search finishes -- before the
        hybrid run completes. ``trace_id``/``session_id`` need no threading: they live on the shared tracer
        reached via ``config.callbacks``.
        """
        return {"parent_run_id": kwargs.get("run_id")}

    def _rerank_or_degrade(self, result: RunnableResult, documents: list[Document]) -> list[Document]:
        """Unwrap a reranker ``RunnableResult``; on failure degrade to the unranked ``documents``.

        Reranking is an optional refinement -- a reranker failure must not abort a hybrid search whose
        sub-searches already succeeded (mirrors ``KnowledgeGraphRetriever._maybe_rerank``). Cancellation
        still propagates.
        """
        if result.status == RunnableStatus.CANCELED:
            raise CanceledException()
        if result.status != RunnableStatus.SUCCESS:
            error = result.error.to_dict() if result.error else "unknown error"
            logger.warning(
                f"Tool {self.name} - {self.id}: reranker '{self.reranker.name}' failed: {error}; "
                f"returning unranked documents."
            )
            return documents
        return (result.output or {}).get("documents", documents)

    def _output_or_degrade(self, result: RunnableResult, label: str) -> dict[str, Any] | None:
        """Like ``_output_or_raise`` but degrades a failed sub-search to ``None`` (logged) instead of raising.

        Cancellation still propagates -- a canceled sub-run is never routed around.
        """
        if result.status == RunnableStatus.CANCELED:
            raise CanceledException()
        if result.status != RunnableStatus.SUCCESS:
            error = result.error.to_dict() if result.error else "unknown error"
            logger.warning(
                f"Tool {self.name} - {self.id}: {label} search failed: {error}; returning the other source."
            )
            return None
        return result.output or {}

    def _resolve_outputs(
        self, vector_result: RunnableResult, graph_result: RunnableResult
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resolve both sub-search outputs, tolerating a single failure. Fails only when BOTH fail.

        A failed source is logged (in ``_output_or_degrade``) and replaced by ``{}`` so the merge proceeds
        on whatever survived. Cancellation still propagates.
        """
        vector_output = self._output_or_degrade(vector_result, "vector")
        graph_output = self._output_or_degrade(graph_result, "graph")
        if vector_output is None and graph_output is None:
            raise ToolExecutionException(
                "Hybrid search failed: both the vector and graph sub-searches errored. "
                "Please analyze the error and take appropriate action.",
                recoverable=True,
            )
        return vector_output or {}, graph_output or {}

    def _dedupe_merge(self, vector_output: dict[str, Any], graph_output: dict[str, Any]) -> list[Document]:
        """Merge and deduplicate vector chunks with graph source documents (pre-rerank).

        Deduplication is by ``Document.id`` when present, else by content. Vector documents win ties so
        their scores/metadata are preserved; every document is tagged with its ``retrieval_source``.
        """
        vector_documents = [self._as_document(item, "vector") for item in (vector_output.get("documents") or [])]

        raw_graph_docs = graph_output.get("source_documents") or []
        graph_documents = [self._as_document(item, "graph") for item in raw_graph_docs]

        merged: list[Document] = []
        seen: set[str] = set()
        for document in [*vector_documents, *graph_documents]:
            key = document.id or (document.content or "")
            if key in seen:
                continue
            seen.add(key)
            merged.append(document)
        return merged

    def _finalize(
        self, documents: list[Document], graph_output: dict[str, Any], limit: int | None
    ) -> dict[str, Any]:
        """Apply the limit cap and append the graph facts block to form the final output."""
        if limit is not None:
            documents = documents[:limit]  # cap uniformly, whether or not a reranker reordered

        # Append the graph's ``facts`` (triples) -- its unique signal; source-doc text is already in the pool.
        graph_facts = (graph_output.get("facts") or "").strip()
        content = self._format_content(documents)
        if graph_facts:
            content = f"{content}\n\n--- Graph Facts ---\n{graph_facts}" if content else graph_facts

        return {"content": content, "documents": documents}

    def _merge(
        self,
        query: str,
        vector_output: dict[str, Any],
        graph_output: dict[str, Any],
        config: RunnableConfig,
        limit: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Merge vector chunks with graph source documents, rerank, and append graph facts."""
        merged = self._dedupe_merge(vector_output, graph_output)
        documents = self._rerank(query, merged, config, **kwargs)
        return self._finalize(documents, graph_output, limit)

    async def _merge_async(
        self,
        query: str,
        vector_output: dict[str, Any],
        graph_output: dict[str, Any],
        config: RunnableConfig,
        limit: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Async counterpart of ``_merge``: awaits the reranker so the event loop is never blocked."""
        merged = self._dedupe_merge(vector_output, graph_output)
        documents = await self._rerank_async(query, merged, config, **kwargs)
        return self._finalize(documents, graph_output, limit)

    def _rerank(
        self, query: str, documents: list[Document], config: RunnableConfig, **kwargs
    ) -> list[Document]:
        """Reorder merged documents by query relevance when a reranker is configured.

        Returns the documents unchanged (order preserved) when no reranker is set. The ``limit`` cap is
        applied by the caller so it holds for both the reranked and the unranked paths.
        """
        if not documents or self.reranker is None:
            return documents

        result = self.reranker.run(
            input_data={"query": query, "documents": documents},
            config=config,
            run_depends=kwargs.get("run_depends", []),
            **self._sub_run_kwargs(kwargs),
        )
        return self._rerank_or_degrade(result, documents)

    async def _rerank_async(
        self, query: str, documents: list[Document], config: RunnableConfig, **kwargs
    ) -> list[Document]:
        """Async counterpart of ``_rerank`` -- awaits ``run_async`` so a sync reranker is offloaded to a
        thread (via ``Node.run_async``) instead of blocking the event loop."""
        if not documents or self.reranker is None:
            return documents

        result = await self.reranker.run_async(
            input_data={"query": query, "documents": documents},
            config=config,
            run_depends=kwargs.get("run_depends", []),
            **self._sub_run_kwargs(kwargs),
        )
        return self._rerank_or_degrade(result, documents)

    @staticmethod
    def _as_document(item: Any, source: str) -> Document:
        document = DynamiqKnowledgebaseVectorSearch._to_document(item)
        metadata = dict(document.metadata or {})
        metadata.setdefault("retrieval_source", source)
        document.metadata = metadata
        return document

    def _format_content(self, documents: list[Document]) -> str:
        """Format merged documents by delegating to the vector search's numbered-source formatter."""
        return self.vector_search._format_content(documents)

    def execute(
        self, input_data: DynamiqKnowledgebaseHybridSearchInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")

        run_depends = kwargs.get("run_depends", [])
        sub_kwargs = self._sub_run_kwargs(kwargs)
        vector_result = self.vector_search.run(
            input_data=self._sub_input(input_data, graph=False), config=config, run_depends=run_depends, **sub_kwargs
        )
        graph_result = self.graph_search.run(
            input_data=self._sub_input(input_data, graph=True), config=config, run_depends=run_depends, **sub_kwargs
        )

        vector_output, graph_output = self._resolve_outputs(vector_result, graph_result)

        return self._merge(
            input_data.query, vector_output, graph_output, config, limit=self._resolve_limit(input_data), **kwargs
        )

    async def execute_async(
        self, input_data: DynamiqKnowledgebaseHybridSearchInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")

        run_depends = kwargs.get("run_depends", [])
        sub_kwargs = self._sub_run_kwargs(kwargs)
        vector_result, graph_result = await asyncio.gather(
            self.vector_search.run_async(
                input_data=self._sub_input(input_data, graph=False),
                config=config,
                run_depends=run_depends,
                **sub_kwargs,
            ),
            self.graph_search.run_async(
                input_data=self._sub_input(input_data, graph=True),
                config=config,
                run_depends=run_depends,
                **sub_kwargs,
            ),
        )

        vector_output, graph_output = self._resolve_outputs(vector_result, graph_result)

        return await self._merge_async(
            input_data.query, vector_output, graph_output, config, limit=self._resolve_limit(input_data), **kwargs
        )
