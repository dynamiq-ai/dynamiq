import asyncio
import hashlib
import weakref
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from dynamiq.connections import VertexAI as VertexAIConnection
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION = (
    "Retrieves relevant text chunks from a Google Cloud Vertex AI RAG Engine knowledge base "
    "(managed RAG corpora, part of the Gemini Enterprise Agent Platform). Google performs embedding "
    "and retrieval server-side; each result includes the chunk text, its source document URI and "
    "display name, and a relevance score. Provide a natural-language 'query'; optionally set 'top_k' "
    "to control how many chunks are returned and 'metadata_filter' to restrict results by file "
    "metadata (RAG Engine metadata filter expression)."
)

CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


class VertexAIRagSearchInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query to retrieve documents.")
    top_k: int | None = Field(
        default=None, ge=1, description="Parameter to provide how many documents to retrieve."
    )
    metadata_filter: str | None = Field(
        default=None,
        description=(
            "Parameter to provide a RAG Engine metadata filter expression to retrieve specific documents. "
            "Overrides the node-level metadata_filter."
        ),
    )


class VertexAIRagSearch(ConnectionNode):
    """Search tool for Google Cloud Vertex AI RAG Engine corpora.

    Calls the `VertexRagService.RetrieveContexts` API through an explicit-credentials GAPIC client:
    Google embeds the query and performs retrieval server-side against a managed RAG corpus, so no
    local embedder or vector store is needed. The process-global `vertexai.init()` is intentionally
    avoided so multiple workflows with different service accounts can run in one process.

    The corpus region must match the connection's `vertex_project_location` (RAG Engine is regional).
    Requires the `aiplatform.ragCorpora.query` permission (e.g. roles/aiplatform.user).
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.SEMANTIC_SEARCH
    name: str = "vertexai-rag-search"
    description: str = DESCRIPTION
    connection: VertexAIConnection | None = None
    rag_corpus_id: str = Field(
        ...,
        description="RAG corpus: bare ID or full 'projects/{p}/locations/{l}/ragCorpora/{id}' resource name.",
    )
    rag_file_ids: list[str] | None = Field(
        default=None, description="Optional RAG file IDs to restrict retrieval to specific files."
    )
    top_k: int | None = Field(default=10, ge=1)
    vector_distance_threshold: float | None = Field(
        default=None,
        description="Keep contexts with vector distance below this value (lower distance = closer).",
    )
    vector_similarity_threshold: float | None = Field(
        default=None,
        description="Keep contexts with vector similarity above this value (mutually exclusive with distance).",
    )
    metadata_filter: str | None = Field(default=None, description="Default RAG Engine metadata filter expression.")
    rank_service_model: str | None = Field(
        default=None,
        description="Semantic ranker model name (e.g. 'semantic-ranker-default@latest').",
    )
    llm_ranker_model: str | None = Field(
        default=None,
        description="LLM ranker model name (e.g. 'gemini-2.0-flash'); mutually exclusive with rank_service_model.",
    )
    metadata_fields: list[str] | None = Field(
        default_factory=lambda: ["source_uri", "source_display_name", "page_first", "page_last"],
        description=(
            "Allowlist of document metadata keys to include in the formatted content (case-insensitive). "
            "Set to None to include every metadata field."
        ),
    )
    timeout: float = 30
    input_schema: ClassVar[type[VertexAIRagSearchInputSchema]] = VertexAIRagSearchInputSchema
    _async_clients: dict[int, tuple[Any, Any]] = PrivateAttr(default_factory=dict)

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = VertexAIConnection()
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def validate_threshold_and_ranking(self):
        if self.vector_distance_threshold is not None and self.vector_similarity_threshold is not None:
            raise ValueError(
                "vector_distance_threshold and vector_similarity_threshold are mutually exclusive; set at most one."
            )
        if self.rank_service_model and self.llm_ranker_model:
            raise ValueError("rank_service_model and llm_ranker_model are mutually exclusive; set at most one.")
        return self

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """Initialize the node and build the Vertex RAG service client from the connection."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.client is None:
            self.client = self._build_client()

    @property
    def project_id(self) -> str:
        if self.connection is None:
            raise ValueError("A VertexAI connection is required to resolve the project.")
        project = self.connection.vertex_project_id or self.connection.project_id
        if not project:
            raise ValueError("The VertexAI connection must provide vertex_project_id (or project_id).")
        return project

    @property
    def location(self) -> str:
        if self.connection is None:
            raise ValueError("A VertexAI connection is required to resolve the location.")
        location = self.connection.vertex_project_location
        if not location:
            raise ValueError("The VertexAI connection must provide vertex_project_location.")
        return location

    def _build_credentials(self) -> Any:
        from google.oauth2 import service_account

        connection = self.connection
        info = {
            "type": "service_account",
            "project_id": connection.project_id,
            "private_key_id": connection.private_key_id,
            "private_key": connection.private_key,
            "client_email": connection.client_email,
            "client_id": connection.client_id,
            "auth_uri": connection.auth_uri,
            "token_uri": connection.token_uri,
            "auth_provider_x509_cert_url": connection.auth_provider_x509_cert_url,
            "client_x509_cert_url": connection.client_x509_cert_url,
            "universe_domain": connection.universe_domain,
        }
        info = {key: value for key, value in info.items() if value}
        return service_account.Credentials.from_service_account_info(info, scopes=[CLOUD_PLATFORM_SCOPE])

    def _endpoint_location(self) -> str:
        """Region for the API endpoint — RetrieveContexts is regional and must target the corpus's region."""
        if self.rag_corpus_id.startswith("projects/"):
            parts = self.rag_corpus_id.split("/")
            if len(parts) < 4 or parts[2] != "locations" or not parts[3]:
                raise ValueError(
                    f"Invalid rag_corpus_id resource name '{self.rag_corpus_id}'; expected "
                    "'projects/{project}/locations/{location}/ragCorpora/{id}'."
                )
            return parts[3]
        return self.location

    def _client_options(self) -> Any:
        from google.api_core.client_options import ClientOptions

        return ClientOptions(api_endpoint=f"{self._endpoint_location()}-aiplatform.googleapis.com")

    def _build_client(self) -> Any:
        from google.cloud import aiplatform_v1

        return aiplatform_v1.VertexRagServiceClient(
            credentials=self._build_credentials(), client_options=self._client_options()
        )

    async def get_async_client(self) -> Any:
        """Return a per-event-loop async RAG service client (async gRPC clients are loop-bound).

        Loop ids get reused after a loop is garbage-collected, so entries carry a weakref to their
        origin loop and dead entries are pruned on access (mirrors ConnectionManager's per-loop
        async client guard); otherwise clients bound to closed loops would leak and be reused.
        """
        from google.cloud import aiplatform_v1

        if self.connection is None:
            raise ValueError("A VertexAI connection is required to build the async client.")

        loop = asyncio.get_running_loop()
        stale_keys = []
        for key, (loop_ref, _) in self._async_clients.items():
            cached_loop = loop_ref()
            # A GC'd loop is trivially stale; a closed loop can never run its client again, and its
            # cached client's gRPC channel would otherwise pin the dead loop in memory forever.
            if cached_loop is None or cached_loop.is_closed():
                stale_keys.append(key)
        for key in stale_keys:
            del self._async_clients[key]

        entry = self._async_clients.get(id(loop))
        if entry is not None and entry[0]() is loop:
            return entry[1]

        client = aiplatform_v1.VertexRagServiceAsyncClient(
            credentials=self._build_credentials(), client_options=self._client_options()
        )
        self._async_clients[id(loop)] = (weakref.ref(loop), client)
        return client

    @property
    def corpus_resource_name(self) -> str:
        if self.rag_corpus_id.startswith("projects/"):
            return self.rag_corpus_id
        if self.connection is None:
            raise ValueError(
                "A bare rag_corpus_id requires a VertexAI connection to resolve the project and location; "
                "provide the full 'projects/{project}/locations/{location}/ragCorpora/{id}' resource name instead."
            )
        return f"projects/{self.project_id}/locations/{self.location}/ragCorpora/{self.rag_corpus_id}"

    def _corpus_parent(self) -> str:
        """The 'projects/{p}/locations/{l}' prefix of the corpus — RetrieveContexts' parent must match it."""
        return "/".join(self.corpus_resource_name.split("/")[:4])

    def _build_request(self, input_data: VertexAIRagSearchInputSchema) -> Any:
        from google.cloud import aiplatform_v1

        query = input_data.query.strip()
        if not query:
            raise ToolExecutionException(
                "The 'query' parameter must be a non-empty string. Please provide a query and try again.",
                recoverable=True,
            )

        top_k = input_data.top_k if input_data.top_k is not None else self.top_k
        metadata_filter = input_data.metadata_filter or self.metadata_filter

        filter_kwargs: dict[str, Any] = {}
        if self.vector_distance_threshold is not None:
            filter_kwargs["vector_distance_threshold"] = self.vector_distance_threshold
        if self.vector_similarity_threshold is not None:
            filter_kwargs["vector_similarity_threshold"] = self.vector_similarity_threshold
        if metadata_filter:
            filter_kwargs["metadata_filter"] = metadata_filter

        ranking = None
        if self.rank_service_model:
            ranking = aiplatform_v1.RagRetrievalConfig.Ranking(
                rank_service=aiplatform_v1.RagRetrievalConfig.Ranking.RankService(model_name=self.rank_service_model)
            )
        elif self.llm_ranker_model:
            ranking = aiplatform_v1.RagRetrievalConfig.Ranking(
                llm_ranker=aiplatform_v1.RagRetrievalConfig.Ranking.LlmRanker(model_name=self.llm_ranker_model)
            )

        retrieval_config = aiplatform_v1.RagRetrievalConfig(
            **({"top_k": top_k} if top_k is not None else {}),
            **({"filter": aiplatform_v1.RagRetrievalConfig.Filter(**filter_kwargs)} if filter_kwargs else {}),
            **({"ranking": ranking} if ranking is not None else {}),
        )

        rag_resource = aiplatform_v1.RetrieveContextsRequest.VertexRagStore.RagResource(
            rag_corpus=self.corpus_resource_name,
            rag_file_ids=self.rag_file_ids or [],
        )
        return aiplatform_v1.RetrieveContextsRequest(
            parent=self._corpus_parent(),
            vertex_rag_store=aiplatform_v1.RetrieveContextsRequest.VertexRagStore(rag_resources=[rag_resource]),
            query=aiplatform_v1.RagQuery(text=query, rag_retrieval_config=retrieval_config),
        )

    def _prepare_request(self, input_data: VertexAIRagSearchInputSchema) -> Any:
        """Build the request, translating configuration errors into recoverable tool errors."""
        try:
            return self._build_request(input_data)
        except ToolExecutionException:
            raise
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to build the request. Error: {str(e)}")
            raise ToolExecutionException(
                f"Failed to prepare the Vertex AI RAG request: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

    def _score_type(self) -> str | None:
        if self.rank_service_model or self.llm_ranker_model:
            return "ranked"
        if self.vector_similarity_threshold is not None:
            return "similarity"
        if self.vector_distance_threshold is not None:
            return "distance"
        return None

    def _to_document(self, context: Any) -> Document:
        metadata: dict[str, Any] = {"rag_corpus": self.corpus_resource_name}
        if context.source_uri:
            metadata["source_uri"] = context.source_uri
        if context.source_display_name:
            metadata["source_display_name"] = context.source_display_name

        page_span = context.chunk.page_span
        if page_span.first_page:
            metadata["page_first"] = page_span.first_page
        if page_span.last_page:
            metadata["page_last"] = page_span.last_page

        score_type = self._score_type()
        if score_type:
            metadata["score_type"] = score_type

        # RetrieveContexts returns no stable chunk id; derive a deterministic one from source + text.
        digest_source = f"{context.source_uri}|{context.text}".encode()
        document_id = hashlib.sha256(digest_source).hexdigest()[:32]

        # Score is an optional proto field whose polarity depends on the corpus backend
        # (e.g. cosine distance for RagManagedDb, similarity for other metrics) — pass it through untouched.
        score = context.score if context._pb.HasField("score") else None

        return Document(id=document_id, content=context.text or "", metadata=metadata, score=score)

    def _format_content(self, documents: list[Document]) -> str:
        """Format documents into a numbered, source-labelled block (mirrors the Dynamiq KB tools)."""
        allowed = None if self.metadata_fields is None else {field.lower() for field in self.metadata_fields}

        formatted_docs: list[str] = []
        for index, doc in enumerate(documents):
            metadata_lines: list[str] = []
            if doc.score is not None:
                metadata_lines.append(f"Score: {doc.score}")
            for key, value in (doc.metadata or {}).items():
                if allowed is not None and key.lower() not in allowed:
                    continue
                metadata_lines.append(f"{key}: {value}")

            metadata_block = "\n".join(metadata_lines) if metadata_lines else "No metadata available."
            content_block = (doc.content or "").strip()

            formatted_docs.append(
                (
                    f"--- Retrieved Source {index + 1} ---\n"
                    f"Metadata:\n{metadata_block}\n\n"
                    f"Content:\n{content_block}\n"
                    f"--- End Source {index + 1} ---"
                ).rstrip()
            )

        return "\n\n".join(formatted_docs)

    def _parse_response(self, response: Any) -> dict[str, Any]:
        try:
            documents = [self._to_document(context) for context in response.contexts.contexts]
            content = self._format_content(documents)
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to parse response. Error: {str(e)}")
            raise ToolExecutionException(
                f"Failed to parse response with error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        if not documents:
            content = "No results found for the given query."

        logger.info(f"Tool {self.name} - {self.id}: retrieved {len(documents)} documents")
        return {"content": content, "documents": documents}

    @staticmethod
    def _error_details(e: Exception) -> str:
        error_code = getattr(e, "code", None)
        if error_code is not None and not callable(error_code):
            return f"{error_code}: {str(e)}"
        return str(e)

    def execute(
        self, input_data: VertexAIRagSearchInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")

        request = self._prepare_request(input_data)

        try:
            response = self.client.retrieve_contexts(request=request, timeout=self.timeout)
        except ToolExecutionException:
            raise
        except Exception as e:
            error_details = self._error_details(e)
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {error_details}")
            raise ToolExecutionException(
                f"Vertex AI RAG retrieval failed with error: {error_details}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        return self._parse_response(response)

    async def execute_async(
        self, input_data: VertexAIRagSearchInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")

        request = self._prepare_request(input_data)

        try:
            if self.connection is None:
                # A client-only node has no credentials to build a loop-bound async client;
                # run the injected sync client in a worker thread instead.
                response = await asyncio.to_thread(self.client.retrieve_contexts, request=request, timeout=self.timeout)
            else:
                client = await self.get_async_client()
                response = await client.retrieve_contexts(request=request, timeout=self.timeout)
        except ToolExecutionException:
            raise
        except Exception as e:
            error_details = self._error_details(e)
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {error_details}")
            raise ToolExecutionException(
                f"Vertex AI RAG retrieval failed with error: {error_details}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        return self._parse_response(response)
