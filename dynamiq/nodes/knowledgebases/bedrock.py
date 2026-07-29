from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field, model_validator

from dynamiq.connections import AWS as AWSConnection
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
    "Retrieves relevant document chunks from an Amazon Bedrock Knowledge Base. "
    "AWS performs embedding and vector search server-side; each result includes the chunk text, "
    "its source location and metadata, and usually a relevance score (higher is more relevant). "
    "Provide a natural-language 'query'; optionally set 'top_k' to control how many chunks are returned "
    "and 'filters' to restrict results by document metadata attributes. Filters use the Bedrock "
    "RetrievalFilter JSON format with exactly one operator per object: equals, notEquals, greaterThan, "
    "greaterThanOrEquals, lessThan, lessThanOrEquals, in, notIn, startsWith, listContains, stringContains "
    '(each takes {"key": <attribute>, "value": <value>}), or andAll/orAll (a list of at least two '
    'sub-filters). Example: {"andAll": [{"equals": {"key": "department", "value": "hr"}}, '
    '{"greaterThan": {"key": "year", "value": 2023}}]}. '
    "Filters only match metadata attributes that were ingested with the documents."
)

FILTER_LEAF_OPERATORS = frozenset(
    {
        "equals",
        "notEquals",
        "greaterThan",
        "greaterThanOrEquals",
        "lessThan",
        "lessThanOrEquals",
        "in",
        "notIn",
        "startsWith",
        "listContains",
        "stringContains",
    }
)
FILTER_GROUP_OPERATORS = frozenset({"andAll", "orAll"})

# location.type -> (location payload key, uri-like field inside the payload)
LOCATION_URI_FIELDS = {
    "S3": ("s3Location", "uri"),
    "WEB": ("webLocation", "url"),
    "CONFLUENCE": ("confluenceLocation", "url"),
    "SALESFORCE": ("salesforceLocation", "url"),
    "SHAREPOINT": ("sharePointLocation", "url"),
    "CUSTOM": ("customDocumentLocation", "id"),
    "KENDRA": ("kendraDocumentLocation", "uri"),
    "SQL": ("sqlLocation", "query"),
    "ONEDRIVE": ("oneDriveLocation", "url"),
    "GOOGLEDRIVE": ("googleDriveLocation", "url"),
}


class BedrockRerankingConfig(BaseModel):
    """Configuration for reranking retrieved results with a Bedrock reranker model.

    Requires a reranker-supported region and a reranker model (e.g. amazon.rerank-v1:0,
    cohere.rerank-v3-5:0). Bedrock assumes the *knowledge base's service role* to rerank, so the
    permissions belong on that role rather than on the caller: `bedrock:Rerank` with
    `Resource: "*"` (it cannot be scoped to a foundation-model ARN — doing so still yields
    AccessDenied) plus `bedrock:InvokeModel` on the reranker model. Marketplace-distributed
    rerankers such as the Cohere models also need their subscription accepted for the account.
    """

    model_arn: str
    number_of_reranked_results: int | None = Field(default=None, ge=1, le=100)
    additional_model_request_fields: dict[str, Any] | None = None
    metadata_selection_mode: Literal["ALL", "SELECTIVE"] | None = None
    metadata_fields_to_include: list[str] | None = None
    metadata_fields_to_exclude: list[str] | None = None

    @model_validator(mode="after")
    def validate_metadata_selection(self):
        if self.metadata_fields_to_include and self.metadata_fields_to_exclude:
            raise ValueError("Provide either metadata_fields_to_include or metadata_fields_to_exclude, not both.")
        if (self.metadata_fields_to_include or self.metadata_fields_to_exclude) and (
            self.metadata_selection_mode != "SELECTIVE"
        ):
            raise ValueError("metadata_fields_to_include/exclude require metadata_selection_mode='SELECTIVE'.")
        return self

    def to_api(self) -> dict[str, Any]:
        model_configuration: dict[str, Any] = {"modelArn": self.model_arn}
        if self.additional_model_request_fields:
            model_configuration["additionalModelRequestFields"] = self.additional_model_request_fields

        bedrock_config: dict[str, Any] = {"modelConfiguration": model_configuration}
        if self.number_of_reranked_results is not None:
            bedrock_config["numberOfRerankedResults"] = self.number_of_reranked_results

        if self.metadata_selection_mode:
            metadata_configuration: dict[str, Any] = {"selectionMode": self.metadata_selection_mode}
            if self.metadata_fields_to_include:
                metadata_configuration["selectiveModeConfiguration"] = {
                    "fieldsToInclude": [{"fieldName": field} for field in self.metadata_fields_to_include]
                }
            elif self.metadata_fields_to_exclude:
                metadata_configuration["selectiveModeConfiguration"] = {
                    "fieldsToExclude": [{"fieldName": field} for field in self.metadata_fields_to_exclude]
                }
            bedrock_config["metadataConfiguration"] = metadata_configuration

        return {"type": "BEDROCK_RERANKING_MODEL", "bedrockRerankingConfiguration": bedrock_config}


class BedrockImplicitFilterAttribute(BaseModel):
    """A metadata attribute the filter model may use when inferring filters from the query."""

    key: str
    type: Literal["STRING", "NUMBER", "BOOLEAN", "STRING_LIST"]
    description: str


class BedrockImplicitFilterConfig(BaseModel):
    """Configuration for implicit (model-inferred) metadata filtering.

    Requires `bedrock:InvokeModel` on the filter model.
    """

    metadata_attributes: list[BedrockImplicitFilterAttribute] = Field(..., min_length=1, max_length=25)
    model_arn: str

    def to_api(self) -> dict[str, Any]:
        return {
            "metadataAttributes": [
                {"key": attribute.key, "type": attribute.type, "description": attribute.description}
                for attribute in self.metadata_attributes
            ],
            "modelArn": self.model_arn,
        }


class BedrockManagedSearchConfig(BaseModel):
    """Configuration for MANAGED-type knowledge bases (fully managed Bedrock KBs, GA June 2026).

    When set, the tool sends `managedSearchConfiguration` instead of `vectorSearchConfiguration`.
    Managed KBs are commonly addressed by their full ARN and support `user_id`-based ACL filtering.
    """

    reranking_model_type: Literal["CUSTOM", "MANAGED", "NONE"] | None = None
    reranking: BedrockRerankingConfig | None = None

    @model_validator(mode="after")
    def validate_reranking(self):
        if self.reranking and self.reranking_model_type != "CUSTOM":
            raise ValueError("A custom reranking config requires reranking_model_type='CUSTOM'.")
        if self.reranking_model_type == "CUSTOM" and not self.reranking:
            raise ValueError("reranking_model_type='CUSTOM' requires a reranking config.")
        return self

    def to_api(self, number_of_results: int | None, filters: dict[str, Any] | None) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if number_of_results is not None:
            config["numberOfResults"] = number_of_results
        if filters:
            config["filter"] = filters
        if self.reranking_model_type:
            config["rerankingModelType"] = self.reranking_model_type
        if self.reranking:
            config["rerankingConfiguration"] = self.reranking.to_api()
        return config


class BedrockGuardrailConfig(BaseModel):
    """Guardrail applied to retrieved content; blocked/masked results surface as guardrail_action."""

    guardrail_id: str
    guardrail_version: str

    def to_api(self) -> dict[str, str]:
        return {"guardrailId": self.guardrail_id, "guardrailVersion": self.guardrail_version}


class BedrockKnowledgeBaseSearchInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query to retrieve documents.")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameter to provide a Bedrock RetrievalFilter JSON object to retrieve specific documents. "
            "Overrides the node-level filters when non-empty."
        ),
    )
    top_k: int | None = Field(
        default=None, ge=1, le=100, description="Parameter to provide how many documents to retrieve."
    )
    user_id: str | None = Field(
        default=None,
        description="Parameter to provide the user identity for ACL-enforced retrieval on managed knowledge bases.",
        json_schema_extra={"is_accessible_to_agent": False},
    )


class BedrockKnowledgeBaseSearch(ConnectionNode):
    """Search tool for Amazon Bedrock Knowledge Bases.

    Calls the `bedrock-agent-runtime` Retrieve API: Bedrock embeds the query and performs the
    vector (or hybrid/managed) search server-side, so no local embedder or vector store is needed.
    Supports classic VECTOR knowledge bases (with optional search-type override, metadata filters,
    Bedrock reranking, and implicit model-inferred filters) as well as MANAGED knowledge bases
    (via `managed_search`, with `user_id` ACL context).

    Requires `bedrock:Retrieve` on the knowledge base for the caller, and boto3 >= 1.43.32. When
    reranking or implicit filtering is configured, Bedrock assumes the knowledge base's service role
    to invoke those models, so `bedrock:Rerank`/`bedrock:InvokeModel` belong on that role instead —
    see `BedrockRerankingConfig` for the exact scoping.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.SEMANTIC_SEARCH
    name: str = "bedrock-knowledgebase-search"
    description: str = DESCRIPTION
    connection: AWSConnection | None = None
    knowledge_base_id: str = Field(..., description="Knowledge base ID, or full ARN for managed/cross-account KBs.")
    top_k: int | None = Field(default=5, ge=1, le=100)
    search_type: Literal["DEFAULT", "HYBRID", "SEMANTIC"] = Field(
        default="DEFAULT",
        description="Vector search type override. DEFAULT lets Bedrock choose; HYBRID is store-dependent.",
    )
    filters: dict[str, Any] = Field(default_factory=dict, description="Default Bedrock RetrievalFilter.")
    score_threshold: float | None = Field(
        default=None,
        description="Client-side minimal relevance score. Results without a score (e.g. SQL rows) are kept.",
    )
    guardrail_config: BedrockGuardrailConfig | None = None
    reranking: BedrockRerankingConfig | None = None
    implicit_filter: BedrockImplicitFilterConfig | None = None
    managed_search: BedrockManagedSearchConfig | None = Field(
        default=None,
        description=(
            "Set for MANAGED-type knowledge bases; retrieval options (top_k, filters, reranking) are then sent "
            "as managedSearchConfiguration instead of vectorSearchConfiguration."
        ),
    )
    user_id: str | None = Field(default=None, description="Default user identity for managed-KB ACL filtering.")
    retrieval_config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Raw retrievalConfiguration escape hatch, sent verbatim. Overrides top_k/search_type/filters/"
            "reranking/implicit_filter/managed_search."
        ),
    )
    metadata_fields: list[str] | None = Field(
        default_factory=lambda: [
            "source_uri",
            "x-amz-bedrock-kb-source-uri",
            "x-amz-bedrock-kb-document-page-number",
            "title",
        ],
        description=(
            "Allowlist of document metadata keys to include in the formatted content (case-insensitive). "
            "Set to None to include every metadata field returned by the API."
        ),
    )
    timeout: float = 30
    input_schema: ClassVar[type[BedrockKnowledgeBaseSearchInputSchema]] = BedrockKnowledgeBaseSearchInputSchema

    def __init__(self, **kwargs):
        if kwargs.get("client") is None and kwargs.get("connection") is None:
            kwargs["connection"] = AWSConnection()
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def validate_search_configuration(self):
        if self.managed_search and (self.reranking or self.implicit_filter or self.search_type != "DEFAULT"):
            raise ValueError(
                "search_type, reranking and implicit_filter apply to VECTOR knowledge bases only and cannot be "
                "combined with managed_search."
            )
        return self

    def init_components(self, connection_manager: ConnectionManager | None = None):
        """Initialize the node and build the `bedrock-agent-runtime` client from the AWS connection."""
        connection_manager = connection_manager or ConnectionManager()
        super().init_components(connection_manager)
        if self.client is None:
            self.client = self._build_client()

    def _build_client(self) -> Any:
        from botocore.config import Config

        session = self.connection.get_boto3_session()
        return session.client(
            "bedrock-agent-runtime",
            config=Config(
                retries={"mode": "standard", "max_attempts": 3},
                connect_timeout=self.timeout,
                read_timeout=self.timeout,
                user_agent_extra="dynamiq",
            ),
        )

    @classmethod
    def _validate_filters(cls, filters: dict[str, Any], path: str = "filters") -> None:
        """Validate the RetrievalFilter shape client-side so the agent gets an actionable error."""
        if not isinstance(filters, dict) or len(filters) != 1:
            raise ToolExecutionException(
                f"Invalid Bedrock filter at '{path}': expected an object with exactly one operator key. "
                f"Allowed operators: {sorted(FILTER_LEAF_OPERATORS)} plus group operators "
                f"{sorted(FILTER_GROUP_OPERATORS)}. Please fix the filter and try again.",
                recoverable=True,
            )

        operator, value = next(iter(filters.items()))
        if operator in FILTER_GROUP_OPERATORS:
            if not isinstance(value, list) or len(value) < 2:
                raise ToolExecutionException(
                    f"Invalid Bedrock filter at '{path}.{operator}': {operator} requires a list of at least "
                    f"2 sub-filters. Please fix the filter and try again.",
                    recoverable=True,
                )
            for index, sub_filter in enumerate(value):
                cls._validate_filters(sub_filter, path=f"{path}.{operator}[{index}]")
        elif operator in FILTER_LEAF_OPERATORS:
            if not isinstance(value, dict) or "key" not in value or "value" not in value:
                raise ToolExecutionException(
                    f"Invalid Bedrock filter at '{path}.{operator}': expected "
                    '{"key": <metadata attribute>, "value": <value>}. Please fix the filter and try again.',
                    recoverable=True,
                )
        else:
            raise ToolExecutionException(
                f"Invalid Bedrock filter operator '{operator}' at '{path}'. Allowed operators: "
                f"{sorted(FILTER_LEAF_OPERATORS)} plus group operators {sorted(FILTER_GROUP_OPERATORS)}. "
                f"Please fix the filter and try again.",
                recoverable=True,
            )

    def _build_retrieval_configuration(
        self, top_k: int | None, filters: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if self.retrieval_config is not None:
            return self.retrieval_config

        if self.managed_search:
            managed_configuration = self.managed_search.to_api(top_k, filters)
            return {"managedSearchConfiguration": managed_configuration} if managed_configuration else None

        vector_configuration: dict[str, Any] = {}
        if top_k is not None:
            vector_configuration["numberOfResults"] = top_k
        if self.search_type != "DEFAULT":
            vector_configuration["overrideSearchType"] = self.search_type
        if filters:
            vector_configuration["filter"] = filters
        if self.reranking:
            vector_configuration["rerankingConfiguration"] = self.reranking.to_api()
        if self.implicit_filter:
            vector_configuration["implicitFilterConfiguration"] = self.implicit_filter.to_api()

        return {"vectorSearchConfiguration": vector_configuration} if vector_configuration else None

    def _build_retrieve_request(self, input_data: BedrockKnowledgeBaseSearchInputSchema) -> dict[str, Any]:
        query = input_data.query.strip()
        if not query:
            raise ToolExecutionException(
                "The 'query' parameter must be a non-empty string. Please provide a query and try again.",
                recoverable=True,
            )

        if self.retrieval_config is not None and (input_data.filters or input_data.top_k is not None):
            logger.warning(
                f"Tool {self.name} - {self.id}: retrieval_config is set; "
                f"agent-provided top_k/filters inputs are ignored."
            )

        filters = input_data.filters or self.filters
        if filters and self.retrieval_config is None:
            self._validate_filters(filters)
        top_k = input_data.top_k if input_data.top_k is not None else self.top_k
        user_id = input_data.user_id if input_data.user_id is not None else self.user_id

        request: dict[str, Any] = {
            "knowledgeBaseId": self.knowledge_base_id,
            "retrievalQuery": {"text": query},
        }
        retrieval_configuration = self._build_retrieval_configuration(top_k, filters)
        if retrieval_configuration:
            request["retrievalConfiguration"] = retrieval_configuration
        if self.guardrail_config:
            request["guardrailConfiguration"] = self.guardrail_config.to_api()
        if user_id:
            request["userContext"] = {"userId": user_id}
        return request

    @staticmethod
    def _resolve_source_uri(location: dict[str, Any]) -> str | None:
        location_type = location.get("type")
        if location_type in LOCATION_URI_FIELDS:
            payload_key, uri_key = LOCATION_URI_FIELDS[location_type]
            payload = location.get(payload_key) or {}
            return payload.get(uri_key)
        # Unknown location type: best-effort scan for a uri-like field so new types degrade gracefully.
        for value in location.values():
            if isinstance(value, dict):
                for uri_key in ("uri", "url", "id", "query"):
                    if value.get(uri_key):
                        return value[uri_key]
        return None

    def _extract_content(self, content: dict[str, Any], source_uri: str | None) -> str:
        content_type = content.get("type") or "TEXT"
        if content_type == "TEXT":
            return content.get("text", "") or ""
        if content_type == "ROW":
            rows = content.get("row") or []
            return "\n".join(f"{row.get('columnName')}: {row.get('columnValue')}" for row in rows)
        if content_type == "IMAGE":
            # byteContent is base64 — never inline it into agent context.
            return f"[image content from {source_uri or 'the knowledge base'}]"
        if content_type == "AUDIO":
            audio = content.get("audio") or {}
            return audio.get("transcription") or f"[audio content from {source_uri or 'the knowledge base'}]"
        if content_type == "VIDEO":
            video = content.get("video") or {}
            return video.get("summary") or f"[video content from {source_uri or 'the knowledge base'}]"
        return content.get("text", "") or ""

    def _to_document(self, result: dict[str, Any]) -> Document:
        content_block = result.get("content") or {}
        location = result.get("location") or {}
        source_uri = self._resolve_source_uri(location)

        metadata: dict[str, Any] = dict(result.get("metadata") or {})
        if source_uri:
            metadata["source_uri"] = source_uri
        if location.get("type"):
            metadata["location_type"] = location["type"]
        if content_block.get("type"):
            metadata["content_type"] = content_block["type"]
        if result.get("documentId"):
            metadata["document_id"] = result["documentId"]

        document_id = metadata.get("x-amz-bedrock-kb-chunk-id") or result.get("documentId")
        document_kwargs: dict[str, Any] = {"id": document_id} if document_id else {}
        return Document(
            content=self._extract_content(content_block, source_uri),
            metadata=metadata,
            score=result.get("score"),
            **document_kwargs,
        )

    def _apply_score_threshold(self, documents: list[Document]) -> list[Document]:
        if self.score_threshold is None:
            return documents
        return [doc for doc in documents if doc.score is None or doc.score >= self.score_threshold]

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

    def _parse_response(self, response: dict[str, Any]) -> dict[str, Any]:
        try:
            raw_results = response.get("retrievalResults") or []
            documents = self._apply_score_threshold([self._to_document(result) for result in raw_results])
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

        output: dict[str, Any] = {"content": content, "documents": documents}
        guardrail_action = response.get("guardrailAction")
        if guardrail_action:
            output["guardrail_action"] = guardrail_action
            if guardrail_action == "INTERVENED":
                output["content"] = (
                    f"{content}\n\nNote: a guardrail intervened; some or all results were blocked or masked."
                ).strip()

        logger.info(f"Tool {self.name} - {self.id}: retrieved {len(documents)} documents")
        return output

    def execute(
        self, input_data: BedrockKnowledgeBaseSearchInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")

        request = self._build_retrieve_request(input_data)

        try:
            response = self.client.retrieve(**request)
        except ToolExecutionException:
            raise
        except Exception as e:
            error_response = getattr(e, "response", None)
            error_code = (error_response.get("Error") or {}).get("Code") if isinstance(error_response, dict) else None
            error_details = f"{error_code}: {str(e)}" if error_code else str(e)
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {error_details}")
            raise ToolExecutionException(
                f"Bedrock Knowledge Base retrieval failed with error: {error_details}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        return self._parse_response(response)
