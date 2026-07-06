from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types import Document
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION = (
    "Retrieves relevant documents from a Dynamiq knowledgebase. "
    "Access control is enforced by the Dynamiq API based on the connection credentials."
)


class DynamiqKnowledgebaseVectorSearchInputSchema(BaseModel):
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


class DynamiqKnowledgebaseVectorSearch(ConnectionNode):
    """Search tool that delegates retrieval to the ACL-enforced Dynamiq knowledgebase API.

    Unlike the local vector-store retrievers, this node does not query a vector store directly.
    It calls ``POST {connection.url}/v1/knowledgebases/{knowledgebase_id}/vector-search`` and forwards
    the response (the API performs embedding + ACL-filtered retrieval and formatting), so it is a
    drop-in replacement for ``VectorStoreRetriever`` at the tool level.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.SEMANTIC_SEARCH
    name: str = "dynamiq-knowledgebase-vector-search"
    description: str = DESCRIPTION
    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    knowledgebase_id: str
    limit: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    similarity_threshold: float | None = None
    metadata_fields: list[str] | None = Field(
        default_factory=lambda: ["title", "source_url", "url", "source"],
        description=(
            "Allowlist of document metadata keys to include in the formatted content (case-insensitive). "
            "Set to None to include every metadata field returned by the API."
        ),
    )
    timeout: float = 30
    user: str | None = None
    success_codes: list[int] = [200]
    input_schema: ClassVar[type[DynamiqKnowledgebaseVectorSearchInputSchema]] = (
        DynamiqKnowledgebaseVectorSearchInputSchema
    )

    def _build_url(self) -> str:
        return f"{self.connection.url.rstrip('/')}/v1/knowledgebases/{self.knowledgebase_id}/vector-search"

    def _build_request_kwargs(self, input_data: DynamiqKnowledgebaseVectorSearchInputSchema) -> dict[str, Any]:
        """Build the kwargs for the search request. Input overrides node-level defaults."""
        filters = input_data.filters or self.filters
        limit = input_data.limit if input_data.limit is not None else self.limit
        similarity_threshold = (
            input_data.similarity_threshold
            if input_data.similarity_threshold is not None
            else self.similarity_threshold
        )
        user = self.user

        body: dict[str, Any] = {"query": input_data.query}
        if limit is not None:
            body["limit"] = limit
        if filters:
            body["filters"] = filters
        if similarity_threshold is not None:
            body["similarity_threshold"] = similarity_threshold
        if user is not None:
            body["user"] = user

        return {
            "method": "POST",
            "url": self._build_url(),
            "headers": self.connection.headers,
            "json": body,
            "timeout": self.timeout,
        }

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """Validate status and forward the API response as retriever-shaped output."""
        if response.status_code not in self.success_codes:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results.")
            raise ToolExecutionException(
                f"Request failed with unexpected status code: {response.status_code} and response: {response.text}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        try:
            data = response.json()
            # The API always returns the documents in a data envelope: {"data": [...documents...]}.
            raw_documents = data.get("data") or []
            documents = [self._to_document(item) for item in raw_documents]
            content = self._format_content(documents)
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to parse response. Error: {str(e)}")
            raise ToolExecutionException(
                f"Failed to parse response with error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        logger.info(f"Tool {self.name} - {self.id}: retrieved {len(documents)} documents")
        return {"content": content, "documents": documents}

    def _format_content(self, documents: list[Document]) -> str:
        """Format documents into a numbered, source-labelled block (mirrors VectorStoreRetriever).

        Only metadata keys in ``self.metadata_fields`` are emitted (case-insensitive) so internal
        fields (embeddings, chunk ids, ACL data, ...) are not leaked. ``metadata_fields=None`` emits all.
        """
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

    @staticmethod
    def _to_document(item: Any) -> Document:
        if isinstance(item, Document):
            return item
        return Document(
            id=item.get("id") or None,
            content=item.get("content", "") or "",
            metadata=item.get("metadata"),
            score=item.get("score"),
        )

    def execute(
        self, input_data: DynamiqKnowledgebaseVectorSearchInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")

        request_kwargs = self._build_request_kwargs(input_data)

        try:
            response = self.client.request(**request_kwargs)
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise ToolExecutionException(
                f"Request failed with error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        return self._parse_response(response)

    async def execute_async(
        self, input_data: DynamiqKnowledgebaseVectorSearchInputSchema, config: RunnableConfig = None, **kwargs
    ) -> dict[str, Any]:
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)
        logger.info(f"Tool {self.name} - {self.id}: started with INPUT DATA:\n{input_data.model_dump()}")

        request_kwargs = self._build_request_kwargs(input_data)
        client = await self.get_async_client()

        try:
            response = await client.request(**request_kwargs)
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise ToolExecutionException(
                f"Request failed with error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        return self._parse_response(response)
