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
    "Retrieves relevant documents from a Dynamiq knowledge base. "
    "Access control is enforced by the Dynamiq API based on the connection credentials."
)


class DynamiqKnowledgebaseVectorStoreRetrieverInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a query to retrieve documents.")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter to provide filters to apply for retrieving specific documents.",
    )
    top_k: int | None = Field(default=None, description="Parameter to provide how many documents to retrieve.")
    similarity_threshold: float | None = Field(
        default=None,
        description="Parameter to provide minimal similarity or maximal distance score for retrieved documents.",
    )


class DynamiqKnowledgebaseVectorStoreRetriever(ConnectionNode):
    """Retriever tool that delegates retrieval to the ACL-enforced Dynamiq knowledge base API.

    Unlike the local vector-store retrievers, this node does not query a vector store directly.
    It calls ``POST {connection.url}/v1/knowledgebases/{knowledge_base_id}/vector-search`` and forwards
    the response (the API performs embedding + ACL-filtered retrieval and formatting), so it is a
    drop-in replacement for ``VectorStoreRetriever`` at the tool level.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.SEMANTIC_SEARCH
    name: str = "dynamiq-knowledgebase-vector-store-retriever"
    description: str = DESCRIPTION
    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    knowledge_base_id: str
    top_k: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    similarity_threshold: float | None = None
    timeout: float = 30
    success_codes: list[int] = [200]
    input_schema: ClassVar[type[DynamiqKnowledgebaseVectorStoreRetrieverInputSchema]] = (
        DynamiqKnowledgebaseVectorStoreRetrieverInputSchema
    )

    def _build_url(self) -> str:
        return f"{self.connection.url.rstrip('/')}/v1/knowledgebases/{self.knowledge_base_id}/vector-search"

    def _build_request_kwargs(self, input_data: DynamiqKnowledgebaseVectorStoreRetrieverInputSchema) -> dict[str, Any]:
        """Build the kwargs for the search request. Input overrides node-level defaults."""
        filters = input_data.filters or self.filters
        top_k = input_data.top_k if input_data.top_k is not None else self.top_k
        similarity_threshold = (
            input_data.similarity_threshold
            if input_data.similarity_threshold is not None
            else self.similarity_threshold
        )

        body: dict[str, Any] = {"query": input_data.query}
        if top_k is not None:
            body["top_k"] = top_k
        if filters:
            body["filters"] = filters
        if similarity_threshold is not None:
            body["similarity_threshold"] = similarity_threshold

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

        data = response.json()
        # Tolerate either {"documents": [...], "content": ...} or a bare list of documents.
        raw_documents = (data.get("documents") or []) if isinstance(data, dict) else (data or [])
        documents = [self._to_document(item) for item in raw_documents]

        content = data.get("content") if isinstance(data, dict) else None
        if content is None:
            content = "\n\n".join((doc.content or "") for doc in documents)

        logger.info(f"Tool {self.name} - {self.id}: retrieved {len(documents)} documents")
        return {"content": content, "documents": documents}

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
        self, input_data: DynamiqKnowledgebaseVectorStoreRetrieverInputSchema, config: RunnableConfig = None, **kwargs
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
        self, input_data: DynamiqKnowledgebaseVectorStoreRetrieverInputSchema, config: RunnableConfig = None, **kwargs
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
