from typing import Any, ClassVar, Literal

from pydantic import BaseModel, Field

from dynamiq.connections import Dynamiq as DynamiqConnection
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION = (
    "Retrieves ACL-filtered facts from a Dynamiq knowledgebase graph for a natural-language query. "
    "Returns relationships (facts) connecting the entities the query is about, with their source "
    "documents. Access control is enforced by the Dynamiq API based on the connection credentials."
)


class DynamiqKnowledgebaseGraphSearchInputSchema(BaseModel):
    query: str = Field(..., description="Natural-language question to retrieve graph facts for.")
    filters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter to provide edge-property filters (e.g. ACL) to narrow results.",
    )
    limit: int | None = Field(default=None, description="Parameter to provide how many facts to retrieve.")
    entities: list[str] | None = Field(
        default=None,
        description="Optional explicit entity names to start traversal from. When omitted, entry entities "
        "are inferred from the query.",
    )
    entity_ids: list[str] | None = Field(
        default=None,
        description="Optional resolved entity ids to start from (unique, variant-proof). Takes precedence "
        "over 'entities' and the query.",
    )


class DynamiqKnowledgebaseGraphSearch(ConnectionNode):
    """Search tool that delegates GRAPH retrieval to the ACL-enforced Dynamiq knowledgebase API.

    The graph analogue of ``DynamiqKnowledgebaseVectorSearch``: instead of vector-similar chunks it
    returns facts (edges) relevant to the query. It calls
    ``POST {connection.url}/v1/knowledgebases/{knowledgebase_id}/graph-search`` and is a transparent
    pass-through — it validates the status and forwards the API response body as-is. The API owns the
    response shape (entity resolution, ACL-filtered traversal, and formatting run server-side); depending
    on server settings it returns ``content`` (the field the agent reads) plus optional extras such as
    ``documents`` / ``facts`` / ``source_documents`` / ``context``.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.SEMANTIC_SEARCH
    name: str = "dynamiq-knowledgebase-graph-search"
    description: str = DESCRIPTION
    connection: DynamiqConnection = Field(default_factory=DynamiqConnection)
    knowledgebase_id: str
    limit: int | None = None
    filters: dict[str, Any] = Field(default_factory=dict)
    timeout: float = 30
    user: str | None = None
    success_codes: list[int] = [200]
    input_schema: ClassVar[type[DynamiqKnowledgebaseGraphSearchInputSchema]] = (
        DynamiqKnowledgebaseGraphSearchInputSchema
    )

    def _build_url(self) -> str:
        return f"{self.connection.url.rstrip('/')}/v1/knowledgebases/{self.knowledgebase_id}/graph-search"

    def _build_request_kwargs(self, input_data: DynamiqKnowledgebaseGraphSearchInputSchema) -> dict[str, Any]:
        """Build the kwargs for the graph-search request. Input overrides node-level defaults."""
        filters = input_data.filters or self.filters
        limit = input_data.limit if input_data.limit is not None else self.limit
        entities = input_data.entities
        entity_ids = input_data.entity_ids
        user = self.user

        body: dict[str, Any] = {"query": input_data.query}
        if limit is not None:
            body["limit"] = limit
        if filters:
            body["filters"] = filters
        if entities:
            body["entities"] = entities
        if entity_ids:
            body["entity_ids"] = entity_ids
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
        """Validate status and forward the API response body as-is.

        The graph-search API owns the response shape: depending on server-side settings it may return
        ``content`` alone, or additionally ``documents`` / ``facts`` / ``source_documents`` / ``context``.
        This node is a transparent pass-through — it relays whatever the API produced. The contract
        guarantees a ``content`` string (the field the agent reads); the rest are optional extras.
        """
        if response.status_code not in self.success_codes:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results.")
            raise ToolExecutionException(
                f"Request failed with unexpected status code: {response.status_code} and response: {response.text}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        try:
            return response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to parse response. Error: {str(e)}")
            raise ToolExecutionException(
                f"Failed to parse response with error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

    def execute(
        self, input_data: DynamiqKnowledgebaseGraphSearchInputSchema, config: RunnableConfig = None, **kwargs
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
        self, input_data: DynamiqKnowledgebaseGraphSearchInputSchema, config: RunnableConfig = None, **kwargs
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
