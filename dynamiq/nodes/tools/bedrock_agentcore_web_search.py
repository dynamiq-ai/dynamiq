import asyncio
import json
from datetime import date
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from dynamiq.connections import AWS as AWSConnection
from dynamiq.connections.agentcore import AgentCoreGatewayClient, AgentCoreGatewayError, normalize_gateway_url
from dynamiq.connections.managers import ConnectionManager
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION_AGENTCORE_WEB_SEARCH = """Searches the web with AWS Bedrock AgentCore Web Search and returns cited results.

Key capabilities:
- Returns the passage of each page that answers the query, with its title, URL and publication date
- Domain allow/deny lists and published date windows for precise result steering

Usage strategy:
- Keep the query short and specific: at most 200 characters
- Cite the URL of every result you use in your answer

Examples:
- {"query": "latest FDA guidance on clinical decision support software", "max_results": 5}
- {"query": "influenza surveillance weekly report", "include_domains": ["cdc.gov"]}
- {"query": "EU AI Act enforcement timeline", "start_published_date": "2026-01-01"}
"""

WEB_SEARCH_TOOL_NAME = "WebSearch"
# The gateway exposes a target's tools as "<target name>___<tool name>".
GATEWAY_TOOL_NAME_DELIMITER = "___"
MAX_QUERY_LENGTH = 200
MAX_RESULTS = 25
MAX_DOMAIN_FILTER_ENTRIES = 100
DEFAULT_RESULT_TITLE = "Untitled result"


class BedrockAgentCoreWebSearchInputSchema(BaseModel):
    """Schema for AgentCore Web Search input parameters."""

    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(
        ...,
        min_length=1,
        max_length=MAX_QUERY_LENGTH,
        description="Natural-language search query of at most 200 characters.",
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=MAX_RESULTS,
        description="Number of results to return, from 1 to 25.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        max_length=MAX_DOMAIN_FILTER_ENTRIES,
        description="Only return results from these domains, e.g. ['cdc.gov']. A domain also matches its subdomains.",
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        max_length=MAX_DOMAIN_FILTER_ENTRIES,
        description="Never return results from these domains.",
    )
    start_published_date: date | None = Field(
        default=None,
        description="Only return pages published on or after this date (YYYY-MM-DD).",
    )
    end_published_date: date | None = Field(
        default=None,
        description="Only return pages published on or before this date (YYYY-MM-DD).",
    )
    brief: str = Field(
        default="Searching the web for information.",
        description="Very brief description of the action being performed. Example: 'Search for AI research papers'.",
    )


class BedrockAgentCoreWebSearchTool(ConnectionNode):
    """
    A tool for web search through an AWS Bedrock AgentCore Gateway with a Web Search target.

    AWS serves the searches: the gateway calls the AWS-managed Web Search connector with its own IAM role, so no
    third-party search provider or API key is involved. The connection's credentials sign every request and need
    bedrock-agentcore:InvokeGateway on the gateway. Filters need the target's connector version 1.2.0 or later, and
    domain filters set on the target apply to every search: a request can narrow them but never widen them.

    Attributes:
        connection (AWS): AWS credentials and region used to sign the gateway requests.
        gateway_url (str): MCP endpoint of the gateway.
        target_name (str, optional): Gateway target that holds the Web Search connector. When unset, the tool is
            looked up on the gateway, which costs one more request per search.
        max_results (int, optional): Number of results to return. The service returns 10 when unset.
        include_domains (list[str], optional): Domains to restrict results to.
        exclude_domains (list[str], optional): Domains to drop results from.
        timeout (float): Timeout in seconds for connecting to the gateway and for each read from it.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "agentcore-web-search"
    description: str = DESCRIPTION_AGENTCORE_WEB_SEARCH
    action_type: ActionType = ActionType.WEB_SEARCH
    is_parallel_execution_allowed: bool = True
    connection: AWSConnection

    gateway_url: str = Field(..., description="MCP endpoint URL of the AgentCore Gateway.")
    target_name: str | None = Field(
        default=None, description="Name of the gateway target that holds the Web Search connector."
    )
    max_results: int | None = Field(default=None, ge=1, le=MAX_RESULTS, description="Number of results to return.")
    include_domains: list[str] | None = Field(
        default=None, max_length=MAX_DOMAIN_FILTER_ENTRIES, description="Domains to restrict results to."
    )
    exclude_domains: list[str] | None = Field(
        default=None, max_length=MAX_DOMAIN_FILTER_ENTRIES, description="Domains to drop results from."
    )
    timeout: float = Field(
        default=30.0, gt=0, description="Timeout in seconds for connecting to the gateway and for each read from it."
    )

    _gateway_client: AgentCoreGatewayClient | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[BedrockAgentCoreWebSearchInputSchema]] = BedrockAgentCoreWebSearchInputSchema

    @field_validator("gateway_url")
    @classmethod
    def validate_gateway_url(cls, gateway_url: str) -> str:
        return normalize_gateway_url(gateway_url)

    def init_components(self, connection_manager: ConnectionManager | None = None):
        super().init_components(connection_manager)
        self._get_gateway_client()

    def _get_gateway_client(self) -> AgentCoreGatewayClient:
        if self._gateway_client is None:
            self._gateway_client = AgentCoreGatewayClient(self.connection, self.gateway_url, self.timeout)
        return self._gateway_client

    def _get_tool_name(self) -> str | None:
        if self.target_name:
            return f"{self.target_name}{GATEWAY_TOOL_NAME_DELIMITER}{WEB_SEARCH_TOOL_NAME}"
        return None

    @staticmethod
    def _find_tool_name(tool_names: list[str]) -> str:
        """Return the one Web Search tool among the tools the gateway lists."""
        suffix = f"{GATEWAY_TOOL_NAME_DELIMITER}{WEB_SEARCH_TOOL_NAME}"
        candidates = sorted(name for name in tool_names if name == WEB_SEARCH_TOOL_NAME or name.endswith(suffix))
        if not candidates:
            raise AgentCoreGatewayError(
                "The gateway has no Web Search tool. Add a Web Search target to it, or set target_name."
            )
        if len(candidates) > 1:
            raise AgentCoreGatewayError(
                f"The gateway has several Web Search tools ({', '.join(candidates)}). Set target_name to choose one."
            )
        return candidates[0]

    def _build_arguments(self, input_data: BedrockAgentCoreWebSearchInputSchema) -> dict[str, Any]:
        """Build the WebSearch tool arguments. Input parameters override node parameters."""

        def resolve(field: str) -> Any:
            value = getattr(input_data, field)
            return value if value is not None else getattr(self, field)

        arguments: dict[str, Any] = {"query": input_data.query}
        if max_results := resolve("max_results"):
            arguments["maxResults"] = max_results

        # An empty include list is a different request from no include list, so empty lists are left out.
        domain_filter = {
            key: domains
            for key, domains in (("include", resolve("include_domains")), ("exclude", resolve("exclude_domains")))
            if domains
        }
        # The bounds are inclusive ISO-8601 UTC timestamps, so each date is widened to cover the whole day.
        published_date_filter = {}
        if input_data.start_published_date:
            published_date_filter["from"] = f"{input_data.start_published_date.isoformat()}T00:00:00Z"
        if input_data.end_published_date:
            published_date_filter["to"] = f"{input_data.end_published_date.isoformat()}T23:59:59Z"

        # Targets before connector version 1.2.0 accept no filters, so the key is only sent when a filter is set.
        filters = {}
        if domain_filter:
            filters["domainFilter"] = domain_filter
        if published_date_filter:
            filters["publishedDateFilter"] = published_date_filter
        if filters:
            arguments["filters"] = filters
        return arguments

    @staticmethod
    def _decode_payload(tool_result: dict[str, Any]) -> dict[str, Any]:
        """Return the search payload, which the connector sends as JSON in the first text content block."""
        text = next(
            (
                block["text"]
                for block in tool_result.get("content") or []
                if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
            ),
            None,
        )
        if tool_result.get("isError"):
            raise AgentCoreGatewayError(f"Web Search returned an error: {text or 'no details'}")
        if text is None:
            raise AgentCoreGatewayError("Web Search returned no text content.")

        try:
            payload = json.loads(text)
        except ValueError:
            payload = None
        if not isinstance(payload, dict):
            raise AgentCoreGatewayError(f"Web Search returned text that is not a JSON object: {text[:200]}")
        return payload

    @staticmethod
    def _format_search_results(results: list[dict[str, Any]]) -> str:
        if not results:
            return "No results returned by web search."

        formatted_results = []
        for index, result in enumerate(results, start=1):
            url = result.get("url")
            formatted_results.append(f"### Result {index}: {result.get('title') or DEFAULT_RESULT_TITLE}")
            formatted_results.append(f"- URL: [{url}]({url})" if url else "- URL: Not available")
            if published_date := result.get("publishedDate"):
                formatted_results.append(f"- Published: {published_date}")
            if text := (result.get("text") or "").strip():
                formatted_results.append(f"- Content: {text}")
            formatted_results.append("")

        return "\n".join(formatted_results).strip()

    @staticmethod
    def _format_sources(results: list[dict[str, Any]]) -> list[str]:
        sources = []
        for result in results:
            title = result.get("title") or DEFAULT_RESULT_TITLE
            url = result.get("url")
            sources.append(f"- [{title}]({url})" if url else f"- {title}")
        return sources

    def _format_search_response(self, payload: dict[str, Any]) -> dict[str, Any]:
        results = [result for result in payload.get("results") or [] if isinstance(result, dict)]
        formatted_results = self._format_search_results(results)
        sources_with_url = self._format_sources(results)
        urls = [result["url"] for result in results if result.get("url")]

        if self.is_optimized_for_agents:
            content = "\n\n".join(["## Sources", "\n".join(sources_with_url), "## Search Results", formatted_results])
            sources = [
                {
                    "url": result.get("url") or "",
                    "title": result.get("title") or "",
                    "published_date": result.get("publishedDate") or "",
                    "text": result.get("text") or "",
                }
                for result in results
            ]
            return {"content": content, "urls": urls, "sources": sources}

        return {
            "content": {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "urls": urls,
                "raw_response": payload,
            }
        }

    def _wrap_search_exception(self, exc: Exception) -> ToolExecutionException:
        error = str(exc).rstrip(".")
        logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {error}")
        return ToolExecutionException(
            f"Tool '{self.name}' failed to retrieve search results. Error: {error}. "
            f"Please analyze the error and take appropriate action.",
            recoverable=True,
        )

    def execute(
        self, input_data: BedrockAgentCoreWebSearchInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Search the web through the gateway. Input parameters override node parameters."""
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        arguments = self._build_arguments(input_data)
        try:
            with self._get_gateway_client().session() as session:
                tool_name = self._get_tool_name() or self._find_tool_name(session.list_tool_names())
                payload = self._decode_payload(session.call_tool(tool_name, arguments))
        except Exception as e:
            raise self._wrap_search_exception(e)

        return self._format_search_response(payload)

    async def execute_async(
        self, input_data: BedrockAgentCoreWebSearchInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Native async execution path mirroring ``execute``."""
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        arguments = self._build_arguments(input_data)
        try:
            # Creating the client reads the AWS configuration files.
            gateway_client = self._gateway_client or await asyncio.to_thread(self._get_gateway_client)
            async with gateway_client.async_session() as session:
                tool_name = self._get_tool_name() or self._find_tool_name(await session.list_tool_names())
                payload = self._decode_payload(await session.call_tool(tool_name, arguments))
        except Exception as e:
            raise self._wrap_search_exception(e)

        return self._format_search_response(payload)
