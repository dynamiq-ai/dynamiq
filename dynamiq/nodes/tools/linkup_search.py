from datetime import date
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Linkup
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION_LINKUP = """Searches the web in real time with Linkup and returns relevant, citable sources.

Key capabilities:
- Returns sources with title, URL and page content, or a sourced answer with citations
- Domain allow/deny lists and published date windows for precise result steering

Usage strategy:
- Use for current events, facts that may have changed, and anything that needs a verifiable source
- Use depth 'standard' for most queries, 'deep' for complex multi-step questions, 'fast' for low latency
- Set output_type to 'sourcedAnswer' when a direct answer with citations is enough

Examples:
- {"query": "latest developments in quantum computing", "max_results": 5}
- {"query": "pandas 3.0 release notes", "include_domains": ["pandas.pydata.org"]}
- {"query": "EU AI Act enforcement timeline", "depth": "deep", "output_type": "sourcedAnswer"}
- {"query": "hydrogen fuel startups funding", "from_date": "2026-01-01", "to_date": "2026-03-31"}
"""

DEFAULT_RESULT_TITLE = "Untitled result"


class LinkupInputSchema(BaseModel):
    """Schema for Linkup search input parameters."""

    query: str = Field(description="Natural-language search query.")
    depth: Literal["fast", "standard", "deep"] | None = Field(
        default=None,
        description=(
            "Search depth: 'fast' for lowest latency, 'standard' for most queries, "
            "'deep' for complex questions that need multiple search iterations."
        ),
    )
    output_type: Literal["searchResults", "sourcedAnswer"] | None = Field(
        default=None,
        description=(
            "'searchResults' returns a list of sources with content, "
            "'sourcedAnswer' returns a natural-language answer with its sources."
        ),
    )
    max_results: int | None = Field(
        default=None,
        ge=1,
        description="Maximum number of search results to return.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Whitelist of domains (e.g. ['arxiv.org', 'nature.com']). Results restricted to these domains.",
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="Blacklist of domains to omit from search results.",
    )
    from_date: date | None = Field(
        default=None,
        description="Only include results published on or after this ISO date (YYYY-MM-DD).",
    )
    to_date: date | None = Field(
        default=None,
        description="Only include results published on or before this ISO date (YYYY-MM-DD).",
    )
    include_images: bool | None = Field(
        default=None,
        description="Whether to include image results in the response.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    include_inline_citations: bool | None = Field(
        default=None,
        description="When output_type is 'sourcedAnswer', add inline citations to the answer.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    brief: str = Field(
        default="Searching the web for information.",
        description="Very brief description of the action being performed. Example: 'Search for AI research papers'.",
    )


class LinkupTool(ConnectionNode):
    """
    A tool for performing web searches using the Linkup API.

    This tool accepts various search parameters and returns relevant search results
    or a sourced answer, with options for filtering by date and domain.

    Attributes:
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        action_type (ActionType): The type of action this tool performs.
        connection (Linkup): The connection instance for the Linkup API.
        depth (str): Search depth ('fast', 'standard' or 'deep').
        output_type (str): Response format ('searchResults' or 'sourcedAnswer').
        max_results (int, optional): Maximum number of search results to return.
        include_domains (list[str], optional): List of domains to include.
        exclude_domains (list[str], optional): List of domains to exclude.
        from_date (date, optional): Include results published on or after this date.
        to_date (date, optional): Include results published on or before this date.
        include_images (bool): Include image results in the response.
        include_inline_citations (bool): Add inline citations to sourced answers.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "linkup-search"
    description: str = DESCRIPTION_LINKUP
    action_type: ActionType = ActionType.WEB_SEARCH
    is_parallel_execution_allowed: bool = True
    connection: Linkup

    depth: Literal["fast", "standard", "deep"] = Field(default="standard", description="Search depth.")
    output_type: Literal["searchResults", "sourcedAnswer"] = Field(
        default="searchResults", description="Response format returned by Linkup."
    )
    max_results: int | None = Field(default=None, ge=1, description="Maximum number of search results to return.")
    include_domains: list[str] | None = Field(default=None, description="List of domains to include in the search.")
    exclude_domains: list[str] | None = Field(default=None, description="List of domains to exclude from the search.")
    from_date: date | None = Field(default=None, description="Only include results published on or after this date.")
    to_date: date | None = Field(default=None, description="Only include results published on or before this date.")
    include_images: bool = Field(default=False, description="Include image results in the response.")
    include_inline_citations: bool = Field(default=False, description="Add inline citations to sourced answers.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[LinkupInputSchema]] = LinkupInputSchema

    @staticmethod
    def _get_sources(search_result: dict[str, Any]) -> list[dict[str, Any]]:
        """Return text sources from either a searchResults or a sourcedAnswer response."""
        if "answer" in search_result:
            return search_result.get("sources") or []
        return [r for r in search_result.get("results") or [] if r.get("type", "text") == "text"]

    @staticmethod
    def _format_search_results(sources: list[dict[str, Any]]) -> str:
        """
        Formats the search results into a human-readable string.

        Args:
            sources (list[dict[str, Any]]): The raw search results.

        Returns:
            str: A formatted string containing the search results.
        """
        if not sources:
            return "No results returned by web search."

        formatted_results = []
        for index, source in enumerate(sources, start=1):
            title = source.get("name") or DEFAULT_RESULT_TITLE
            url = source.get("url")
            content = (source.get("content") or source.get("snippet") or "").strip()

            formatted_results.append(f"### Result {index}: {title}")
            formatted_results.append(f"- URL: [{url}]({url})" if url else "- URL: Not available")
            if content:
                formatted_results.append(f"- Content: {content}")
            formatted_results.append("")

        return "\n".join(formatted_results).strip()

    @staticmethod
    def _format_sources(sources: list[dict[str, Any]]) -> list[str]:
        """Create markdown-friendly source list."""
        formatted_sources = []
        for source in sources:
            title = source.get("name") or DEFAULT_RESULT_TITLE
            url = source.get("url")
            formatted_sources.append(f"- [{title}]({url})" if url else f"- {title}")
        return formatted_sources

    def _build_search_payload(self, input_data: LinkupInputSchema) -> tuple[dict[str, Any], str]:
        """Return (payload, connection_url)."""

        def resolve(field: str) -> Any:
            value = getattr(input_data, field)
            return value if value is not None else getattr(self, field)

        payload = {
            "q": input_data.query,
            "depth": resolve("depth"),
            "outputType": resolve("output_type"),
            "maxResults": resolve("max_results"),
            "includeDomains": resolve("include_domains"),
            "excludeDomains": resolve("exclude_domains"),
            "fromDate": resolve("from_date"),
            "toDate": resolve("to_date"),
            "includeImages": resolve("include_images"),
        }
        if payload["outputType"] == "sourcedAnswer":
            payload["includeInlineCitations"] = resolve("include_inline_citations")

        for date_field in ("fromDate", "toDate"):
            if isinstance(payload.get(date_field), date):
                payload[date_field] = payload[date_field].isoformat()

        payload = {k: v for k, v in payload.items() if v is not None}

        connection_url = urljoin(self.connection.url, "search")
        return payload, connection_url

    def _wrap_search_exception(self, exc: Exception) -> ToolExecutionException:
        logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(exc)}")
        return ToolExecutionException(
            f"Tool '{self.name}' failed to retrieve search results. Error: {str(exc)}. "
            f"Please analyze the error and take appropriate action.",
            recoverable=True,
        )

    def _format_search_response(self, search_result: dict) -> dict[str, Any]:
        answer = search_result.get("answer")
        sources = self._get_sources(search_result)
        formatted_results = self._format_search_results(sources)
        sources_with_url = self._format_sources(sources)
        urls = [s.get("url") for s in sources]

        if self.is_optimized_for_agents:
            result_parts = ["## Sources", "\n".join(sources_with_url)]
            if answer:
                result_parts.extend(["## Answer", answer])
            result_parts.extend(["## Search Results", formatted_results])
            result = "\n\n".join(result_parts)

            structured_sources = [
                {
                    "url": s.get("url") or "",
                    "title": s.get("name") or "",
                    "content": s.get("content") or s.get("snippet") or "",
                }
                for s in sources
            ]

            output = {"content": result, "urls": urls, "sources": structured_sources}
        else:
            result = {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "urls": urls,
                "answer": answer,
                "raw_response": search_result,
            }
            output = {"content": result}

        return output

    def execute(self, input_data: LinkupInputSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """
        Executes the search using the Linkup API and returns the formatted results.

        Input parameters override node parameters when provided.
        """

        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        payload, connection_url = self._build_search_payload(input_data)

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                json=payload,
                headers=self.connection.headers,
            )
            response.raise_for_status()
            search_result = response.json()
        except Exception as e:
            raise self._wrap_search_exception(e)

        return self._format_search_response(search_result)

    async def execute_async(
        self, input_data: LinkupInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Native async execution path mirroring ``execute``."""

        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        payload, connection_url = self._build_search_payload(input_data)
        client = await self.get_async_client()

        try:
            response = await client.request(
                method=self.connection.method,
                url=connection_url,
                json=payload,
                headers=self.connection.headers,
            )
            response.raise_for_status()
            search_result = response.json()
        except Exception as e:
            raise self._wrap_search_exception(e)

        return self._format_search_response(search_result)
