from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Keenable
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.tools.utils import http_error_detail
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION_KEENABLE = """Searches the web with Keenable and returns sources with title, URL and a text excerpt.

Key capabilities:
- Fast web search built for AI agents, usable without an API key
- Returns title, URL, excerpt and publication date when available

Usage strategy:
- Use for current events, documentation lookups and facts that need a verifiable source
- Keep queries specific; set max_results to limit the number of sources

Examples:
- {"query": "latest developments in quantum computing", "max_results": 5}
- {"query": "pydantic v2 model_validator docs"}
"""

DEFAULT_RESULT_TITLE = "Untitled result"


class KeenableInputSchema(BaseModel):
    """Schema for Keenable search input parameters."""

    query: str = Field(description="Natural-language search query.")
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of search results to return.",
    )
    brief: str = Field(
        default="Searching the web for information.",
        description="Very brief description of the action being performed. Example: 'Search for AI research papers'.",
    )


class KeenableTool(ConnectionNode):
    """
    A tool for performing web searches using the Keenable API.

    Attributes:
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        action_type (ActionType): The type of action this tool performs.
        connection (Keenable): The connection instance for the Keenable API.
        max_results (int): Maximum number of search results to return.
        max_content_chars (int): Excerpts longer than this are truncated.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "keenable-search"
    description: str = DESCRIPTION_KEENABLE
    action_type: ActionType = ActionType.WEB_SEARCH
    is_parallel_execution_allowed: bool = True
    connection: Keenable

    max_results: int = Field(default=5, ge=1, le=20, description="Maximum number of search results to return.")
    max_content_chars: int = Field(default=1000, ge=1, description="Maximum characters kept per result excerpt.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[KeenableInputSchema]] = KeenableInputSchema

    def _content(self, result: dict[str, Any]) -> str:
        """Prefer ``description``; fall back to ``snippet`` (which can be long) and truncate."""
        text = (result.get("description") or "").strip() or (result.get("snippet") or "").strip()
        if len(text) > self.max_content_chars:
            text = text[: self.max_content_chars].rstrip() + "..."
        return text

    def _sources(self, search_result: dict[str, Any]) -> list[dict[str, Any]]:
        sources = []
        for r in search_result.get("results") or []:
            url = r.get("url")
            if not url:
                continue
            sources.append(
                {
                    "url": url,
                    "title": r.get("title") or DEFAULT_RESULT_TITLE,
                    "content": self._content(r),
                    "published_at": r.get("published_at"),
                }
            )
        return sources

    @staticmethod
    def _format_search_results(sources: list[dict[str, Any]]) -> str:
        if not sources:
            return "No results returned by web search."
        lines = []
        for index, s in enumerate(sources, start=1):
            lines.append(f"### Result {index}: {s['title']}")
            lines.append(f"- URL: [{s['url']}]({s['url']})")
            if s.get("published_at"):
                lines.append(f"- Published: {s['published_at']}")
            if s["content"]:
                lines.append(f"- Content: {s['content']}")
            lines.append("")
        return "\n".join(lines).strip()

    def _build_search_payload(self, input_data: KeenableInputSchema) -> tuple[dict[str, Any], str]:
        """Return (payload, connection_url)."""
        max_results = input_data.max_results if input_data.max_results is not None else self.max_results
        payload = {"query": input_data.query, "max_results": max_results}
        return payload, urljoin(self.connection.url, self.connection.search_path)

    def _wrap_search_exception(self, exc: Exception) -> ToolExecutionException:
        error = http_error_detail(exc)
        logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {error}")
        return ToolExecutionException(
            f"Tool '{self.name}' failed to retrieve search results. Error: {error}. "
            f"Please analyze the error and take appropriate action.",
            recoverable=True,
        )

    def _format_search_response(self, search_result: dict) -> dict[str, Any]:
        sources = self._sources(search_result)
        formatted_results = self._format_search_results(sources)
        sources_with_url = [f"- [{s['title']}]({s['url']})" for s in sources]
        urls = [s["url"] for s in sources]

        if self.is_optimized_for_agents:
            result = "\n\n".join(["## Sources", "\n".join(sources_with_url), "## Search Results", formatted_results])
            structured = [{"url": s["url"], "title": s["title"], "content": s["content"]} for s in sources]
            return {"content": result, "urls": urls, "sources": structured}
        return {
            "content": {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "urls": urls,
                "raw_response": search_result,
            }
        }

    def execute(
        self, input_data: KeenableInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Executes the search using the Keenable API and returns the formatted results."""
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
        self, input_data: KeenableInputSchema, config: RunnableConfig | None = None, **kwargs
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
