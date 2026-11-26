from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import SpecStream
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.tools.utils import http_error_detail
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION_SPECSTREAM_SEARCH = """Searches the live web with SpecStream and returns verified sources with excerpts.

Key capabilities:
- Live web search returning titles, URLs, snippets, and page excerpts
- Supports default and fast tiers, news mode, country/language filters, and compact responses

Examples:
- {"query": "latest developments in quantum computing", "num": 5}
- {"query": "python 3.13 release notes", "speed": "fast", "num": 10}
"""


class SpecStreamSearchInputSchema(BaseModel):
    """Schema for SpecStream web search input parameters."""

    query: str = Field(description="Natural-language search query.")
    num: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of search results to return (1 to 20).",
    )
    speed: Literal["default", "fast"] | None = Field(
        default=None,
        description="Search tier: 'default' for full recall and passages, 'fast' for lower latency.",
    )
    news: bool | None = Field(
        default=None,
        description="Restrict search to recent news coverage.",
    )
    passages: bool | None = Field(
        default=None,
        description="Include extracted page excerpts alongside search snippets.",
    )
    country: str | None = Field(
        default=None,
        description="Optional ISO-3166 country code (e.g. 'US', 'FR').",
    )
    language: str | None = Field(
        default=None,
        description="Optional ISO-639 language code (e.g. 'en', 'fr').",
    )
    brief: str = Field(
        default="Searching the live web via SpecStream.",
        description="Very brief description of the search being performed.",
    )


class SpecStreamSearchTool(ConnectionNode):
    """A tool for performing live web searches using the SpecStream Web Search API."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "specstream-search"
    description: str = DESCRIPTION_SPECSTREAM_SEARCH
    action_type: ActionType = ActionType.WEB_SEARCH
    is_parallel_execution_allowed: bool = True
    connection: SpecStream

    num: int = Field(default=10, ge=1, le=20, description="Default number of results to return.")
    speed: Literal["default", "fast"] = Field(default="default", description="Default search tier.")
    passages: bool = Field(default=True, description="Include extracted passages by default.")
    news: bool = Field(default=False, description="Search news by default.")
    country: str | None = Field(default=None, description="Default country code.")
    language: str | None = Field(default=None, description="Default language code.")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[SpecStreamSearchInputSchema]] = SpecStreamSearchInputSchema

    def _build_search_payload(self, input_data: SpecStreamSearchInputSchema) -> tuple[dict[str, Any], str]:
        payload: dict[str, Any] = {
            "q": input_data.query,
            "num": input_data.num if input_data.num is not None else self.num,
            "speed": input_data.speed if input_data.speed is not None else self.speed,
            "passages": input_data.passages if input_data.passages is not None else self.passages,
        }
        news = input_data.news if input_data.news is not None else self.news
        if news:
            payload["news"] = True
        country = input_data.country if input_data.country is not None else self.country
        if country:
            payload["country"] = country
        language = input_data.language if input_data.language is not None else self.language
        if language:
            payload["language"] = language

        connection_url = urljoin(self.connection.url.rstrip("/") + "/", "web-search")
        return payload, connection_url

    def _wrap_search_exception(self, exc: Exception) -> ToolExecutionException:
        error = http_error_detail(exc)
        logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {error}")
        return ToolExecutionException(
            f"Tool '{self.name}' failed to retrieve search results. Error: {error}. "
            f"Please analyze the error and take appropriate action.",
            recoverable=True,
        )

    def _format_search_response(self, search_result: dict[str, Any]) -> dict[str, Any]:
        raw_sources = search_result.get("sources") or []
        sources = [s for s in raw_sources if isinstance(s, dict)]
        urls = [str(s.get("url") or "") for s in sources if s.get("url")]
        sources_with_url = [
            f"- [{s.get('title') or 'Untitled'}]({s.get('url') or ''})"
            for s in sources
            if s.get("url")
        ]
        formatted_items = []
        for s in sources:
            title = s.get("title") or "Untitled"
            url = s.get("url") or ""
            body = s.get("excerpt") or s.get("snippet") or ""
            formatted_items.append(f"### {title}\nURL: {url}\n{body}".strip())
        formatted_results = "\n\n".join(formatted_items)

        if self.is_optimized_for_agents:
            parts = ["## Sources", "\n".join(sources_with_url), "## Search Results", formatted_results]
            structured_sources = [
                {
                    "url": str(s.get("url") or ""),
                    "title": str(s.get("title") or ""),
                    "content": str(s.get("excerpt") or s.get("snippet") or ""),
                }
                for s in sources
            ]
            return {"content": "\n\n".join(parts), "urls": urls, "sources": structured_sources}

        return {
            "content": {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "urls": urls,
                "raw_response": search_result,
            }
        }

    def execute(
        self, input_data: SpecStreamSearchInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
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
            raise self._wrap_search_exception(e) from e

        return self._format_search_response(search_result)

    async def execute_async(
        self, input_data: SpecStreamSearchInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
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
            raise self._wrap_search_exception(e) from e

        return self._format_search_response(search_result)
