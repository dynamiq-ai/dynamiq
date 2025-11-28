from copy import deepcopy
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Firecrawl
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_FIRECRAWL_SEARCH = """Search the web with Firecrawl across web, news, and image verticals.

What it does:
- Returns SERP results with geo/time filters and category biasing (github/research/pdf)
- Lets you control result count, recency, geo bias, timeout, and URL validation in one call

Parameters (FirecrawlSearchInput):
- `query` (required): search string.
- `limit` (1-100): max results to return (default 5).
- `sources` (list of objects): choose verticals with optional per-source settings:
  - Web: {"type":"web","tbs":"qdr:d","location":"San Francisco,California,United States"}
  - News: {"type":"news"}
  - Images: {"type":"images"}
- `categories` (list): bias results toward content types, e.g., ["github","research","pdf"].
- `tbs`: time filter (qdr:h/d/w/m/y or custom ranges like cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY).
- `location`: city/state/country string for geo bias.
- `country`: ISO country code for geo targeting (e.g., "US").
- `timeout`: request timeout in milliseconds (default 60000).
- `ignoreInvalidURLs`: true to drop invalid links from the response.

Examples:
{"query": "firecrawl docs", "limit": 5}
{"query": "openai funding", "sources": [{"type": "news"}], "limit": 3}
{"query": "sunset wallpaper imagesize:1920x1080", "sources": [{"type": "images"}], "limit": 5}
{"query": "langchain github", "categories": ["github"], "tbs": "qdr:w"}"""


class SourceWeb(BaseModel):
    type: Literal["web"] = Field(default="web", description="Web search results.")
    tbs: str | None = Field(
        default=None,
        description="Time-based search filter (qdr:h/d/w/m/y or "
        "custom ranges like cdr:1,cd_min:MM/DD/YYYY,cd_max:MM/DD/YYYY).",
    )
    location: str | None = Field(default=None, description="Location bias for this web source.")


class SourceImages(BaseModel):
    type: Literal["images"] = Field(default="images", description="Image search results.")


class SourceNews(BaseModel):
    type: Literal["news"] = Field(default="news", description="News search results.")


SourceType = SourceWeb | SourceImages | SourceNews


CategoryType = Literal["github", "research", "pdf"]


class FirecrawlSearchInput(BaseModel):
    """Schema exposed to agents using the Firecrawl search tool."""

    model_config = ConfigDict(populate_by_name=True)

    query: str = Field(..., description="Search query to execute on Firecrawl.")
    limit: int | None = Field(default=None, ge=1, le=100, description="Maximum number of search results to return.")
    sources: list[SourceType] | None = Field(
        default=None, description="Result types to fetch: web/news/images with optional per-source options."
    )
    categories: list[CategoryType] | None = Field(
        default=None,
        description="Optional category filters for GitHub/research/PDF focused searches.",
    )
    tbs: str | None = Field(
        default=None,
        description="Time-based search filter (qdr:h/d/w/m/y or custom ranges like "
        "cdr:1,cd_min:12/1/2024,cd_max:12/31/2024).",
    )
    location: str | None = Field(
        default=None,
        description="Location string for search results (e.g., 'San Francisco,California,United States').",
    )
    country: str | None = Field(default=None, description="ISO country code for geo-targeting (e.g., 'US').")
    timeout: int | None = Field(default=None, description="Request timeout in milliseconds.")
    ignore_invalid_urls: bool | None = Field(
        default=None,
        alias="ignoreInvalidURLs",
        description="Exclude invalid URLs from search results when piping into other endpoints.",
    )


class FirecrawlSearchTool(ConnectionNode):
    """A tool for performing Firecrawl searches."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Firecrawl Search Tool"
    description: str = DESCRIPTION_FIRECRAWL_SEARCH
    connection: Firecrawl
    query: str | None = None

    limit: int = Field(default=5, ge=1, le=100, description="Number of results to request.")
    sources: list[SourceType] = Field(default_factory=lambda: [SourceWeb()], description="Result verticals to include.")
    categories: list[CategoryType] = Field(default_factory=list, description="Optional category filters.")
    tbs: str | None = Field(default=None, description="Time-based search filter passed to Firecrawl.")
    location: str | None = Field(default=None, description="Geographic bias for the search query.")
    country: str = Field(default="US", description="ISO country code for geo-targeting.")
    timeout: int = Field(default=60000, description="Request timeout in milliseconds.")
    ignore_invalid_urls: bool = Field(
        default=False,
        alias="ignoreInvalidURLs",
        description="Exclude invalid URLs from results to reduce downstream errors.",
    )
    input_schema: ClassVar[type[FirecrawlSearchInput]] = FirecrawlSearchInput
    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    MAX_DESCRIPTION_CHARS: ClassVar[int] = 300
    MAX_CONTENT_CHARS: ClassVar[int] = 800

    @staticmethod
    def _truncate(text: str | None, limit: int) -> str:
        """Trim text to the specified length."""
        if not text:
            return ""
        if len(text) <= limit:
            return text
        return text[:limit].rsplit(" ", 1)[0].rstrip("\n") + "..."

    def _build_search_payload(self, query: str, overrides: FirecrawlSearchInput) -> dict[str, Any]:
        """Construct the payload for the Firecrawl search API."""

        def resolve(field_name: str) -> Any:
            if hasattr(overrides, field_name):
                value = getattr(overrides, field_name)
                if value is not None:
                    return value
            return deepcopy(getattr(self, field_name, None))

        limit = resolve("limit") or 5
        sources = resolve("sources") or [SourceWeb()]

        payload = {
            "query": query,
            "limit": limit,
            "sources": [
                source.model_dump(exclude_none=True, by_alias=True) if isinstance(source, BaseModel) else source
                for source in sources
            ],
            "categories": resolve("categories"),
            "location": resolve("location"),
            "tbs": resolve("tbs"),
            "country": resolve("country"),
            "timeout": resolve("timeout"),
            "ignoreInvalidURLs": resolve("ignore_invalid_urls"),
        }

        return {k: v for k, v in payload.items() if v is not None}

    def _format_scraped_results(self, query: str, results: list[dict[str, Any]]) -> str:
        if not results:
            return f'## Firecrawl Search Results for "{query}"\nNo results returned.'

        sections = [f'## Firecrawl Search Results for "{query}"']
        for index, item in enumerate(results, start=1):
            title = item.get("title") or "Untitled result"
            url = item.get("url")
            description = self._truncate(item.get("description"), self.MAX_DESCRIPTION_CHARS)
            sections.append(f"### Result {index}: {title}")
            if url:
                sections.append(f"- URL: [{url}]({url})")
            if description:
                sections.append(f"- Description: {description}")

            for field, label in (("markdown", "Markdown"), ("summary", "Summary"), ("html", "HTML")):
                content = self._truncate(item.get(field), self.MAX_CONTENT_CHARS)
                if content:
                    sections.append(f"- {label}: {content}")

            if links := item.get("links"):
                link_lines = "\n".join(f"  * {link}" for link in links)
                sections.append(f"- Links:\n{link_lines}")

            sections.append("")

        return "\n".join(sections).strip()

    def _format_result_list(self, label: str, results: list[dict[str, Any]]) -> list[str]:
        if not results:
            return []

        section = [f"### {label}"]
        for index, item in enumerate(results, start=1):
            title = item.get("title") or "Untitled result"
            description = self._truncate(item.get("description") or item.get("snippet"), self.MAX_DESCRIPTION_CHARS)
            url = item.get("url") or item.get("imageUrl")

            if url:
                section.append(f"- {index}. [{title}]({url})")
            else:
                section.append(f"- {index}. {title}")
            if description:
                section.append(f"  - {description}")
        return section

    def _format_indexed_results(self, query: str, data: dict[str, Any]) -> str:
        sections = [f'## Firecrawl Search Results for "{query}"']
        sections.extend(self._format_result_list("Web Results", data.get("web", [])))
        sections.extend(self._format_result_list("News Results", data.get("news", [])))
        sections.extend(self._format_result_list("Image Results", data.get("images", [])))

        if len(sections) == 1:
            sections.append("No results returned.")

        return "\n".join(sections)

    def _format_agent_response(self, query: str, data: Any) -> str:
        """Format the response for agent consumption using Markdown."""
        if isinstance(data, list):
            return self._format_scraped_results(query, data)
        if isinstance(data, dict):
            return self._format_indexed_results(query, data)
        return f'## Firecrawl Search Results for "{query}"\nNo results returned.'

    def execute(
        self, input_data: FirecrawlSearchInput, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute the search tool with the provided input data."""
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query or self.query
        if not query:
            logger.error(f"Tool {self.name} - {self.id}: failed to get input data.")
            raise ValueError("Query is required for search")

        search_payload = self._build_search_payload(query, input_data)
        connection_url = urljoin(self.connection.url, "search")

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                json=search_payload,
                headers=self.connection.headers,
            )
            response.raise_for_status()
            search_result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to execute the requested action. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        data = search_result.get("data")
        if self.is_optimized_for_agents:
            result = self._format_agent_response(query, data)
        else:
            result = {"success": search_result.get("success", False), "query": query, "data": data}

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
