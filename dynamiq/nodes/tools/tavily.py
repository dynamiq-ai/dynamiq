import json
from datetime import date
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Tavily
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_TAVILY = """Search the web with Tavily's latest API,
including automatic parameter tuning, structured filters, and enriched responses.

Key Capabilities:
- Auto-parameterization toggles (let Tavily infer best depth/topic) or manual control
- Basic vs advanced search depth with chunked content snippets
- Topic-specific queries (general/news/finance) plus country/time filters
- Optional LLM answers, raw HTML/text content, favicons, and rich image metadata

Usage Strategy:
- Enable auto_parameters for broad queries, override specific knobs when precision matters
- Prefer advanced depth with chunks_per_source for research-grade answers
- Combine time_range/start_date/end_date with include/exclude_domains for curated monitoring
- Turn on include_answer/include_raw_content/include_images when agents must summarize or embed

Parameter Guide:
- query: Required natural language search query
- auto_parameters: Let Tavily choose topic/search_depth for you (beta, extra credit cost)
- search_depth: `basic` (fast) vs `advanced` (thorough + chunk snippets)
- topic: `general`, `news`, or `finance`
- max_results: Limit (default 5, up to 20)
- chunks_per_source: 1-3 chunks per source (advanced-only)
- time_range / start_date / end_date: Recency controls using relative or ISO dates
- include_answer: `true`/`false` or `basic`/`advanced` summaries
- include_raw_content: `true`/`false` or `markdown`/`text` full content
- include_images & include_image_descriptions: Return related images
- include_favicon: Adds site favicons to each result
- include_domains / exclude_domains / country: Focus or filter sources
- use_cache: Reuse existing Tavily results when possible

Examples:
- {"query": "React performance optimization", "search_depth": "advanced", "chunks_per_source": 2}
- {"query": "latest AI developments", "topic": "news",
"time_range": "week", "include_answer": "basic"}
- {"query": "S&P 500 outlook", "topic": "finance", "auto_parameters": true}
- {"query": "machine learning tutorials",
"include_domains": ["coursera.org", "kaggle.com"], "include_raw_content": "markdown"}
- {"query": "Python best practices", "include_answer": "advanced",
"include_images": true, "include_image_descriptions": true}"""


class TavilyInputSchema(BaseModel):
    query: str = Field(..., description="Natural language Tavily search query.")
    auto_parameters: bool | None = Field(
        default=None,
        description="Let Tavily automatically select topic/search_depth for the query (beta, costs extra credits).",
    )
    search_depth: Literal["basic", "advanced"] | None = Field(
        default=None,
        description="Search depth; `basic` for speed " "or `advanced` for deeper crawling and chunk extraction.",
    )
    topic: Literal["general", "news", "finance"] | None = Field(
        default=None,
        description="Content focus for the query. "
        "`general` for broad sources, `news` for current events, `finance` for markets.",
    )
    chunks_per_source: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description="Max number of 500-character chunks returned per source (advanced-depth only).",
    )
    time_range: Literal["day", "week", "month", "year", "d", "w", "m", "y"] | None = Field(
        default=None,
        description="Relative time window for results based "
        "on publish/update date (day/week/month/year or shorthand d/w/m/y).",
    )
    start_date: date | None = Field(default=None, description="Return results on or after this ISO date (YYYY-MM-DD).")
    end_date: date | None = Field(default=None, description="Return results on or before this ISO date (YYYY-MM-DD).")
    max_results: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Maximum number of results to return (default 5, up to 20).",
    )
    include_images: bool | None = Field(
        default=None,
        description="Whether to include relevant images in the response.",
    )
    include_image_descriptions: bool | None = Field(
        default=None,
        description="When images are requested, also return short descriptions for each image.",
    )
    include_favicon: bool | None = Field(
        default=None,
        description="Include the favicon URL for each result.",
    )
    include_answer: bool | Literal["basic", "advanced"] | None = Field(
        default=None,
        description="Return an LLM-generated summary. `basic` is concise, `advanced` is more detailed.",
    )
    include_raw_content: bool | Literal["markdown", "text"] | None = Field(
        default=None,
        description="Return cleaned page content. `markdown` keeps rich formatting, `text` forces plaintext.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="List of domains to preferentially include (max 300).",
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="List of domains to exclude from the search (max 150).",
    )
    country: str | None = Field(
        default=None,
        description="Boost results from a specific country (only when topic is `general`).",
    )
    use_cache: bool | None = Field(
        default=None,
        description="Use cached Tavily results when available.",
    )


class TavilyTool(ConnectionNode):
    """
    TavilyTool is a ConnectionNode that interfaces with the Tavily search service.

    All parameters can be set during initialization and optionally overridden during execution.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Tavily Search Tool"
    description: str = DESCRIPTION_TAVILY
    connection: Tavily

    auto_parameters: bool = Field(
        default=False,
        description="Automatically let Tavily tune parameters (costs 2 credits per request).",
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="`basic` for faster results, `advanced` for higher recall and chunk snippets.",
    )
    topic: Literal["general", "news", "finance"] = Field(
        default="general",
        description="Content focus for the search query.",
    )
    max_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results to return.",
    )
    chunks_per_source: int | None = Field(
        default=3,
        ge=1,
        le=3,
        description="The number of chunks to return per source (default: 3, range: 1-3).",
    )
    time_range: Literal["day", "week", "month", "year", "d", "w", "m", "y"] | None = Field(
        default=None,
        description="The time range back from the current date to filter results. "
        "Useful when looking for sources that have published data. "
        "Available options are: `day`, `week`, `month`, `year`, `d`, `w`, `m`, `y`.",
    )
    start_date: date | None = Field(default=None, description="Only return results published on/after this date.")
    end_date: date | None = Field(default=None, description="Only return results published on/before this date.")
    include_images: bool = Field(default=False, description="Include images in search results.")
    include_image_descriptions: bool = Field(
        default=False,
        description="Include short descriptions for each returned image (requires include_images).",
    )
    include_favicon: bool = Field(default=False, description="Include favicon URLs for each search result.")
    include_answer: bool | Literal["basic", "advanced"] = Field(
        default=False,
        description="Include an LLM-generated summary (`basic` or `advanced`).",
    )
    include_raw_content: bool | Literal["markdown", "text"] = Field(
        default=False,
        description="Include cleaned page content (`markdown` or `text`).",
    )
    include_domains: list[str] = Field(default_factory=list, description="The domains to include in search results.")
    exclude_domains: list[str] = Field(default_factory=list, description="The domains to exclude from search results.")
    country: str | None = Field(
        default=None,
        description="Boost results originating from the provided country (topic must be `general`).",
    )
    use_cache: bool = Field(
        default=True,
        description="Use cache for search results.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[TavilyInputSchema]] = TavilyInputSchema

    def _format_search_results(self, results: dict[str, Any]) -> str:
        """
        Formats the search results into a readable string format.

        Args:
            results (dict[str, Any]): The raw search results from Tavily.

        Returns:
            str: The formatted search results as a string.
        """
        formatted_results = []
        for result in results.get("results", []):
            formatted_results.append(f"Source: {result.get('url')}")
            formatted_results.append(f"Title: {result.get('title')}")
            formatted_results.append(f"Content: {result.get('content')}")
            if result.get("raw_content"):
                formatted_results.append(f"Full Content: {result.get('raw_content')}")
            if result.get("favicon"):
                formatted_results.append(f"Favicon: {result.get('favicon')}")
            formatted_results.append(f"Relevance Score: {result.get('score')}")
            formatted_results.append("")  # Blank line between results

        return "\n".join(formatted_results).strip()

    def execute(self, input_data: TavilyInputSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """
        Executes the search operation using the provided input data.
        Parameters from input_data override the node's default parameters if provided.

        Args:
            input_data (TavilyInputSchema): The input data containing the search query and optional parameters.
            config (RunnableConfig | None): Optional configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: The result of the search operation.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        search_data = {
            "query": input_data.query,
            "auto_parameters": self.auto_parameters,
            "search_depth": self.search_depth,
            "topic": self.topic,
            "max_results": self.max_results,
            "include_images": self.include_images,
            "include_image_descriptions": self.include_image_descriptions,
            "include_favicon": self.include_favicon,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
            "include_domains": self.include_domains,
            "exclude_domains": self.exclude_domains,
            "country": self.country,
            "use_cache": self.use_cache,
            "chunks_per_source": self.chunks_per_source,
            "time_range": self.time_range,
            "start_date": self.start_date,
            "end_date": self.end_date,
        }

        input_dict = input_data.model_dump(exclude_unset=True)
        for key, value in input_dict.items():
            if value is not None:
                search_data[key] = value

        date_fields = ("start_date", "end_date")
        for date_field in date_fields:
            if isinstance(search_data.get(date_field), date):
                search_data[date_field] = search_data[date_field].isoformat()

        if search_data.get("search_depth") != "advanced":
            search_data.pop("chunks_per_source", None)

        search_data = {key: value for key, value in search_data.items() if value is not None}
        request_payload = dict(search_data)

        connection_url = urljoin(self.connection.url, "/search")

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                json={**self.connection.data, **request_payload},
            )
            response.raise_for_status()
            search_result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve search results. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        formatted_results = self._format_search_results(search_result)
        sources_with_url = []
        for result in search_result.get("results", []):
            title, url = result.get("title"), result.get("url")
            if title and url:
                entry = f"[{title}]({url})"
            else:
                entry = url or title or ""
            if result.get("favicon"):
                entry += f" (favicon: {result.get('favicon')})"
            if entry:
                sources_with_url.append(entry)

        images_info = []
        for image in search_result.get("images", []):
            url = image.get("url")
            if not url:
                continue
            description = image.get("description") or "Image"
            images_info.append(f"- {description}: {url}")

        metadata_lines: list[str] = []
        if search_result.get("response_time") is not None:
            metadata_lines.append(f"- Response Time: {search_result.get('response_time')}s")
        if search_result.get("request_id"):
            metadata_lines.append(f"- Request ID: {search_result.get('request_id')}")
        auto_parameters = search_result.get("auto_parameters")
        if auto_parameters:
            auto_params_str = ", ".join(f"{key}={value}" for key, value in auto_parameters.items())
            metadata_lines.append(f"- Auto Parameters: {auto_params_str}")

        if self.is_optimized_for_agents:
            result = (
                "## Sources with URLs\n"
                + "\n".join([f"- {source}" for source in sources_with_url])
                + "\n\n## Search results for: "
                + f"'{search_result.get('query', input_data.query)}'\n\n"
                + formatted_results
            )
            if search_result.get("answer", ""):
                result += f"\n\n## Summary Answer\n\n{search_result.get('answer')}"
            if images_info:
                result += "\n\n## Images\n" + "\n".join(images_info)
            if metadata_lines:
                result += "\n\n## Metadata\n" + "\n".join(metadata_lines)
            if request_payload:
                result += "\n\n## Request Parameters\n```json\n" + json.dumps(request_payload, indent=2) + "\n```"
            result += "\n\n## Raw Response\n```json\n" + json.dumps(search_result, indent=2) + "\n```"
        else:
            result = {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "raw_response": search_result,
                "images": search_result.get("images", []),
                "answer": search_result.get("answer", ""),
                "query": search_result.get("query", ""),
                "response_time": search_result.get("response_time", 0),
                "request_id": search_result.get("request_id"),
                "auto_parameters": auto_parameters or {},
                "search_parameters": request_payload,
            }

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
