from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Tavily
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_TAVILY = """Searches the web using Tavily with natural language queries and adjustable search depth.

Key Capabilities:
- Basic/advanced search modes for speed vs thoroughness balance
- Natural language processing for complex questions
- Topic-specific searches (general, news) with domain filtering
- AI-generated summaries and full content extraction

Usage Strategy:
- Basic: Quick factual queries, current events
- Advanced: Complex research topics, comprehensive analysis
- Use include_answer for summaries, time_range for recent results

Parameter Guide:
- query: Search query to find relevant information
- search_depth: basic (fast) vs advanced (thorough)
- topic: general vs news for content type
- include_answer: AI summary alongside sources
- include_raw_content: Full page text for analysis
- time_range: Filter results by recency (day, week, month, year)
- max_results: Limit results (default: 5, range: 1-20)
- include_images: Include images in results
- include_domains: Specific domains to include
- exclude_domains: Domains to exclude from results
- use_cache: Use cached results when available
- chunks_per_source: Number of chunks to return per source (default: 3, range: 1-3)


Examples:
- {"query": "React performance 2024", "search_depth": "advanced"}
- {"query": "GPT-4 news", "topic": "news", "time_range": "week"}
- {"query": "ML tutorials", "include_domains": ["coursera.org"]}"""


class TavilyInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a search query.")
    search_depth: str | None = Field(
        default=None, description="The search depth to use; must be either `basic` or `advanced`."
    )
    topic: str | None = Field(
        default=None,
        description="The topic to search for; must be either `general` or `news`.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    chunks_per_source: int | None = Field(
        default=3,
        ge=1,
        le=3,
        description="The number of chunks to return per source (default: 3, range: 1-3).",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    time_range: str | None = Field(
        default=None,
        description="The time range back from the current date to filter results. "
        "Useful when looking for sources that have published data. "
        "Available options are only one of: `day`, `week`, `month`, `year`, `d`, `w`, `m`, `y`.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    max_results: int | None = Field(
        default=None, description="The maximum number of search results to return (default: 5, range: 1-20)."
    )
    include_images: bool | None = Field(
        default=None,
        description="Include images in search results.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    include_answer: bool | None = Field(
        default=None,
        description="Include a summarized answer in search results.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    include_raw_content: bool | None = Field(default=None, description="Include full page content in search results.")
    include_domains: list[str] | None = Field(
        default=None,
        description="Specific domains to include in search results.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="Domains to exclude from search results.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    use_cache: bool | None = Field(
        default=None,
        description="Use cached results when available.",
        json_schema_extra={"is_accessible_to_agent": False},
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

    search_depth: str = Field(default="basic", description="The search depth to use.")
    topic: str = Field(default="general", description="The topic to search for.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="The maximum number of search results to return.",
    )
    chunks_per_source: int | None = Field(
        default=3,
        ge=1,
        le=3,
        description="The number of chunks to return per source (default: 3, range: 1-3).",
    )
    time_range: str | None = Field(
        default=None,
        description="The time range back from the current date to filter results. "
        "Useful when looking for sources that have published data. "
        "Available options are only one of: `day`, `week`, `month`, `year`, `d`, `w`, `m`, `y`.",
    )
    include_images: bool = Field(default=False, description="Include images in search results.")
    include_answer: bool = Field(default=False, description="Include answer in search results.")
    include_raw_content: bool = Field(default=False, description="Include raw content in search results.")
    include_domains: list[str] = Field(default_factory=list, description="The domains to include in search results.")
    exclude_domains: list[str] = Field(default_factory=list, description="The domains to exclude from search results.")
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
            "search_depth": self.search_depth,
            "topic": self.topic,
            "max_results": self.max_results,
            "include_images": self.include_images,
            "include_answer": self.include_answer,
            "include_raw_content": self.include_raw_content,
            "include_domains": self.include_domains,
            "exclude_domains": self.exclude_domains,
            "use_cache": self.use_cache,
            "chunks_per_source": self.chunks_per_source,
            "time_range": self.time_range,
        }

        input_dict = input_data.model_dump(exclude_unset=True)
        for key, value in input_dict.items():
            if value is not None:
                search_data[key] = value

        connection_url = urljoin(self.connection.url, "/search")

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                json={**self.connection.data, **search_data},
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
        sources_with_url = [
            f"[{result.get('title')}]({result.get('url')})"
            for result in search_result.get("results", [])
        ]

        if self.is_optimized_for_agents:
            result = (
                "## Sources with URLs\n"
                + "\n".join([f"- {source}" for source in sources_with_url])
                + "\n\n## Search results for: "
                + f"'{input_data.query}'\n\n"
                + formatted_results
            )
            if search_result.get("answer", ""):
                result += f"\n\n## Summary Answer\n\n{search_result.get('answer')}"
        else:
            result = {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "raw_response": search_result,
                "images": search_result.get("images", []),
                "answer": search_result.get("answer", ""),
                "query": search_result.get("query", ""),
                "response_time": search_result.get("response_time", 0),
            }

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
