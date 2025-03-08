from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Tavily
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_TAVILY = """## Tavily Search Tool
### Description
A web search tool that delivers relevant results from trusted internet sources, specializing in factual information, current events, and topic-specific knowledge.

### Capabilities
- Perform natural language web searches with adjustable depth and focus.
- Filter by topic categories and specific domains.
- Control result quantity and quality.
- Include optional image content, summarized answers, and raw page data.
- Access current information beyond your knowledge base.

### Parameters
- `query`: Your search query (e.g., "latest quantum computing advances").
- `search_depth`: Must be either `basic` or `advanced` (default: `basic`).
- `topic`: Must be either `general` (default) or `news`.
- `max_results`: Number of results (default: 5, range: 1-20).
- `include_raw_content`: Include full page content (default: false).

### Usage Examples
1. Basic search:
   {
     "query": "effects of climate change on coral reefs"
   }
2. Advanced topic-specific search:
   {
     "query": "breakthrough Alzheimer's treatments",
     "search_depth": "advanced",
     "include_raw_content": true
   }

### Tips
- More specific queries yield more relevant results.
- `search_depth: advanced` improves quality but increases response time.
- `include_raw_content` significantly increases response size.
- Topic-specific searches filter out irrelevant content.
- Relevance scores help identify authoritative sources.
"""  # noqa E501


class TavilyInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a search query.")
    search_depth: str | None = Field(
        default=None, description="The search depth to use; must be either `basic` or `advanced`."
    )
    topic: str | None = Field(
        default=None,
        description="The topic to search for; must be either `general` or `news`.",
        is_accessible_to_agent=False,
    )
    max_results: int | None = Field(
        default=None, description="The maximum number of search results to return (default: 5, range: 1-20)."
    )
    include_images: bool | None = Field(
        default=None, description="Include images in search results.", is_accessible_to_agent=False
    )
    include_answer: bool | None = Field(
        default=None, description="Include a summarized answer in search results.", is_accessible_to_agent=False
    )
    include_raw_content: bool | None = Field(default=None, description="Include full page content in search results.")
    include_domains: list[str] | None = Field(
        default=None, description="Specific domains to include in search results.", is_accessible_to_agent=False
    )
    exclude_domains: list[str] | None = Field(
        default=None, description="Domains to exclude from search results.", is_accessible_to_agent=False
    )
    use_cache: bool | None = Field(
        default=None, description="Use cached results when available.", is_accessible_to_agent=False
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
                "<Sources with URLs>\n"
                + "\n".join(sources_with_url)
                + f"\n<\\Sources with URLs>\n\n<Search results for query {input_data.query}>\n"
                + formatted_results
                + f"\n<\\Search results for query {input_data.query}>"
            )
            if search_result.get("answer", "") != "":
                result += f"\n\n<Answer>\n{search_result.get('answer')}\n<\\Answer>"
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
