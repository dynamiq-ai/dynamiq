import enum
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.connections import ScaleSerp
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_SERP = """## Scale SERP Search Tool
### Overview
A web search tool that retrieves current information from the internet using Scale SERP API.
Access search results including web pages, news, images, and videos with organized titles, links, and descriptive snippets.
### When to Use
- Find current information beyond your knowledge base.
- Verify facts or recent developments.
- Research specific topics, people, companies, or events.
- Locate authoritative sources.
- Discover relevant visual content.
### Parameters
- `query`: Search term (e.g., "latest climate change report")
- `url`: Specific website to search (e.g., "https://www.example.com")
- `limit`: Number of results (default: 10, max: 1000)
- `search_type`: Search category:
  - `web`: Standard results (default)
  - `news`: Recent articles
  - `images`: Image results
  - `videos`: Video results
### Examples
1. Basic web search:
   {
     "query": "renewable energy breakthroughs 2024",
     "limit": 5
   }
2. News search:
   {
     "query": "latest economic policy updates",
     "search_type": "news",
     "limit": 3
   }
### Notes
- Use specific, concise search terms for best results.
- News results reflect most recent publications.
"""  # noqa E501


class SearchType(str, enum.Enum):
    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"

    @property
    def result_key(self) -> str:
        """Returns the corresponding result key for the search type"""
        return {
            SearchType.WEB: "organic_results",
            SearchType.NEWS: "news_results",
            SearchType.IMAGES: "image_results",
            SearchType.VIDEOS: "video_results",
        }[self]


class ScaleSerpInputSchema(BaseModel):
    query: str | None = Field(default=None, description="Parameter to provide a search query.")
    url: str | None = Field(default=None, description="Parameter to provide a search url.")
    limit: int | None = Field(default=None, description="Parameter to specify the number of results to return.")
    search_type: SearchType = Field(
        default=SearchType.WEB, description="Type of search to perform (web, news, images, videos)"
    )

    @model_validator(mode="after")
    def validate_query_url(self):
        """Validate that either query or url is specified if both are provided"""
        if self.url and self.query:
            raise ValueError("Cannot specify both 'query' and 'url' at the same time.")
        return self


class ScaleSerpTool(ConnectionNode):
    """
    A tool for performing web searches using the Scale SERP API.

    This tool accepts a query or URL and returns search results based on the specified
    search type (organic, news, images, videos). The results include titles, links, and snippets.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        connection (ScaleSerp): The connection instance for the Scale SERP API.
        query (str): The default search query to use.
        url (str): The default URL to search.
        limit (int): The default number of search results to return.
        search_type (SearchType): The type of search to perform.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Scale Serp Search Tool"
    description: str = DESCRIPTION_SERP
    connection: ScaleSerp

    query: str = Field(default="", description="The default search query to use")
    url: str = Field(default="", description="The default URL to search")
    limit: int = Field(default=10, ge=1, le=1000, description="The default number of search results to return")
    search_type: SearchType = Field(default=SearchType.WEB, description="The type of search to perform")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[ScaleSerpInputSchema]] = ScaleSerpInputSchema

    def _format_search_results(self, results: dict[str, Any]) -> str:
        """
        Formats the search results into a human-readable string.
        """
        content_results = results.get(self.search_type.result_key, [])

        formatted_results = []
        for result in content_results:
            formatted_results.extend(
                [
                    f"Title: {result.get('title')}",
                    f"Link: {result.get('link')}",
                    f"Snippet: {result.get('snippet', 'N/A')}",
                    "",
                ]
            )

        return "\n".join(formatted_results).strip()

    def get_params(
        self, query: str | None = None, url: str | None = None, search_type: SearchType | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Prepare the parameters for the API request.
        """
        params = {"api_key": self.connection.api_key, **kwargs}

        current_search_type = search_type or self.search_type

        if current_search_type != SearchType.WEB:
            params["search_type"] = current_search_type

        if query:
            params["q"] = query
        elif url:
            params["url"] = url

        return {k: v for k, v in params.items() if v is not None}

    def execute(
        self, input_data: ScaleSerpInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the search using the Scale SERP API and returns the formatted results.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query or self.query
        url = input_data.url or self.url
        limit = input_data.limit or self.limit
        search_type = input_data.search_type or self.search_type

        if not query and not url:
            raise ToolExecutionException(
                "Either 'query' or 'url' must be provided in input data or node parameters.", recoverable=True
            )

        try:
            response = self.client.request(
                method=self.connection.method,
                url=urljoin(self.connection.url, "/search"),
                params=self.get_params(query=query, url=url, num=limit, search_type=search_type),
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
        content_results = search_result.get(search_type.result_key, [])

        sources_with_url = [
            f"{result.get('title')}: ({result.get('link')})"
            for result in content_results
        ]

        if self.is_optimized_for_agents:
            return {
                "content": (
                    "<Sources with URLs>\n"
                    + "\n".join(sources_with_url)
                    + "</Sources with URLs>\n\n<Search results>"
                    + formatted_results
                    + "</Search results>"
                )
            }

        return {
            "content": {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "urls": [result.get("link") for result in content_results],
                "raw_response": search_result,
            }
        }
