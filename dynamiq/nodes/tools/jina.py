import enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Jina
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_SCRAPE = """## Jina Scrape Tool
### Overview
The Jina Scrape Tool extracts content from web pages.
### Capabilities
- Extract web content in various formats
- Return well-formatted content for further processing
### When to Use
- Extract information from specific webpages
- Convert web content to more readable formats
### Input Parameters
- **url** (string, required): Complete URL of the webpage to scrape
"""  # noqa: E501

DESCRIPTION_SEARCH = """## Jina Search Tool
### Overview
The Jina Search Tool enables web searches using Jina AI's search engine,
delivering structured results to quickly find relevant information across the internet.
### Capabilities
- Process natural language and keyword queries
- Return search results with titles, descriptions, and URLs
- Optionally include images from results
- Retrieve full content from specific results when needed
- Limit results to focus on most relevant information
- Present results in a readable, structured format
### When to Use
- Research topics requiring multiple perspectives
- Discover resources related to specific queries
- Access current information that may not be in your knowledge base
### Input Parameters
- **query** (string, required): The search query text in natural language or keywords
- **max_results** (integer, optional, default: 5): Maximum number of results to return (1-100)
### Usage Examples
#### Basic Search
{
  "query": "climate change solutions 2025"
}
#### Search with More Results
{
  "query": "best programming languages for beginners",
  "max_results": 10
}
### Best Practices
1. **Be Specific**: Use clear, specific queries for better results.
2. **Limit Results**: Start with fewer results (3-5) for focused information.
3. **Use Natural Language**: Phrases or questions provide better semantic matching.
4. **Add Context**: Include relevant terms to narrow results (e.g., "Python language" vs. "Python").
"""  # noqa: E501


class JinaScrapeInputSchema(BaseModel):
    url: str | None = Field(None, description="Parameter to provide a url of the page to scrape.")


class JinaResponseFormat(str, enum.Enum):
    DEFAULT = "default"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    SCREENSHOT = "screenshot"
    PAGESHOT = "pageshot"


class JinaScrapeTool(ConnectionNode):
    """
    A tool for scraping web pages, powered by Jina.

    This class is responsible for scraping the content of a web page using Jina.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        SCRAPE_PATH(str): The constant path to perform scrape request using Jina API.
        connection (Jina): The connection instance for the Jina API.
        timeout(int): The timeout of the scraping process.
        url (Optional[str]): The URL to scrape, can be set during initialization.
        input_schema (JinaScrapeInputSchema): The input schema for the tool.
    """

    SCRAPE_PATH: ClassVar[str] = "https://r.jina.ai/"

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Jina Scraper Tool"
    description: str = DESCRIPTION_SCRAPE
    response_format: JinaResponseFormat = JinaResponseFormat.DEFAULT
    connection: Jina
    timeout: int = 60
    url: str | None = Field(None, description="URL to scrape")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[JinaScrapeInputSchema]] = JinaScrapeInputSchema

    def execute(self, input_data: JinaScrapeInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Executes the web scraping process.

        Args:
            input_data (JinaScrapeInputSchema): input data for the tool, which includes the url to scrape.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the URL and the scraped content.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        url = input_data.url or self.url
        if not url:
            raise ToolExecutionException(
                "No URL provided. Please provide a URL either during node initialization or execution.",
                recoverable=True,
            )

        headers = {
            **self.connection.headers,
            **({"X-Return-Format": self.response_format} if self.response_format != JinaResponseFormat.DEFAULT else {}),
            "X-Timeout": str(self.timeout),
        }

        connection_url = self.SCRAPE_PATH + url

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                headers=headers,
            )
            response.raise_for_status()
            scrape_result = response.text
            if self.response_format in [JinaResponseFormat.PAGESHOT, JinaResponseFormat.SCREENSHOT]:
                scrape_result = response.content
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to execute the requested action. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            result = f"<Source URL>\n{url}\n<\\Source URL>" f"\n<Scraped result>\n{scrape_result}\n<\\Scraped result>"
        else:
            result = {"url": url, "content": scrape_result}
        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return {"content": result}


class JinaSearchInputSchema(BaseModel):
    query: str | None = Field(None, description="Parameter to provide a search query.")
    max_results: int | None = Field(
        None,
        description="The maximum number of search results to return.",
    )


class JinaSearchTool(ConnectionNode):
    """
    A tool for performing web searches using the Jina AI API.

    This tool accepts various search parameters and returns relevant search results.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        connection (Jina): The connection instance for the Jina API.
        include_images(bool): Whether include images in the search results.
        query (Optional[str]): The search query, can be set during initialization.
        max_results (int): Maximum number of results to return.
        input_schema (JinaSearchInputSchema): The input schema for the tool.
        include_full_content(bool): Whether include full content of the search results.
    """

    SEARCH_PATH: ClassVar[str] = "https://s.jina.ai/"

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Jina Search Tool"
    description: str = DESCRIPTION_SEARCH
    connection: Jina
    include_images: bool = Field(default=False, description="Include images in search results.")
    include_full_content: bool = False
    query: str | None = Field(None, description="Search query")
    max_results: int = Field(default=5, ge=1, le=100, description="Maximum number of search results to return")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[JinaSearchInputSchema]] = JinaSearchInputSchema

    def _format_search_results(self, results: dict[str, Any]) -> str:
        """
        Formats the search results into a readable string format.

        Args:
            results (dict[str, Any]): The raw search results from Jina.

        Returns:
            str: The formatted search results as a string.
        """
        formatted_results = []
        for result in results.get("data", []):
            formatted_results.extend(
                [
                    f"Source: {result.get('url')}",
                    f"Title: {result.get('title')}",
                    f"Description: {result.get('description')}",
                    *(
                        [f"Content: {result.get('content')}"]
                        if self.include_full_content and result.get("content") != ""
                        else []
                    ),
                    "",
                ]
            )

        return "\n".join(formatted_results).strip()

    def execute(self, input_data: JinaSearchInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Executes the web search process.

        Args:
            input_data (JinaSearchInputSchema): input data for the tool, which includes the query to search.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the search results.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query or self.query
        if not query:
            raise ToolExecutionException(
                "No query provided. Please provide a query either during node initialization or execution.",
                recoverable=True,
            )

        max_results = input_data.max_results or self.max_results

        headers = {
            **self.connection.headers,
            **({"X-Retain-Images": "none"} if self.include_images is False else {}),
            "Accept": "application/json",
        }

        params = {"count": max_results}

        connection_url = self.SEARCH_PATH + query

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                headers=headers,
                params=params,
            )
            response.raise_for_status()
            search_result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve search results. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        formatted_results = self._format_search_results(search_result)
        sources_with_url = [f"[{result.get('title')}]({result.get('url')})" for result in search_result.get("data", [])]

        if self.is_optimized_for_agents:
            result = (
                "<Sources with URLs>\n"
                + "\n".join(sources_with_url)
                + f"\n<\\Sources with URLs>\n\n<Search results for query {query}>\n"
                + formatted_results
                + f"\n<\\Search results for query {query}>"
            )
        else:
            result = {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "raw_response": search_result,
                "images": search_result.get("images", []),
                "query": query,
            }

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
