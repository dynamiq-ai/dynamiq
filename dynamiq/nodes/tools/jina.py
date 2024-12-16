import enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Jina
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class JinaScrapeInputSchema(BaseModel):
    url: str = Field(default="", description="Parameter to provide a url of the page to scrape.")


class ResponseFormat(str, enum.Enum):
    DEFAULT = "default"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    SCREENSHOT = "screenshot"
    PAGESHOT = "pageshot"


SCRAPE_PATH = "https://r.jina.ai/"
SEARCH_PATH = "https://s.jina.ai/"


class JinaScrapeTool(ConnectionNode):
    """
    A tool for scraping web pages, powered by Jina.

    This class is responsible for scraping the content of a web page using Jina.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Jina Scraper Tool"
    description: str = (
        "A tool for scraping web pages, powered by Jina. " "You can use this tool to scrape the content of a web page."
    )
    response_format: ResponseFormat = ResponseFormat.DEFAULT
    connection: Jina
    url: str | None = None
    timeout: int = 60000
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[JinaScrapeInputSchema]] = JinaScrapeInputSchema

    def execute(self, input_data: JinaScrapeInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Executes the web scraping process.

        Args:
            input_data (dict[str, Any]): A dictionary containing 'input' key with the URL to scrape.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the URL and the scraped content.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        # Ensure the config is set up correctly
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        url = input_data.url or self.url
        if not url:
            logger.error(f"Tool {self.name} - {self.id}: failed to get input data.")
            raise ValueError("URL is required for scraping")

        headers = {
            **self.connection.headers,
            **({"X-Return-Format": self.response_format} if self.response_format != ResponseFormat.DEFAULT else {}),
            "X-Timeout": str(self.timeout),
        }

        connection_url = SCRAPE_PATH + url

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                headers=headers,
            )
            response.raise_for_status()
            scrape_result = response.text
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {e}")

            raise

        if self.is_optimized_for_agents:
            result = (
                f"<Source URL>\n{input_data.url}\n<\\Source URL>"
                f"\n<Scraped result>\n{scrape_result}\n<\\Scraped result>"
            )
        else:
            result = {"url": input_data.url, "content": scrape_result}
        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return {"content": result}


class JinaSeacrhInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a search query.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="The maximum number of search results to return.",
    )


class JinaSearchTool(ConnectionNode):
    """
    A tool for search service, powered by Jina.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Jina Search Tool"
    description: str = "A tool for searching the web, powered by Jina."
    connection: Jina
    model_config = ConfigDict(arbitrary_types_allowed=True)
    include_images: bool = Field(default=False, description="Include images in search results.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="The maximum number of search results to return.",
    )
    input_schema: ClassVar[type[JinaSeacrhInputSchema]] = JinaSeacrhInputSchema

    def _format_search_results(self, results: dict[str, Any]) -> str:
        """
        Formats the search results into a readable string format.

        Args:
            results (dict[str, Any]): The raw search results from Tavily.

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
                    f"Content: {result.get('content')}",
                    "",  # Blank line between results
                ]
            )

        return "\n".join(formatted_results).strip()

    def execute(self, input_data: JinaSeacrhInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Executes the web scraping process.

        Args:
            input_data (dict[str, Any]): A dictionary containing 'input' key with the URL to scrape.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the URL and the scraped content.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        # Ensure the config is set up correctly
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query
        max_results = input_data.max_results or self.max_results

        headers = {
            **self.connection.headers,
            **({"X-Retain-Images": "none"} if self.include_images is False else {}),
            "Accept": "application/json",
        }

        params = {"count": max_results}

        connection_url = SEARCH_PATH + query

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

            raise

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
