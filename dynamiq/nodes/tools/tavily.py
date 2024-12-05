from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Tavily
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class TavilyInputSchema(BaseModel):
    query: str = Field(..., description="Parameter to provide a search query.")


class TavilyTool(ConnectionNode):
    """
    TavilyTool is a ConnectionNode that interfaces with the Tavily search service.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The node group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool's functionality.
        connection (Tavily): The connection object for interacting with Tavily.
        search_depth (str): The depth of the search, default is 'basic'.
        topic (str): The topic to search for, default is 'general'.
        max_results (int): Maximum number of search results to return, default is 5.
        include_images (bool): Flag to include images in search results, default is False.
        include_answer (bool): Flag to include an answer in search results, default is False.
        include_raw_content (bool): Flag to include raw content in search results, default is False.
        include_domains (list[str]): Domains to include in search results.
        exclude_domains (list[str]): Domains to exclude from search results.
        use_cache (bool): Flag to use cache for search results, default is True.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Tavily Search Tool"
    description: str = (
        "A tool for searching the web, powered by Tavily. "
    )
    connection: Tavily

    search_depth: str = Field(default="basic", description="The search depth to use.")
    topic: str = Field(default="general", description="The topic to search for.")
    max_results: int = Field(
        default=5,
        ge=1,
        le=100,
        description="The maximum number of search results to return.",
    )
    include_images: bool = Field(
        default=False, description="Include images in search results."
    )
    include_answer: bool = Field(
        default=False, description="Include answer in search results."
    )
    include_raw_content: bool = Field(
        default=False, description="Include raw content in search results."
    )
    include_domains: list[str] = Field(
        default_factory=list, description="The domains to include in search results."
    )
    exclude_domains: list[str] = Field(
        default_factory=list, description="The domains to exclude from search results."
    )
    use_cache: bool = Field(default=True, description="Use cache for search results.")

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

        Args:
            input_data (dict[str, Any]): The input data containing the search query.
            config (RunnableConfig | None): Optional configuration for the execution.
            **kwargs: Additional keyword arguments.

        Returns:
            dict[str, Any]: The result of the search operation.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query = input_data.query
        search_data = {
            "query": query,
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
            logger.error(
                f"Tool {self.name} - {self.id}: failed to get results. Error: {e}"
            )
            raise

        formatted_results = self._format_search_results(search_result)
        sources_with_url = [
            f"[{result.get('title')}]({result.get('url')})"
            for result in search_result.get("results", [])
        ]
        if self.is_optimized_for_agents:
            result = (
                "<Sources with URLs>\n"
                + "\n".join(sources_with_url)
                + f"\n<\\Sources with URLs>\n\n<Search results for query {query}>\n"
                + formatted_results
                + f"\n<\\Search results for query {query}>"
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
