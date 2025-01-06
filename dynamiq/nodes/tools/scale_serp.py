import enum
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamiq.connections import ScaleSerp
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class SearchType(str, enum.Enum):
    WEB = "web"
    NEWS = "news"
    IMAGES = "images"
    VIDEOS = "videos"


class ScaleSerpInputSchema(BaseModel):
    query: str = Field(default="", description="Parameter to provide a search query.")
    url: str = Field(default="", description="Parameter to provide a search url.")
    limit: str = Field(
        default="", description="Parameter to specify the number of results to return, by default is set to 10."
    )

    @model_validator(mode="after")
    def validate_query_url(self):
        """Validate that either query or url is specified"""
        if not self.url and not self.query:
            raise ValueError("Either 'query' or 'url' has to be specified.")
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
        limit (int): The default number of search results to return.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Scale Serp Search Tool"
    description: str = (
        "A tool for searching the web, powered by Scale SERP. "
        "You can use this tool to search the web for information."
    )
    connection: ScaleSerp
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="The default number of search results to return",
    )
    search_type: SearchType = SearchType.WEB

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[ScaleSerpInputSchema]] = ScaleSerpInputSchema

    def _format_search_results(self, results: dict[str, Any]) -> str:
        """
        Formats the search results into a human-readable string.

        Args:
            results (dict[str, Any]): The raw search results.

        Returns:
            str: A formatted string containing the search results.
        """
        formatted_results = []
        content_results = results.get("organic_results", [])

        if self.search_type == SearchType.NEWS:
            content_results = results.get("news_results", [])
        elif self.search_type == SearchType.IMAGES:
            content_results = results.get("image_results", [])
        elif self.search_type == SearchType.VIDEOS:
            content_results = results.get("video_results", [])

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

    def get_params(self, query: str | None = None, url: str | None = None, **kwargs) -> dict[str, Any]:
        """
        Prepare the parameters for the API request.
        """
        params = {"api_key": self.connection.api_key, "search_type": self.search_type, **kwargs}
        if self.search_type == SearchType.WEB:
            params.pop("search_type")

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

        Args:
            input_data (dict[str, Any]): The input data containing the search query or URL.
            config (RunnableConfig | None, optional): Optional configuration for the execution.

        Returns:
            dict[str, Any]: A dictionary containing the search results and metadata.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query: str | None = input_data.query
        url: str | None = input_data.url
        limit: int = input_data.limit or self.limit

        if not query and not url:
            return {
                "content": "Error: Either 'input' (for query) or 'url' must be provided."
            }

        search_params = self.get_params(query=query, url=url, num=limit)

        connection_url = urljoin(self.connection.url, "/search")

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                params=search_params,
            )
            response.raise_for_status()
            search_result = response.json()
        except Exception as e:
            logger.error(
                f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}"
            )
            raise

        formatted_results = self._format_search_results(search_result)
        content_results = search_result.get("organic_results", [])
        if self.search_type == SearchType.NEWS:
            content_results = search_result.get("news_results", [])
        elif self.search_type == SearchType.IMAGES:
            content_results = search_result.get("image_results", [])
        elif self.search_type == SearchType.VIDEOS:
            content_results = search_result.get("video_results", [])

        sources_with_url = [
            f"{result.get('title')}: ({result.get('link')})"
            for result in content_results
        ]

        if self.is_optimized_for_agents:
            result = (
                "<Sources with URLs>\n"
                + "\n".join(sources_with_url)
                + "</Sources with URLs>\n\n<Search results>"
                + formatted_results
                + "</Search results>"
            )
        else:
            urls = [result.get("link") for result in content_results]

            result = {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "urls": urls,
                "raw_response": search_result,
            }

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
