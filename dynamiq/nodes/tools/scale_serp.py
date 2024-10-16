from typing import Any, Literal
from urllib.parse import urljoin

from pydantic import ConfigDict, Field

from dynamiq.connections import ScaleSerp
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


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
    name: str = "Scale SERP Search Tool"
    description: str = (
        "A tool for searching the web, powered by Scale SERP. "
        "You can use this tool to search the web for information."
        "Input should be a dictionary with a key 'input' containing the query to search."
        "Also there can be a key 'limit' to specify the number of results to return, default is 10."
    )
    connection: ScaleSerp
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="The default number of search results to return",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

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

        if self.connection.search_type == "news":
            content_results = results.get("news_results", [])
        elif self.connection.search_type == "images":
            content_results = results.get("image_results", [])
        elif self.connection.search_type == "videos":
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

    def execute(
        self, input_data: dict[str, Any], config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """
        Executes the search using the Scale SERP API and returns the formatted results.

        Args:
            input_data (dict[str, Any]): The input data containing the search query or URL.
            config (RunnableConfig | None, optional): Optional configuration for the execution.

        Returns:
            dict[str, Any]: A dictionary containing the search results and metadata.
        """
        logger.debug(
            f"Tool {self.name} - {self.id}: started with input data {input_data}"
        )

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        query: str | None = input_data.get("input") or input_data.get("query")
        url: str | None = input_data.get("url")
        limit: int = input_data.get("limit", self.limit)

        if not query and not url:
            return {
                "content": "Error: Either 'input' (for query) or 'url' must be provided."
            }

        search_params = self.connection.get_params(query=query, url=url, num=limit)

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

        logger.debug(f"Tool {self.name} - {self.id}: search result {search_result}")

        formatted_results = self._format_search_results(search_result)

        content_results = search_result.get("organic_results", [])
        if self.connection.search_type == "news":
            content_results = search_result.get("news_results", [])
        elif self.connection.search_type == "images":
            content_results = search_result.get("image_results", [])
        elif self.connection.search_type == "videos":
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

        logger.debug(f"Tool {self.name} - {self.id}: finished with result {str(result)[:200]}...")

        return {"content": result}
