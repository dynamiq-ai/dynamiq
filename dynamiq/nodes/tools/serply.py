from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Serply
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.types.cancellation import check_cancellation
from dynamiq.utils.logger import logger

DESCRIPTION_SERPLY = """Performs a Google web search and returns organic results with titles, links, and snippets.

Key Capabilities:
- Google search results for general research and current information
- Country and language targeting for localized results
- Customizable result counts (1-10)

Usage Strategy:
Use for general research, fact-checking, and finding documentation or current information.
Set gl and hl when results should be localized to a specific country or language.

Examples:
- Basic search: {"query": "open source agent frameworks"}
- Localized search: {"query": "weather in Berlin", "gl": "de", "hl": "de"}"""


class SerplyInputSchema(BaseModel):
    query: str = Field(default="", description="Parameter to provide a search query.")
    limit: int | None = Field(default=None, description="Parameter to specify the number of results to return.")


class SerplyTool(ConnectionNode):
    """
    A tool for performing web searches using the Serply API.

    This tool accepts a query and returns Google search results. The results include
    titles, links, and snippets.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        connection (Serply): The connection instance for the Serply API.
        query (str): The default search query to use.
        limit (int): The default number of search results to return.
        gl (str): The country code used to localize results.
        hl (str): The language code used to localize results.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.WEB_SEARCH
    name: str = "serply-search"
    description: str = DESCRIPTION_SERPLY
    is_parallel_execution_allowed: bool = True
    connection: Serply

    query: str = Field(default="", description="The default search query to use")
    # Serply serves one page per request and returns at most 10 organic results.
    limit: int = Field(default=10, ge=1, le=10, description="The default number of search results to return")
    gl: str = Field(default="", description="The country code used to localize results, for example 'us'")
    hl: str = Field(default="", description="The language code used to localize results, for example 'en'")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[SerplyInputSchema]] = SerplyInputSchema

    def _build_request_kwargs(self, input_data: SerplyInputSchema) -> tuple[dict[str, Any], str, int]:
        """Return (request_kwargs, query, limit)."""
        query = input_data.query or self.query
        limit = input_data.limit or self.limit

        if not query:
            raise ToolExecutionException(
                "Parameter 'query' must be provided in input data or node parameters.", recoverable=True
            )

        params: dict[str, Any] = {"q": query, "num": limit}
        if self.gl:
            params["gl"] = self.gl
        if self.hl:
            params["hl"] = self.hl

        request_kwargs = {
            "method": self.connection.method,
            "url": f"{self.connection.url}/v1/search",
            "headers": self.connection.headers,
            "params": params,
        }
        return request_kwargs, query, limit

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
        """Formats the search results into a human-readable string."""
        formatted_results = []
        for result in results:
            formatted_results.extend(
                [
                    f"Title: {result.get('title')}",
                    f"Link: {result.get('link')}",
                    f"Snippet: {result.get('description', 'N/A')}",
                    "",
                ]
            )

        return "\n".join(formatted_results).strip()

    def _handle_response(self, response: Any, query: str, limit: int) -> dict[str, Any]:
        search_result = response.json()

        if response.status_code >= 400:
            error = search_result.get("detail") if isinstance(search_result, dict) else None
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {error}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve search results. "
                f"Error: {error}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        # Serply can return a couple more results than requested, so trim to the limit.
        results = search_result.get("results", [])[:limit]
        formatted_results = self._format_search_results(results)
        sources_with_url = [f"[{result.get('title')}]({result.get('link')})" for result in results]

        if self.is_optimized_for_agents:
            return {
                "content": (
                    "## Sources with URLs\n"
                    + "\n".join(sources_with_url)
                    + f"\n\n## Search results for '{query}'\n"
                    + formatted_results
                )
            }

        return {
            "content": {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "urls": [result.get("link") for result in results],
                "raw_response": search_result,
            }
        }

    def _wrap_request_exception(self, exc: Exception) -> ToolExecutionException:
        logger.error(f"Tool {self.name} - {self.id}: unexpected error occurred. Error: {str(exc)}")
        return ToolExecutionException(
            f"Tool '{self.name}' encountered an unexpected error. "
            f"Error: {str(exc)}. Please analyze the error and take appropriate action.",
            recoverable=True,
        )

    def execute(self, input_data: SerplyInputSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """
        Executes the search using the Serply API and returns the formatted results.
        """
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        request_kwargs, query, limit = self._build_request_kwargs(input_data)
        try:
            response = self.client.request(**request_kwargs)
            return self._handle_response(response, query, limit)
        except ToolExecutionException:
            raise
        except Exception as e:
            raise self._wrap_request_exception(e)

    async def execute_async(
        self, input_data: SerplyInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Native async execution path mirroring ``execute``."""
        config = ensure_config(config)
        check_cancellation(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        request_kwargs, query, limit = self._build_request_kwargs(input_data)
        client = await self.get_async_client()
        try:
            response = await client.request(**request_kwargs)
            return self._handle_response(response, query, limit)
        except ToolExecutionException:
            raise
        except Exception as e:
            raise self._wrap_request_exception(e)
