from enum import Enum
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Exa
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class QueryType(str, Enum):
    keyword = "keyword"
    neural = "neural"
    auto = "auto"


class ExaInputSchema(BaseModel):
    """Schema for Exa search input parameters."""

    query: str = Field(description="The search query string.")
    include_full_content: bool = Field(
        default=False, description="If true, retrieve full content, highlights, and summaries for search results."
    )
    use_autoprompt: bool = Field(
        default=False, description="If true, query will be converted to a Exa query. Default false."
    )
    query_type: QueryType = Field(
        default=QueryType.auto,
        description="Type of query to be used. Options are 'keyword', 'neural', or 'auto'. Default is 'auto'.",
    )
    category: str | None = Field(
        default=None, description="A data category to focus on (e.g., company, research paper, news article)."
    )
    limit: int = Field(default=10, ge=1, le=100, description="Number of search results to return. Default 10.")
    include_domains: list[str] | None = Field(default=None, description="List of domains to include in the search.")
    exclude_domains: list[str] | None = Field(default=None, description="List of domains to exclude from the search.")
    include_text: list[str] | None = Field(default=None, description="Strings that must be present in webpage text.")
    exclude_text: list[str] | None = Field(
        default=None, description="Strings that must not be present in webpage text."
    )


class ExaTool(ConnectionNode):
    """
    A tool for performing web searches using the Exa AI API.

    This tool accepts various search parameters and returns relevant search results
    with options for filtering by date, domain, and content.

    Attributes:
        group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
        name (str): The name of the tool.
        description (str): A brief description of the tool.
        connection (Exa): The connection instance for the Exa API.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Exa Search Tool"
    description: str = (
        "A tool for searching the web using Exa AI. "
        "Provides advanced search capabilities with options for filtering results."
    )
    connection: Exa

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[ExaInputSchema]] = ExaInputSchema

    @staticmethod
    def to_camel_case(snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
        """
        Formats the search results into a human-readable string.

        Args:
            results (list[dict[str, Any]]): The raw search results.

        Returns:
            str: A formatted string containing the search results.
        """
        formatted_results = []
        for result in results:
            formatted_results.extend(
                [
                    f"Title: {result.get('title', 'N/A')}",
                    f"URL: {result.get('url', 'N/A')}",
                    f"Published Date: {result.get('publishedDate', 'N/A')}",
                    f"Author: {result.get('author', 'N/A')}",
                    f"Score: {result.get('score', 'N/A')}",
                ]
            )

            if "highlights" in result and result["highlights"]:
                formatted_results.extend(
                    [
                        "Highlights:",
                        *[f"  • {highlight}" for highlight in result["highlights"]],
                    ]
                )

            if "summary" in result and result["summary"]:
                formatted_results.extend(["Summary:", f"  {result['summary']}"])

            formatted_results.append("")

        return "\n".join(formatted_results).strip()

    def execute(self, input_data: ExaInputSchema, config: RunnableConfig | None = None, **kwargs) -> dict[str, Any]:
        """
        Executes the search using the Exa API and returns the formatted results.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        payload = {k: v for k, v in input_data.model_dump().items() if v is not None}
        include_full_content = payload.pop("include_full_content", False)

        payload = {
            "query": payload.pop("query"),
            "useAutoprompt": payload.pop("use_autoprompt", False),
            "type": payload.pop("query_type", "neural"),
            "numResults": payload.pop("limit", 10),
            **{self.to_camel_case(k): v for k, v in payload.items()},
        }

        if include_full_content:
            payload["contents"] = {
                "text": {"maxCharacters": 1000, "includeHtmlTags": False},
                "highlights": {"numSentences": 3, "highlightsPerUrl": 2, "query": payload["query"]},
                "summary": {"query": f"Summarize the main points about {payload['query']}"},
            }

        connection_url = urljoin(self.connection.url, "search")

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                json=payload,
                headers=self.connection.headers,
            )
            response.raise_for_status()
            search_result = response.json()
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {str(e)}")
            raise

        results = search_result.get("results", [])
        formatted_results = self._format_search_results(results)

        sources_with_url = [f"{result.get('title')}: ({result.get('url')})" for result in results]

        if self.is_optimized_for_agents:
            result_parts = ["<Sources with URLs>", "\n".join(sources_with_url), "</Sources with URLs>"]

            result_parts.extend(["<Search results>", formatted_results, "</Search results>"])

            result = "\n\n".join(result_parts)
        else:
            urls = [result.get("url") for result in results]
            result = {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "urls": urls,
                "raw_response": search_result,
            }

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
