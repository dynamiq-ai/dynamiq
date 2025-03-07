from enum import Enum
from typing import Any, ClassVar, Literal
from urllib.parse import urljoin

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Exa
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_EXA = """## Exa Search Tool
### Overview
Exa Search Tool provides web search capabilities powered by Exa AI's semantic search technology.
Search the internet for current information with powerful filtering options and retrieve full webpage content when needed.
### Capabilities
- Perform keyword and semantic searches across the web
- Filter results by domains, text content, and categories
- Retrieve highlights, summaries, and full content from webpages
- Access metadata including titles, URLs, dates, and authors
### Input Parameters
- **query** (string, required): Search query text
- **include_full_content** (boolean, optional, default: false): Retrieves complete content when true
- **use_autoprompt** (boolean, optional, default: false): Enhances query automatically when true. Enabled by default for auto search, optional for neural search, and not available for keyword search.
- **query_type** (string, optional, default: "auto"): "keyword" (exact match), "neural" (semantic), or "auto"
- **category** (string, optional): Focus on specific data types (only company, research paper, news, pdf, github, tweet, personal site, linkedin profile, financial report)
- **limit** (integer, optional, default: 10): Number of results to return (1-100)
### Examples of action input for tool usage
#### Basic Search
{
  "query": "renewable energy advancements 2024"
}
#### Filtered Domain Search
{
  "query": "machine learning applications",
  "category": "research paper",
  "limit": 15
}
#### Comprehensive Research
{
  "query": "climate change policy",
  "query_type": "neural",
  "include_full_content": true,
  "category": "research paper"
}
### Best Practices
1. Use specific queries with key terms and context
2. Combine domain and text filters for improved relevance
3. Use "neural" for concept searches, "keyword" for exact matches
4. Request fewer results (5-15) for higher quality information
5. Use include_full_content sparingly to maintain response speed
"""  # noqa E501


class QueryType(str, Enum):
    keyword = "keyword"
    neural = "neural"
    auto = "auto"


class ExaInputSchema(BaseModel):
    """Schema for Exa search input parameters."""

    query: str = Field(description="The search query string.")
    include_full_content: bool | None = Field(
        default=None,
        description="If true, retrieve full content, highlights, and summaries for search results.",
        is_accessible_to_agent=True,
    )
    use_autoprompt: bool | None = Field(
        default=None,
        description="If true, query will be converted to a Exa query."
        "Enabled by default for auto search, optional for neural search, and not available for keyword search.",
        is_accessible_to_agent=False,
    )
    query_type: QueryType | None = Field(
        default=None,
        description="Type of query to be used. Options are 'keyword', 'neural', or 'auto'."
        "Neural uses an embeddings-based model, keyword is google-like SERP. "
        "Default is auto, which automatically decides between keyword and neural.",
        is_accessible_to_agent=False,
    )
    category: str | None = Field(
        default=None,
        description="A data category to focus on."
        "Options are company, research paper, news, pdf,"
        " github, tweet, personal site, linkedin profile, financial report.",
        is_accessible_to_agent=True,
    )
    limit: int | None = Field(
        default=None, ge=1, le=100, description="Number of search results to return.", is_accessible_to_agent=False
    )
    include_domains: list[str] | None = Field(
        default=None, description="List of domains to include in the search.", is_accessible_to_agent=False
    )
    exclude_domains: list[str] | None = Field(
        default=None, description="List of domains to exclude from the search.", is_accessible_to_agent=False
    )
    include_text: list[str] | None = Field(
        default=None, description="Strings that must be present in webpage text.", is_accessible_to_agent=False
    )
    exclude_text: list[str] | None = Field(
        default=None, description="Strings that must not be present in webpage text.", is_accessible_to_agent=False
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
        include_full_content (bool): If true, retrieve full content, highlights, and summaries.
        use_autoprompt (bool): If true, query will be converted to a Exa query.
        query_type (QueryType): Type of query to be used.
        category (str, optional): A data category to focus on.
        limit (int): Number of search results to return.
        include_domains (list[str], optional): List of domains to include.
        exclude_domains (list[str], optional): List of domains to exclude.
        include_text (list[str], optional): Strings that must be present.
        exclude_text (list[str], optional): Strings that must not be present.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Exa Search Tool"
    description: str = DESCRIPTION_EXA
    connection: Exa

    include_full_content: bool = Field(
        default=False, description="If true, retrieve full content, highlights, and summaries for search results."
    )
    use_autoprompt: bool = Field(default=False, description="If true, query will be converted to a Exa query.")
    query_type: QueryType = Field(default=QueryType.auto, description="Type of query to be used.")
    category: str | None = Field(default=None, description="A data category to focus on.")
    limit: int = Field(default=10, ge=1, le=100, description="Number of search results to return.")
    include_domains: list[str] | None = Field(default=None, description="List of domains to include in the search.")
    exclude_domains: list[str] | None = Field(default=None, description="List of domains to exclude from the search.")
    include_text: list[str] | None = Field(default=None, description="Strings that must be present in webpage text.")
    exclude_text: list[str] | None = Field(
        default=None, description="Strings that must not be present in webpage text."
    )

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

        Input parameters override node parameters when provided.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        payload = {
            "query": input_data.query,
            "useAutoprompt": (
                input_data.use_autoprompt if input_data.use_autoprompt is not None else self.use_autoprompt
            ),
            "type": input_data.query_type if input_data.query_type is not None else self.query_type,
            "numResults": input_data.limit if input_data.limit is not None else self.limit,
            "includeDomains": (
                input_data.include_domains if input_data.include_domains is not None else self.include_domains
            ),
            "excludeDomains": (
                input_data.exclude_domains if input_data.exclude_domains is not None else self.exclude_domains
            ),
            "includeText": input_data.include_text if input_data.include_text is not None else self.include_text,
            "excludeText": input_data.exclude_text if input_data.exclude_text is not None else self.exclude_text,
            "category": input_data.category if input_data.category is not None else self.category,
        }

        if isinstance(payload["type"], QueryType):
            payload["type"] = payload["type"].value

        payload = {k: v for k, v in payload.items() if v is not None}

        include_full_content = (
            input_data.include_full_content
            if input_data.include_full_content is not None
            else self.include_full_content
        )

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
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to retrieve search results. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

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
