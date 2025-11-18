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

DESCRIPTION_EXA = """Searches the web using Exa with semantic understanding and advanced filtering capabilities.

Key capabilities:
- Neural, keyword, or auto search modes for both conceptual and literal queries
- Domain, category, and crawl/published date filtering for precision control
- Text filters (must/must-not include) plus moderation for safer responses
- Full contents retrieval (text, highlights, summaries, context strings, subpages) via typed options
- Auto-prompt optimization for smarter queries when desired

Usage strategy:
- Neural: abstract topics ("AI safety roadmaps", "climate tech breakthroughs")
- Keyword: direct lookups ("OpenAI GPT-4o blog", "python pandas read_csv")
- Fast iteration: start with `limit=3`, widen if needed; tighten crawl/published windows for recency
- Request `contents` when the agent plans to quote source text or needs structured summaries

Parameter quick-reference:
- query: search phrase (required)
- type: `keyword`, `neural`, `auto`, or `fast`
- limit: 1-100 results (default 10)
- include_domains / exclude_domains: whitelist or blacklist specific sites
- start_crawl_date / end_crawl_date: filter by when Exa discovered the page
- start_published_date / end_published_date: filter by when the page was published
- include_text / exclude_text: require or forbid phrases in the page body
- category: constrain to company, research paper, news, pdf, github, tweet, personal site, linkedin profile, financial report
- context: boolean or object to control combined context string length

Examples:
- {"query": "AI research papers", "type": "neural", "limit": 10}
- {"query": "pandas tutorial", "include_domains": ["medium.com"], "contents": {"text": true}}
- {"query": "hydrogen fuel startups", "start_published_date": "2024-01-01T00:00:00.000Z", "limit": 5}
"""  # noqa: E501

DEFAULT_RESULT_TITLE = "Untitled result"
DEFAULT_RESULT_AUTHOR = "Unknown"
DEFAULT_RESULT_PUBLISHED_DATE = "Unknown"


class QueryType(str, Enum):
    keyword = "keyword"
    neural = "neural"
    auto = "auto"


class CategoryType(str, Enum):
    company = "company"
    research_paper = "research paper"
    news = "news"
    pdf = "pdf"
    github = "github"
    tweet = "tweet"
    personal_site = "personal site"
    linkedin_profile = "linkedin profile"
    financial_report = "financial report"


class ContentsTextOptions(BaseModel):
    """Advanced options for controlling text extraction."""

    max_characters: int | None = Field(
        default=None,
        alias="maxCharacters",
        description="Maximum character limit for the full page text.",
    )
    include_html_tags: bool | None = Field(
        default=None,
        alias="includeHtmlTags",
        description="Include HTML tags in the response.",
    )

    model_config = ConfigDict(populate_by_name=True)


class ContentsHighlightsOptions(BaseModel):
    """Configuration for highlights snippets."""

    num_sentences: int | None = Field(
        default=None,
        alias="numSentences",
        ge=1,
        description="The number of sentences to return for each snippet.",
    )
    highlights_per_url: int | None = Field(
        default=None,
        alias="highlightsPerUrl",
        ge=1,
        description="Number of snippets to return for each result.",
    )
    query: str | None = Field(default=None, description="Custom query to direct highlight selection.")

    model_config = ConfigDict(populate_by_name=True)


class ContentsSummaryOptions(BaseModel):
    """Configuration for content summaries."""

    query: str | None = Field(default=None, description="Custom query for the summary.")
    summary_schema: dict[str, Any] | None = Field(
        default=None,
        alias="schema",
        description="JSON schema for structured output.",
    )

    model_config = ConfigDict(populate_by_name=True)


class ContentsExtrasOptions(BaseModel):
    """Extra metadata extraction configuration."""

    links: int | None = Field(default=None, ge=0, description="Number of URLs to return per page.")
    image_links: int | None = Field(
        default=None,
        alias="imageLinks",
        ge=0,
        description="Number of image links to return per page.",
    )

    model_config = ConfigDict(populate_by_name=True)


class ContentsRequest(BaseModel):
    """Schema mirroring Exa's ContentsRequest payload."""

    text: bool | ContentsTextOptions | None = Field(
        default=None,
        description="If true or configured, returns full page text.",
    )
    highlights: ContentsHighlightsOptions | None = Field(
        default=None,
        description="Text snippets the LLM identifies as relevant.",
    )
    summary: ContentsSummaryOptions | None = Field(default=None, description="Summary configuration.")
    livecrawl: Literal["never", "fallback", "always", "preferred"] | None = Field(
        default=None,
        description="Livecrawl strategy for page fetches.",
    )
    livecrawl_timeout: int | None = Field(
        default=None,
        alias="livecrawlTimeout",
        description="Timeout for livecrawling in milliseconds.",
    )
    subpages: int | None = Field(
        default=None,
        description="Number of subpages to crawl per result.",
    )
    subpage_target: str | list[str] | None = Field(
        default=None,
        alias="subpageTarget",
        description="Keyword(s) to identify specific subpages.",
    )
    extras: ContentsExtrasOptions | None = Field(default=None, description="Extra parameters to return.")
    context: bool | None = Field(
        default=None,
        description="Return combined page contents as a context string for LLMs.",
    )

    model_config = ConfigDict(populate_by_name=True)


class ExaInputSchema(BaseModel):
    """Schema for Exa search input parameters."""

    query: str = Field(description="The search query string.")
    include_full_content: bool | None = Field(
        default=None,
        description="If true, retrieve full content, highlights, and summaries for search results.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    use_autoprompt: bool | None = Field(
        default=None,
        description="If true, query will be converted to a Exa query."
        "Enabled by default for auto search, optional for neural search, and not available for keyword search.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    query_type: QueryType | None = Field(
        default=None,
        description="Type of query to be used. Options are 'keyword', 'neural', or 'auto'."
        "Neural uses an embeddings-based model, keyword is google-like SERP. "
        "Default is auto, which automatically decides between keyword and neural.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    category: CategoryType | None = Field(
        default=None,
        description="A data category to focus on."
        "Options are company, research paper, news, pdf,"
        " github, tweet, personal site, linkedin profile, financial report.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description="Number of search results to return.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="List of domains to include in the search.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="List of domains to exclude from the search.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    include_text: list[str] | None = Field(
        default=None,
        description="Strings that must be present in webpage text.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    exclude_text: list[str] | None = Field(
        default=None,
        description="Strings that must not be present in webpage text.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    start_crawl_date: str | None = Field(
        default=None,
        description=("Only include links crawled after this ISO 8601 date. Expected format 2023-01-01T00:00:00.000Z."),
        json_schema_extra={"is_accessible_to_agent": True},
    )
    end_crawl_date: str | None = Field(
        default=None,
        description=("Only include links crawled before this ISO 8601 date. Expected format 2023-12-31T00:00:00.000Z."),
        json_schema_extra={"is_accessible_to_agent": True},
    )
    start_published_date: str | None = Field(
        default=None,
        description="Only include links with a published date after this ISO 8601 date.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    end_published_date: str | None = Field(
        default=None,
        description="Only include links with a published date before this ISO 8601 date.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    context: bool | dict[str, Any] | None = Field(
        default=None,
        description=(
            "Return page contents as a combined context string for LLMs. Can be a simple boolean or advanced options."
        ),
        json_schema_extra={"is_accessible_to_agent": True},
    )
    moderation: bool | None = Field(
        default=None,
        description="Enable Exa's content moderation filter for unsafe content.",
        json_schema_extra={"is_accessible_to_agent": True},
    )
    contents: ContentsRequest | None = Field(
        default=None,
        description="Advanced contents configuration mirroring Exa's ContentsRequest schema.",
        json_schema_extra={"is_accessible_to_agent": False},
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
        category (CategoryType): A data category to focus on.
        limit (int): Number of search results to return.
        include_domains (list[str], optional): List of domains to include.
        exclude_domains (list[str], optional): List of domains to exclude.
        include_text (list[str], optional): Strings that must be present.
        exclude_text (list[str], optional): Strings that must not be present.
        start_crawl_date (str, optional): Include links crawled after this date.
        end_crawl_date (str, optional): Include links crawled before this date.
        start_published_date (str, optional): Include links published after this date.
        end_published_date (str, optional): Include links published before this date.
        context (bool | dict, optional): Return combined context content for results.
        moderation (bool, optional): Enable Exa moderation filter.
        contents (dict, optional): Advanced contents configuration overriding include_full_content defaults.
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
    category: CategoryType | None = Field(default=None, description="A data category to focus on.")
    limit: int = Field(default=10, ge=1, le=100, description="Number of search results to return.")
    include_domains: list[str] | None = Field(default=None, description="List of domains to include in the search.")
    exclude_domains: list[str] | None = Field(default=None, description="List of domains to exclude from the search.")
    include_text: list[str] | None = Field(default=None, description="Strings that must be present in webpage text.")
    exclude_text: list[str] | None = Field(
        default=None, description="Strings that must not be present in webpage text."
    )
    start_crawl_date: str | None = Field(
        default=None, description="Only include links crawled after this ISO 8601 date."
    )
    end_crawl_date: str | None = Field(
        default=None, description="Only include links crawled before this ISO 8601 date."
    )
    start_published_date: str | None = Field(
        default=None, description="Only include links published after this ISO 8601 date."
    )
    end_published_date: str | None = Field(
        default=None, description="Only include links published before this ISO 8601 date."
    )
    context: bool | dict[str, Any] | None = Field(
        default=None, description="Return page contents as a combined context string for LLMs."
    )
    moderation: bool | None = Field(default=None, description="Enable Exa's content moderation filter.")
    contents: ContentsRequest | None = Field(
        default=None, description="Advanced contents configuration mirroring Exa's ContentsRequest schema."
    )

    MAX_SNIPPET_CHARS: ClassVar[int] = 800
    MAX_CONTEXT_CHARS: ClassVar[int] = 4000
    MAX_HIGHLIGHTS: ClassVar[int] = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[ExaInputSchema]] = ExaInputSchema

    @staticmethod
    def to_camel_case(snake_str: str) -> str:
        """Convert snake_case to camelCase."""
        components = snake_str.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        """Trim text to the specified length while keeping whole sentences when possible."""

        text = (text or "").strip()
        if not text or len(text) <= limit:
            return text

        truncated = text[:limit].rsplit(" ", 1)[0]
        return truncated.rstrip("\n") + "..."

    def _format_search_results(self, results: list[dict[str, Any]]) -> str:
        """
        Formats the search results into a human-readable string.

        Args:
            results (list[dict[str, Any]]): The raw search results.

        Returns:
            str: A formatted string containing the search results.
        """
        if not results:
            return "No results returned by Exa."

        formatted_results = []
        for index, result in enumerate(results, start=1):
            title = result.get("title") or DEFAULT_RESULT_TITLE
            url = result.get("url")
            published = result.get("publishedDate") or DEFAULT_RESULT_PUBLISHED_DATE
            author = result.get("author") or DEFAULT_RESULT_AUTHOR
            score = result.get("score")

            formatted_results.append(f"### Result {index}: {title}")
            if url:
                formatted_results.append(f"- URL: [{url}]({url})")
            else:
                formatted_results.append("- URL: Not available")

            formatted_results.extend(
                [
                    f"- Published: {published}",
                    f"- Author: {author}",
                    f"- Score: {score if score is not None else 'N/A'}",
                ]
            )

            highlights = (result.get("highlights") or [])[: self.MAX_HIGHLIGHTS]
            if highlights:
                formatted_results.append("- Highlights:")
                formatted_results.extend([f"  * {highlight.strip()}" for highlight in highlights])

            summary = (result.get("summary") or "").strip()
            if summary:
                formatted_results.append(f"- Summary: {summary}")

            text = (result.get("text") or "").strip()
            if text:
                snippet = self._truncate(text, self.MAX_SNIPPET_CHARS)
                if snippet:
                    formatted_results.append(f"- Snippet: {snippet}")

            formatted_results.append("")

        return "\n".join(formatted_results).strip()

    def _format_sources(self, results: list[dict[str, Any]]) -> list[str]:
        """Create markdown-friendly source list."""

        sources = []
        for result in results:
            title = result.get("title") or DEFAULT_RESULT_TITLE
            url = result.get("url")
            if url:
                sources.append(f"- [{title}]({url})")
            else:
                sources.append(f"- {title}")
        return sources

    def _format_context_section(self, context_blob: str | None) -> str | None:
        """Summarize the context blob returned by Exa without overwhelming the agent."""

        if not context_blob:
            return None

        return self._truncate(context_blob, self.MAX_CONTEXT_CHARS)

    @staticmethod
    def _serialize_contents(contents: ContentsRequest) -> dict[str, Any]:
        """Serialize a ContentsRequest into the API payload shape."""

        return contents.model_dump(by_alias=True, exclude_none=True)

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
            "startCrawlDate": (
                input_data.start_crawl_date if input_data.start_crawl_date is not None else self.start_crawl_date
            ),
            "endCrawlDate": (
                input_data.end_crawl_date if input_data.end_crawl_date is not None else self.end_crawl_date
            ),
            "startPublishedDate": (
                input_data.start_published_date
                if input_data.start_published_date is not None
                else self.start_published_date
            ),
            "endPublishedDate": (
                input_data.end_published_date if input_data.end_published_date is not None else self.end_published_date
            ),
            "context": input_data.context if input_data.context is not None else self.context,
            "moderation": input_data.moderation if input_data.moderation is not None else self.moderation,
        }

        if isinstance(payload["type"], QueryType):
            payload["type"] = payload["type"].value

        payload = {k: v for k, v in payload.items() if v is not None}

        include_full_content = (
            input_data.include_full_content
            if input_data.include_full_content is not None
            else self.include_full_content
        )
        contents_configuration = input_data.contents if input_data.contents is not None else self.contents

        if contents_configuration is not None:
            payload["contents"] = self._serialize_contents(contents_configuration)
        elif include_full_content:
            default_contents = ContentsRequest(
                text=ContentsTextOptions(max_characters=1000, include_html_tags=False),
                highlights=ContentsHighlightsOptions(
                    num_sentences=3,
                    highlights_per_url=2,
                    query=payload["query"],
                ),
                summary=ContentsSummaryOptions(query=f"Summarize the main points about {payload['query']}"),
                context=True,
            )
            payload["contents"] = self._serialize_contents(default_contents)

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

        sources_with_url = self._format_sources(results)
        formatted_context = self._format_context_section(search_result.get("context"))

        if self.is_optimized_for_agents:
            result_parts = ["## Sources", "\n".join(sources_with_url)]
            result_parts.extend(["## Search Results", formatted_results])
            if formatted_context:
                result_parts.extend(["## Context", formatted_context])
            result = "\n\n".join(result_parts)
        else:
            urls = [result.get("url") for result in results]
            result = {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "urls": urls,
                "context": formatted_context,
                "raw_response": search_result,
            }

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
