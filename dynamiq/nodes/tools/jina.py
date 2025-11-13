import enum
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Jina
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_SCRAPE = """Scrapes web content from URLs using Jina with CSS selector targeting and content filtering.

Key Capabilities:
- Clean text extraction with CSS selector precision
- Content filtering to remove unwanted elements
- Optional link and image extraction
- Selective targeting for specific page sections

Usage Strategy:
- Use target_selector for specific content areas
- Use remove_selector to filter out ads, navigation
- Enable include_links/include_images based on analysis needs

Parameter Guide:
- url: URL to scrape (e.g., "https://example.com")
- target_selector: CSS selector for specific content (".content", "#main")
- remove_selector: Filter unwanted elements (".ads", ".nav")
- include_links/include_images: Additional content extraction
- engine: "browser" for JS-heavy sites, "direct" for speed

Examples:
- {"url": "https://example.com", "target_selector": ".content"}
- {"url": "https://news.com", "remove_selector": ".ads"}
- {"url": "https://blog.com", "include_images": true}"""

DESCRIPTION_SEARCH = """Searches the web with the Jina Search API and returns LLM-ready SERP data.

Highlights:
- Supports geo/language targeting plus per-domain searches for authoritative sources
- Controls payload size with `include_full_content`, `X-Respond-With`, and image/links summaries
- Flexible renderers (markdown/html/text/screenshot/pageshot) and browser engines
- Complete header access (`X-With-*`, `X-Retain-Images`, `X-No-Cache`, `X-Proxy-Url`, etc.) for advanced use cases

Common Parameters:
- query (required): What to search for
- max_results (1-100), page: Tune breadth and pagination
- country, location, language, locale: Localize SERP responses
- site: Lock search scope to a domain
- return_format: markdown, html, text, screenshot, or pageshot
- include_links / include_images: Summaries (`true`) or exhaustive lists (`all`)
- include_favicons / include_favicon: Surface favicons per SERP result or per page
- include_full_content: Pull full article text (switches engine to browser automatically unless overridden)
- no_cache, timeout, cookies, proxy_url: Reliability controls
- retain_images, respond_with, engine: Low-level knobs for payload control

Example Invocations:
- {"query": "restaurants near downtown", "site": "yelp.com", "include_links": true}
- {"query": "latest ML papers", "return_format": "markdown", "include_full_content": true, "include_images": "all"}
- {"query": "Tokyo weather", "country": "JP", "language": "ja", "max_results": 5}"""


class JinaScrapeInputSchema(BaseModel):
    url: str | None = Field(None, description="URL of the page to scrape")
    target_selector: str | None = Field(None, description="CSS selector to focus on specific elements")
    remove_selector: str | None = Field(None, description="CSS selector to exclude elements")
    include_links: bool | None = Field(None, description="Include links summary")
    include_images: bool | None = Field(None, description="Include images summary")
    generate_alt_text: bool | None = Field(None, description="Generate alt text for images")
    engine: str | None = Field(None, description="Engine: 'browser' for JS-heavy sites, 'direct' for speed")


class JinaResponseFormat(str, enum.Enum):
    DEFAULT = "default"
    MARKDOWN = "markdown"
    HTML = "html"
    TEXT = "text"
    SCREENSHOT = "screenshot"
    PAGESHOT = "pageshot"


class JinaScrapeTool(ConnectionNode):
    """
    A tool for scraping web pages, powered by Jina Reader API.

    This class provides comprehensive web scraping capabilities with advanced
    customization options for content extraction and filtering.
    """

    SCRAPE_PATH: ClassVar[str] = "https://r.jina.ai/"

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Jina Scraper Tool"
    description: str = DESCRIPTION_SCRAPE
    response_format: JinaResponseFormat = JinaResponseFormat.MARKDOWN
    connection: Jina
    timeout: int = 60
    url: str | None = Field(None, description="URL to scrape")

    # Advanced options
    target_selector: str | None = Field(None, description="CSS selector to focus on specific elements")
    remove_selector: str | None = Field(None, description="CSS selector to exclude elements")
    include_links: bool = Field(default=False, description="Include links summary")
    include_images: bool = Field(default=False, description="Include images summary")
    generate_alt_text: bool = Field(default=True, description="Generate alt text for images")
    engine: str = Field(default="direct", description="Engine: 'browser' or 'direct'")
    no_cache: bool = Field(default=False, description="Bypass cache")

    model_config = ConfigDict(arbitrary_types_allowed=True)
    input_schema: ClassVar[type[JinaScrapeInputSchema]] = JinaScrapeInputSchema

    def _build_headers(self, input_data: JinaScrapeInputSchema) -> dict[str, str]:
        """Build request headers based on configuration and input parameters."""
        headers = {
            **self.connection.headers,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-Timeout": str(self.timeout),
        }

        if self.response_format != JinaResponseFormat.DEFAULT:
            headers["X-Return-Format"] = self.response_format.value

        engine = input_data.engine or self.engine
        if engine in ["browser", "direct", "cf-browser-rendering"]:
            headers["X-Engine"] = engine

        target_selector = input_data.target_selector or self.target_selector
        if target_selector:
            headers["X-Target-Selector"] = target_selector

        remove_selector = input_data.remove_selector or self.remove_selector
        if remove_selector:
            headers["X-Remove-Selector"] = remove_selector

        include_links = input_data.include_links if input_data.include_links is not None else self.include_links
        if include_links:
            headers["X-With-Links-Summary"] = "true"

        include_images = input_data.include_images if input_data.include_images is not None else self.include_images
        if include_images:
            headers["X-With-Images-Summary"] = "true"

        generate_alt = (
            input_data.generate_alt_text if input_data.generate_alt_text is not None else self.generate_alt_text
        )
        if generate_alt:
            headers["X-With-Generated-Alt"] = "true"

        if self.no_cache:
            headers["X-No-Cache"] = "true"

        return headers

    def _build_request_body(self, input_data: JinaScrapeInputSchema) -> dict[str, Any]:
        """Build request body according to Jina Reader API specification."""
        url = input_data.url or self.url
        if not url:
            raise ToolExecutionException(
                "No URL provided. Please provide a URL either during node initialization or execution.",
                recoverable=True,
            )

        return {"url": url}

    def _parse_response(self, response_data: dict) -> tuple[str, dict, dict]:
        """Parse Jina API response and extract content, links, and images."""
        if "data" not in response_data:
            raise ToolExecutionException(
                "Invalid response format from Jina API - missing 'data' field",
                recoverable=True,
            )

        data = response_data["data"]
        content = data.get("content", "")
        links = data.get("links", {})
        images = data.get("images", {})

        return content, links, images

    def execute(self, input_data: JinaScrapeInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Executes the web scraping process using the Jina Reader API.

        Args:
            input_data (JinaScrapeInputSchema): Input data for the tool
            config (RunnableConfig, optional): Configuration for the runnable
            **kwargs: Additional arguments

        Returns:
            dict[str, Any]: Dictionary containing the scraped content and metadata
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        headers = self._build_headers(input_data)
        request_body = self._build_request_body(input_data)

        try:
            response = self.client.request(
                method="POST",
                url=self.SCRAPE_PATH,
                headers=headers,
                json=request_body,
            )
            response.raise_for_status()

            if self.response_format in [JinaResponseFormat.PAGESHOT, JinaResponseFormat.SCREENSHOT]:
                scrape_result = response.content
                links, images = {}, {}
            else:
                response_data = response.json()
                scrape_result, links, images = self._parse_response(response_data)

        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to execute the requested action. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        url = request_body["url"]

        if self.is_optimized_for_agents:
            result_parts = [f"## Source URL\n{url}", f"## Scraped Content\n\n{scrape_result}"]

            if links:
                links_list = [f"- [{text}]({url})" for text, url in links.items()]
                result_parts.append("## Links Found\n" + "\n".join(links_list))

            if images:
                images_list = [f"- {desc}: {url}" for desc, url in images.items()]
                result_parts.append("## Images Found\n" + "\n".join(images_list))

            result = "\n\n".join(result_parts)
        else:
            result = {
                "url": url,
                "content": scrape_result,
                "links": links,
                "images": images,
                "metadata": {
                    "response_format": self.response_format.value,
                    "engine_used": headers.get("X-Engine", "direct"),
                    "target_selector": headers.get("X-Target-Selector"),
                    "remove_selector": headers.get("X-Remove-Selector"),
                },
            }

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return {"content": result}


SummaryPreference = bool | Literal["all"]
EnginePreference = Literal["browser", "direct", "cf-browser-rendering"]
RetainImagesPreference = Literal["none"]


class JinaSearchInputSchema(BaseModel):
    query: str | None = Field(None, description="Search query text")
    max_results: int | None = Field(None, description="Maximum number of search results (1-100)")
    country: str | None = Field(None, description="Two-letter country code for search region")
    location: str | None = Field(None, description="Geographic location for search origin")
    language: str | None = Field(None, description="Two-letter language code for results")
    page: int | None = Field(None, description="Page offset for pagination")
    site: str | None = Field(None, description="Limit search to specific domain")
    return_format: JinaResponseFormat | None = Field(
        None, description="Response format (markdown, html, text, screenshot, pageshot)"
    )
    include_images: SummaryPreference | None = Field(
        None, description="Include image summaries (True) or 'all' images per SERP entry"
    )
    include_links: SummaryPreference | None = Field(
        None, description="Include link summaries (True) or 'all' links per SERP entry"
    )
    include_favicons: bool | None = Field(None, description="Include SERP favicons")
    include_favicon: bool | None = Field(None, description="Include individual page favicon")
    include_full_content: bool | None = Field(None, description="Include full content of search results")
    retain_images: RetainImagesPreference | None = Field(
        None, description="Controls X-Retain-Images header (e.g., 'none' to strip images)"
    )
    respond_with: str | None = Field(None, description="Explicit X-Respond-With value")
    engine: EnginePreference | None = Field(None, description="Engine override ('browser', 'direct')")
    no_cache: bool | None = Field(None, description="Bypass cache for real-time data")
    generate_alt_text: bool | None = Field(None, description="Generate alt text for images without alt tags")
    timeout: int | None = Field(None, description="Request timeout in seconds")
    locale: str | None = Field(None, description="Browser locale setting")
    cookies: str | None = Field(None, description="Custom cookie settings")
    proxy_url: str | None = Field(None, description="Proxy URL for requests")


class JinaSearchTool(ConnectionNode):
    """
      A tool for performing web searches using the Jina AI API.

      This tool accepts various search parameters and returns relevant search results.

    Attributes:
          group (Literal[NodeGroup.TOOLS]): The group to which this tool belongs.
          name (str): The name of the tool.
          description (str): A brief description of the tool.
          connection (Jina): The connection instance for the Jina API.
          query (Optional[str]): The search query, can be set during initialization.
          max_results (int): Maximum number of results to return.
          country (Optional[str]): Two-letter country code for search region.
          location (Optional[str]): Geographic location for search origin.
          language (Optional[str]): Two-letter language code for results.
          page (int): Page offset for pagination.
          site (Optional[str]): Domain to limit search to.
          return_format (JinaResponseFormat): Response format preference.
          include_images (bool): Include images in search results.
          include_links (bool | Literal["all"]): Include link summaries or all links.
          include_favicons (bool): Include SERP favicons.
          include_favicon (bool): Include individual page favicon.
          include_full_content (bool): Include full content of search results.
          retain_images (Literal["none"] | None): Override X-Retain-Images header.
          respond_with (Optional[str]): Explicit X-Respond-With header (e.g., "no-content").
          engine (Optional[str]): Override engine ("browser", "direct", or "cf-browser-rendering").
          no_cache (bool): Bypass cache for real-time data.
          generate_alt_text (bool): Generate alt text for images.
          timeout (Optional[int]): Request timeout in seconds.
          locale (Optional[str]): Browser locale setting.
          cookies (Optional[str]): Custom cookie settings.
          proxy_url (Optional[str]): Proxy URL for requests.
    """

    SEARCH_PATH: ClassVar[str] = "https://s.jina.ai/"

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Jina Search Tool"
    description: str = DESCRIPTION_SEARCH
    connection: Jina
    query: str | None = Field(None, description="Search query")
    max_results: int = Field(default=5, ge=1, le=100, description="Maximum number of search results")

    country: str | None = Field(None, description="Two-letter country code (e.g., 'US', 'GB')")
    location: str | None = Field(None, description="Geographic location for search origin")
    language: str | None = Field(None, description="Two-letter language code (e.g., 'en', 'es')")

    page: int = Field(default=0, ge=0, description="Page offset for pagination")

    site: str | None = Field(None, description="Domain to limit search to")
    return_format: JinaResponseFormat = Field(default=JinaResponseFormat.DEFAULT, description="Response format")

    include_images: SummaryPreference = Field(
        default=False, description="Include image summaries (True) or 'all' images per SERP entry"
    )
    include_links: SummaryPreference = Field(
        default=False, description="Include link summaries (True) or 'all' links per SERP entry"
    )
    include_favicons: bool = Field(default=False, description="Include SERP favicons")
    include_favicon: bool = Field(default=False, description="Include individual page favicon")
    include_full_content: bool = Field(default=False, description="Include full content of results")
    retain_images: RetainImagesPreference | None = Field(
        default=None, description="Controls X-Retain-Images header (e.g., 'none' to strip images)"
    )
    respond_with: str | None = Field(default=None, description="Explicit X-Respond-With header value")
    engine: EnginePreference | None = Field(
        default=None, description="Override engine ('browser', 'direct', or 'cf-browser-rendering')"
    )

    no_cache: bool = Field(default=False, description="Bypass cache for real-time data")
    generate_alt_text: bool = Field(default=False, description="Generate alt text for images")
    timeout: int | None = Field(None, description="Request timeout in seconds")
    locale: str | None = Field(None, description="Browser locale setting")
    cookies: str | None = Field(None, description="Custom cookie settings")
    proxy_url: str | None = Field(None, description="Proxy URL for requests")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[JinaSearchInputSchema]] = JinaSearchInputSchema

    @staticmethod
    def _summary_header_value(value: SummaryPreference | None) -> str | None:
        """Map boolean or literal summary preferences to header values."""
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized == "all":
                return "all"
            if normalized in {"true", "unique"}:
                return "true"
            return None
        if value:
            return "true"
        return None

    def _build_headers(self, input_data: JinaSearchInputSchema, include_full: bool) -> dict[str, str]:
        """Build request headers based on configuration and input parameters."""
        headers = {
            **self.connection.headers,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        site = input_data.site or self.site
        if site:
            headers["X-Site"] = site

        include_links_value = input_data.include_links if input_data.include_links is not None else self.include_links
        links_header = self._summary_header_value(include_links_value)
        if links_header:
            headers["X-With-Links-Summary"] = links_header

        include_images_value = (
            input_data.include_images if input_data.include_images is not None else self.include_images
        )
        images_header = self._summary_header_value(include_images_value)
        if images_header:
            headers["X-With-Images-Summary"] = images_header

        retain_images_value = input_data.retain_images or self.retain_images
        if retain_images_value:
            headers["X-Retain-Images"] = retain_images_value
        elif not images_header:
            headers["X-Retain-Images"] = "none"

        no_cache = input_data.no_cache if input_data.no_cache is not None else self.no_cache
        if no_cache:
            headers["X-No-Cache"] = "true"

        generate_alt = (
            input_data.generate_alt_text if input_data.generate_alt_text is not None else self.generate_alt_text
        )
        if generate_alt:
            headers["X-With-Generated-Alt"] = "true"

        respond_with = input_data.respond_with or self.respond_with
        if respond_with:
            headers["X-Respond-With"] = respond_with
        elif not include_full:
            headers["X-Respond-With"] = "no-content"

        include_favicon = input_data.include_favicon if input_data.include_favicon is not None else self.include_favicon
        if include_favicon:
            headers["X-With-Favicon"] = "true"

        return_format = input_data.return_format or self.return_format
        if return_format != JinaResponseFormat.DEFAULT:
            headers["X-Return-Format"] = return_format.value

        engine = input_data.engine or self.engine
        if engine:
            headers["X-Engine"] = engine
        else:
            headers["X-Engine"] = "browser" if include_full else "direct"

        include_favicons = (
            input_data.include_favicons if input_data.include_favicons is not None else self.include_favicons
        )
        if include_favicons:
            headers["X-With-Favicons"] = "true"

        timeout = input_data.timeout or self.timeout
        if timeout:
            headers["X-Timeout"] = str(timeout)

        cookies = input_data.cookies or self.cookies
        if cookies:
            headers["X-Set-Cookie"] = cookies

        proxy_url = input_data.proxy_url or self.proxy_url
        if proxy_url:
            headers["X-Proxy-Url"] = proxy_url

        locale = input_data.locale or self.locale
        if locale:
            headers["X-Locale"] = locale

        return headers

    def _build_request_body(self, input_data: JinaSearchInputSchema) -> dict[str, Any]:
        """Build request body according to Jina API specification."""
        query = input_data.query or self.query
        if not query:
            raise ToolExecutionException(
                "No query provided. Please provide a query either during node initialization or execution.",
                recoverable=True,
            )

        body = {"q": query}

        max_results = input_data.max_results or self.max_results
        if max_results:
            body["num"] = max_results

        country = input_data.country or self.country
        if country:
            body["gl"] = country

        location = input_data.location or self.location
        if location:
            body["location"] = location

        language = input_data.language or self.language
        if language:
            body["hl"] = language

        page = input_data.page if input_data.page is not None else self.page
        if page > 0:
            body["page"] = page

        return body

    def _format_search_results(self, results: dict[str, Any], include_full: bool) -> str:
        """Format the search results into a readable string format."""
        formatted_results = []
        data = results.get("data", [])
        for idx, result in enumerate(data, start=1):
            title = result.get("title") or result.get("url") or f"Result {idx}"
            formatted_results.append(f"### Result {idx}: {title}")
            if result.get("url"):
                formatted_results.append(f"- URL: {result.get('url')}")
            if result.get("description"):
                formatted_results.append(f"- Description: {result.get('description')}")
            if include_full and result.get("content"):
                formatted_results.append(f"- Content: {result.get('content')}")
            if result.get("images"):
                image_lines = [f"    • {label}: {img_url}" for label, img_url in result["images"].items()]
                formatted_results.append("- Images:\n" + "\n".join(image_lines))
            formatted_results.append("")
        return "\n".join(formatted_results).strip()

    def execute(self, input_data: JinaSearchInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Execute the web search process with full API parameter support.

        Args:
            input_data (JinaSearchInputSchema): Input data with search parameters.
            config (RunnableConfig, optional): Configuration for the runnable.
            **kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the search results.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        include_full = (
            input_data.include_full_content
            if input_data.include_full_content is not None
            else self.include_full_content
        )

        headers = self._build_headers(input_data, include_full)
        request_body = self._build_request_body(input_data)

        try:
            response = self.client.request(
                method="POST",
                url=self.SEARCH_PATH,
                headers=headers,
                json=request_body,
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

        formatted_results = self._format_search_results(search_result, include_full)
        sources_with_url = [
            f"[{result.get('title') or result.get('url')}]({result.get('url')})"
            for result in search_result.get("data", [])
            if result.get("url")
        ]

        if self.is_optimized_for_agents:
            result_sections = [
                f"## Jina Search Results for '{request_body['q']}'",
                "",
            ]
            if sources_with_url:
                result_sections.extend(["### Sources", *sources_with_url, ""])
            if formatted_results:
                result_sections.append(formatted_results)
            else:
                result_sections.append("No results were returned by Jina Search.")
            result = "\n".join(result_sections).strip()
        else:
            images = {}
            for d in search_result.get("data", []):
                images.update(d.get("images", {}))

            result = {
                "result": formatted_results,
                "sources_with_url": sources_with_url,
                "raw_response": search_result,
                "images": images,
                "query": request_body["q"],
                "request_body": request_body,
                "headers_used": {k: v for k, v in headers.items() if k.startswith("X-")},
                "include_full_content": include_full,
            }

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return {"content": result}
