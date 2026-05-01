import json
from copy import deepcopy
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Firecrawl
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.nodes.types import ActionType
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_FIRECRAWL = """Scrapes web content from a URL and returns cleaned content.

Key capabilities:
- Precise content targeting with CSS tag inclusion/exclusion filters
- Optional geo-location for region-aware scraping

Usage strategy:
- Use exclude_tags to strip navigation, ads, or footers from the output
- Use include_tags to force-include specific HTML elements (e.g. tables, images)
- Set location to emulate a specific country/language for region-aware pages

Examples:
{"url": "https://example.com"}
{"url": "https://blog.com", "exclude_tags": ["nav", "footer"]}
{"url": "https://shop.com", "include_tags": ["table"], "location": {"country": "DE"}}"""


class LocationSettings(BaseModel):
    """Settings for location emulation."""

    country: str = "US"
    languages: list[str] | None = None


class Action(BaseModel):
    """Action to perform before content extraction."""

    type: str
    milliseconds: int | None = None
    selector: str | None = None
    text: str | None = None
    key: str | None = None
    all: bool | None = None
    full_page: bool | None = Field(default=None, alias="fullPage")
    quality: int | None = None
    viewport: dict[str, int] | None = None
    script: str | None = None
    format: str | None = None
    landscape: bool | None = None
    scale: float | None = None

    model_config = ConfigDict(extra="allow", populate_by_name=True)


class FirecrawlInputSchema(BaseModel):
    """Schema exposed to agents using the tool."""

    model_config = ConfigDict(populate_by_name=True)

    url: str = Field(..., description="URL of the page to scrape.")
    formats: list[str | dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Tool output formats. Accepts plain strings (markdown/html/rawHtml/links/summary/"
            "screenshot/json/changeTracking/branding) or objects with format-specific options."
        ),
        json_schema_extra={"is_accessible_to_agent": False},
    )
    only_main_content: bool | None = Field(
        default=None,
        alias="onlyMainContent",
        description="True trims boilerplate (nav/footer) for article-style pages; False keeps the full DOM.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    include_tags: list[str] | None = Field(
        default=None,
        alias="includeTags",
        description="List of HTML tag names to force-include in the output (e.g. ['table', 'img']).",
    )
    exclude_tags: list[str] | None = Field(
        default=None,
        alias="excludeTags",
        description="HTML tag names that should be stripped from the response.",
    )
    max_age: int | None = Field(
        default=None,
        alias="maxAge",
        description="Cache freshness window in ms (default is 172800000 = two days).",
        json_schema_extra={"is_accessible_to_agent": False}
    )
    headers: dict[str, Any] | None = Field(
        default=None,
        description="Custom HTTP headers (cookies, user-agent, auth tokens) to forward with the scrape.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    wait_for: int | None = Field(
        default=None,
        alias="waitFor",
        description="Delay in ms before scraping to let dynamic content render (default 0).",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    mobile: bool | None = Field(
        default=None,
        description="True emulates a mobile device viewport + UA, useful for responsive layouts/screenshots.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    skip_tls_verification: bool | None = Field(
        default=None,
        alias="skipTlsVerification",
        description="True disables TLS verification (default), set False for strict cert checks.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    timeout: int | None = Field(
        default=None,
        description="Request timeout in ms for the upstream fetch.",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    parsers: list[str | dict[str, Any]] | None = Field(
        default=None,
        description=(
            "Controls file parsers. Defaults to ['pdf']; pass [] to disable auto PDF parsing or "
            "objects like {'type': 'pdf', 'maxPages': 5} to limit cost."
        ),
        json_schema_extra={"is_accessible_to_agent": False},
    )
    actions: list[Action] | None = Field(
        default=None,
        description=(
            "Optional automation instructions executed before scraping. Supports wait/screenshot/click/write/"
            "press/scroll/scrape/executeJavascript/pdf actions following the tool's schema."
        ),
        json_schema_extra={"is_accessible_to_agent": False},
    )
    location: LocationSettings | None = Field(
        default=None,
        description="Country + preferred languages for proxy/language emulation (defaults to US if omitted).",
    )
    remove_base64_images: bool | None = Field(
        default=None,
        alias="removeBase64Images",
        description="True strips giant base64 <img> blobs and replaces them with placeholders (default True).",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    block_ads: bool | None = Field(
        default=None,
        alias="blockAds",
        description="Enable ad + cookie popup blocking (default True).",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    proxy: Literal["basic", "stealth", "auto"] | None = Field(
        default=None,
        description=(
            "Proxy tier: 'basic' (fast, low protection), 'stealth' (solves harder anti-bot, higher cost), or "
            "'auto' (retry with stealth on failure by default)."
        ),
        json_schema_extra={"is_accessible_to_agent": False},
    )
    store_in_cache: bool | None = Field(
        default=None,
        alias="storeInCache",
        description="True lets the tool index/cache the page "
        "(default True, forced False when using sensitive params).",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    zero_data_retention: bool | None = Field(
        default=None,
        alias="zeroDataRetention",
        description="Opt-in compliance flag; True enables zero data retention mode (requires approval).",
        json_schema_extra={"is_accessible_to_agent": False},
    )
    brief: str = Field(
        default="Scraping the web for information.",
        description="Very brief description of the action being performed. Example: 'Scrape the web for information.'.",
    )


class FirecrawlTool(ConnectionNode):
    """A tool for scraping web pages using the Firecrawl service."""
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    action_type: ActionType = ActionType.WEB_SCRAPE
    name: str = "firecrawl-scrape"
    description: str = DESCRIPTION_FIRECRAWL
    is_parallel_execution_allowed: bool = True
    connection: Firecrawl
    url: str | None = None
    input_schema: ClassVar[type[FirecrawlInputSchema]] = FirecrawlInputSchema

    formats: list[str | dict[str, Any]] = Field(default_factory=lambda: ["markdown"])
    only_main_content: bool = Field(default=True, alias="onlyMainContent")
    include_tags: list[str] | None = Field(default=None, alias="includeTags")
    exclude_tags: list[str] | None = Field(default=None, alias="excludeTags")
    max_age: int | None = Field(default=None, alias="maxAge")
    headers: dict[str, Any] | None = Field(default=None, description="Custom HTTP headers to forward with the scrape.")
    wait_for: int = Field(default=0, alias="waitFor")
    mobile: bool = Field(default=False, description="Emulate a mobile device viewport and user-agent.")
    skip_tls_verification: bool = Field(default=True, alias="skipTlsVerification")
    timeout: int = Field(default=30000, description="Request timeout in milliseconds for the upstream fetch.")
    parsers: list[str | dict[str, Any]] | None = Field(
        default=None, description="File parsers to apply. Defaults to ['pdf'] when None."
    )
    actions: list[Action] | None = None
    location: LocationSettings | None = None
    remove_base64_images: bool = Field(default=True, alias="removeBase64Images")
    block_ads: bool = Field(default=True, alias="blockAds")
    proxy: Literal["basic", "stealth", "auto"] | None = None
    store_in_cache: bool | None = Field(default=None, alias="storeInCache")
    zero_data_retention: bool | None = Field(default=None, alias="zeroDataRetention")

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    def _build_scrape_data(self, url: str, overrides: FirecrawlInputSchema) -> dict:
        """Build the request payload for the Firecrawl API."""

        def resolve(field_name: str) -> Any:
            if hasattr(overrides, field_name):
                value = getattr(overrides, field_name)
                if value is not None:
                    return value
            return deepcopy(getattr(self, field_name, None))

        formats = resolve("formats") or ["markdown"]
        only_main_content = resolve("only_main_content")
        if only_main_content is None:
            only_main_content = True

        base_data = {
            "url": url,
            "formats": formats,
            "onlyMainContent": only_main_content,
        }

        conditional_fields = {
            "includeTags": resolve("include_tags"),
            "excludeTags": resolve("exclude_tags"),
            "maxAge": resolve("max_age"),
            "headers": resolve("headers"),
            "waitFor": resolve("wait_for"),
            "mobile": resolve("mobile"),
            "skipTlsVerification": resolve("skip_tls_verification"),
            "timeout": resolve("timeout"),
            "parsers": resolve("parsers"),
            "removeBase64Images": resolve("remove_base64_images"),
            "blockAds": resolve("block_ads"),
            "proxy": resolve("proxy"),
            "storeInCache": resolve("store_in_cache"),
            "zeroDataRetention": resolve("zero_data_retention"),
        }

        actions = resolve("actions")
        if actions:
            conditional_fields["actions"] = [
                action.model_dump(exclude_none=True, by_alias=True) if isinstance(action, Action) else action
                for action in actions
            ]

        location = resolve("location")
        if location:
            conditional_fields["location"] = (
                location.model_dump(exclude_none=True) if isinstance(location, BaseModel) else location
            )

        # Filter out None values and merge with base data
        filtered_fields = {k: v for k, v in conditional_fields.items() if v is not None}
        return {**base_data, **filtered_fields}

    @staticmethod
    def _json_section(title: str, payload: Any) -> str:
        return f"## {title}\n```json\n{json.dumps(payload, indent=2, ensure_ascii=False)}\n```"

    def _format_agent_response(self, url: str, response: dict[str, Any]) -> str:
        """Format the response for agent consumption using Markdown."""
        data = response.get("data", {}) or {}
        sections = [
            "## Web Scraping Result",
            f"- URL: {url}",
            f"- Success: {response.get('success', False)}",
        ]

        warning = data.get("warning")
        if warning:
            sections.append(f"- Warning: {warning}")

        content_fields = [
            ("markdown", "Markdown"),
            ("summary", "Summary"),
            ("html", "HTML"),
            ("rawHtml", "Raw HTML"),
        ]

        for key, label in content_fields:
            value = data.get(key)
            if value:
                sections.append(f"## {label}\n{value}")

        if screenshot := data.get("screenshot"):
            sections.append(f"## Screenshot URL\n{screenshot}")

        if links := data.get("links"):
            links_section = "\n".join(f"- {link}" for link in links)
            sections.append(f"## Links\n{links_section}")

        actions = data.get("actions")
        if actions:
            sections.append(self._json_section("Action Results", actions))

        metadata = data.get("metadata")
        if metadata:
            sections.append(self._json_section("Metadata", metadata))

        change_tracking = data.get("changeTracking")
        if change_tracking:
            sections.append(self._json_section("Change Tracking", change_tracking))

        branding = data.get("branding")
        if branding:
            sections.append(self._json_section("Branding", branding))

        return "\n\n".join(sections)

    def _resolve_scrape_request(self, input_data: FirecrawlInputSchema) -> tuple[str, dict, str]:
        url = input_data.url or self.url
        if not url:
            logger.error(f"Tool {self.name} - {self.id}: failed to get input data.")
            raise ValueError("URL is required for scraping")
        scrape_data = self._build_scrape_data(url, input_data)
        connection_url = self.connection.url + "scrape"
        return url, scrape_data, connection_url

    def _wrap_scrape_exception(self, exc: Exception) -> ToolExecutionException:
        logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {exc}")
        return ToolExecutionException(
            f"Tool '{self.name}' failed to execute the requested action. Error: {str(exc)}. "
            f"Please analyze the error and take appropriate action.",
            recoverable=True,
        )

    def _format_scrape_output(self, url: str, scrape_result: dict) -> dict[str, Any]:
        if self.is_optimized_for_agents:
            result = self._format_agent_response(url, scrape_result)
            data = scrape_result.get("data") or {}
            page_metadata = data.get("metadata") or {}
            output = {"content": result, "urls": [url], "page_metadata": page_metadata}
        else:
            result = {"success": scrape_result.get("success", False), "url": url, **(scrape_result.get("data") or {})}
            output = {"content": result}
        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return output

    def execute(
        self, input_data: FirecrawlInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute the scraping tool with the provided input data."""
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        url, scrape_data, connection_url = self._resolve_scrape_request(input_data)

        try:
            response = self.client.request(
                method=self.connection.method,
                url=connection_url,
                json=scrape_data,
                headers=self.connection.headers,
            )
            response.raise_for_status()
            scrape_result = response.json()
        except Exception as e:
            raise self._wrap_scrape_exception(e)

        return self._format_scrape_output(url, scrape_result)

    async def execute_async(
        self, input_data: FirecrawlInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Native async execution path mirroring ``execute``."""
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        url, scrape_data, connection_url = self._resolve_scrape_request(input_data)
        client = await self.get_async_client()

        try:
            response = await client.request(
                method=self.connection.method,
                url=connection_url,
                json=scrape_data,
                headers=self.connection.headers,
            )
            response.raise_for_status()
            scrape_result = response.json()
        except Exception as e:
            raise self._wrap_scrape_exception(e)

        return self._format_scrape_output(url, scrape_result)
