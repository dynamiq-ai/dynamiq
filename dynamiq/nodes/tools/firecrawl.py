import json
from copy import deepcopy
from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Firecrawl
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_FIRECRAWL = """Scrapes web content

Key Capabilities:
- Fully rendered pages with JavaScript support, proxy rotation, and mobile emulation
- Mix-and-match output formats (markdown, html, rawHtml, links, summary, screenshot, json, changeTracking, branding)
- Fine control over scraping context: caching, headers, parsers (PDF handling), location, timeouts, and TLS settings
- Pre-scrape actions (wait, click, scroll, screenshot, custom JS) for complex pages

Usage Strategy:
- Set `only_main_content` to True for article-style pages; disable when layout context matters
- Combine multiple `formats` (e.g. `['markdown', {'type': 'json', 'schema': {...}}]`)
for both readable and structured data
- Use `max_age` or `store_in_cache=False` depending on freshness vs cost requirements
- Supply `actions` and `wait_for` when dynamic content needs interaction before scraping

Parameter Guide:
- url: target URL (required)
- formats: list of format strings or objects from Firecrawl's Formats schema
- actions: ordered list describing automation steps before scraping
- location/proxy/mobile: control geography, language, and device profile for tougher sites

Example:
{"url": "https://example.com", "formats": ["markdown", "links"], "only_main_content": True, "proxy": "auto"}"""


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
        description="Firecrawl formats (e.g. ['markdown', {'type': 'json', 'schema': {...}}]).",
    )
    only_main_content: bool | None = Field(
        default=None,
        alias="onlyMainContent",
        description="When True, return only the primary article/content region.",
    )
    include_tags: list[str] | None = Field(default=None, alias="includeTags")
    exclude_tags: list[str] | None = Field(default=None, alias="excludeTags")
    max_age: int | None = Field(
        default=None,
        alias="maxAge",
        description="Use cached copy if newer than this age in ms.",
    )
    headers: dict[str, Any] | None = Field(default=None, description="Custom headers/cookies for the request.")
    wait_for: int | None = Field(default=None, alias="waitFor", description="Delay (ms) before extracting content.")
    mobile: bool | None = Field(default=None, description="Emulate mobile device if True.")
    skip_tls_verification: bool | None = Field(default=None, alias="skipTlsVerification")
    timeout: int | None = Field(default=None, description="Request timeout in ms.")
    parsers: list[str | dict[str, Any]] | None = Field(
        default=None,
        description="Control how files (e.g., PDFs) are processed. Accepts ['pdf'] or parser objects.",
    )
    actions: list[Action] | None = Field(default=None, description="Automation steps (wait, click, screenshot, etc.).")
    location: LocationSettings | None = Field(default=None, description="Location emulation (country + languages).")
    remove_base64_images: bool | None = Field(default=None, alias="removeBase64Images")
    block_ads: bool | None = Field(default=None, alias="blockAds")
    proxy: Literal["basic", "stealth", "auto"] | None = Field(
        default=None, description="Proxy mode. Defaults to Firecrawl auto when omitted."
    )
    store_in_cache: bool | None = Field(default=None, alias="storeInCache")
    zero_data_retention: bool | None = Field(default=None, alias="zeroDataRetention")


class FirecrawlTool(ConnectionNode):
    """A tool for scraping web pages using the Firecrawl service."""
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Firecrawl Tool"
    description: str = DESCRIPTION_FIRECRAWL
    connection: Firecrawl
    url: str | None = None
    input_schema: ClassVar[type[FirecrawlInputSchema]] = FirecrawlInputSchema

    formats: list[str | dict[str, Any]] = Field(default_factory=lambda: ["markdown"])
    only_main_content: bool = Field(default=True, alias="onlyMainContent")
    include_tags: list[str] | None = Field(default=None, alias="includeTags")
    exclude_tags: list[str] | None = Field(default=None, alias="excludeTags")
    max_age: int | None = Field(default=None, alias="maxAge")
    headers: dict[str, Any] | None = None
    wait_for: int = Field(default=0, alias="waitFor")
    mobile: bool = False
    skip_tls_verification: bool = Field(default=True, alias="skipTlsVerification")
    timeout: int = 30000
    parsers: list[str | dict[str, Any]] | None = None
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
            "## Firecrawl Scrape Result",
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

    def execute(
        self, input_data: FirecrawlInputSchema, config: RunnableConfig | None = None, **kwargs
    ) -> dict[str, Any]:
        """Execute the scraping tool with the provided input data."""
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        url = input_data.url or self.url
        if not url:
            logger.error(f"Tool {self.name} - {self.id}: failed to get input data.")
            raise ValueError("URL is required for scraping")

        scrape_data = self._build_scrape_data(url, input_data)
        connection_url = self.connection.url + "scrape"

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
            logger.error(
                f"Tool {self.name} - {self.id}: failed to get results. Error: {e}"
            )
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to execute the requested action. Error: {str(e)}. "
                f"Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            result = self._format_agent_response(url, scrape_result)
        else:
            result = {"success": scrape_result.get("success", False), "url": url, **(scrape_result.get("data") or {})}

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
