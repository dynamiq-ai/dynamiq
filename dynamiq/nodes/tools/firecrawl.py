from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Firecrawl
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_FIRECRAWL = """## Firecrawl Web Scraping Tool
### Overview
The Firecrawl Tool is a powerful web scraping utility that extracts high-fidelity content from websites.
### Capabilities
- Extract content from any accessible webpage.
### When to Use
- When you need to extract information from a specific webpage.
- When you require content in a structured, readable format.
- When parsing complex web applications or content behind user interactions.
### Input Parameters
- **url** (string, required): URL of the webpage to scrape.
Must be a valid, accessible URL including protocol (http/https).
### Usage Examples
#### Basic Scraping
{
  "url": "https://example.com/article/123"
}
"""


class JsonOptions(BaseModel):
    """Options for configuring JSON extraction."""

    schema: dict | None = None
    system_prompt: str | None = Field(None, alias="systemPrompt")
    prompt: str | None = None


class LocationSettings(BaseModel):
    """Settings for location emulation."""

    country: str = "US"
    languages: list[str] | None = None


class Action(BaseModel):
    """Action to perform before content extraction."""

    type: str
    milliseconds: int | None = None
    selector: str | None = None


class FirecrawlInputSchema(BaseModel):
    url: str = Field(default="", description="Parameter to specify the url of the page to be scraped.")


class FirecrawlTool(ConnectionNode):
    """A tool for scraping web pages using the Firecrawl service."""
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Firecrawl Tool"
    description: str = DESCRIPTION_FIRECRAWL
    connection: Firecrawl
    url: str | None = None
    input_schema: ClassVar[type[FirecrawlInputSchema]] = FirecrawlInputSchema

    formats: list[str] = Field(default_factory=lambda: ["markdown"])
    only_main_content: bool = Field(default=True, alias="onlyMainContent")
    include_tags: list[str] | None = Field(default=None, alias="includeTags")
    exclude_tags: list[str] | None = Field(default=None, alias="excludeTags")
    headers: dict | None = None
    wait_for: int = Field(default=0, alias="waitFor")
    mobile: bool = False
    skip_tls_verification: bool = Field(default=False, alias="skipTlsVerification")
    timeout: int = 30000
    json_options: JsonOptions | None = Field(default=None, alias="jsonOptions")
    actions: list[Action] | None = None
    location: LocationSettings | None = None
    remove_base64_images: bool = Field(default=False, alias="removeBase64Images")
    block_ads: bool = Field(default=True, alias="blockAds")
    proxy: str | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    def _build_scrape_data(self, url: str) -> dict:
        """Build the request payload for the Firecrawl API."""
        base_data = {
            "url": url,
            "formats": self.formats,
            "onlyMainContent": self.only_main_content,
        }

        conditional_fields = {
            "includeTags": self.include_tags,
            "excludeTags": self.exclude_tags,
            "headers": self.headers,
            "waitFor": self.wait_for if self.wait_for > 0 else None,
            "mobile": self.mobile if self.mobile else None,
            "skipTlsVerification": self.skip_tls_verification if self.skip_tls_verification else None,
            "timeout": self.timeout if self.timeout != 30000 else None,
            "removeBase64Images": self.remove_base64_images if self.remove_base64_images else None,
            "proxy": self.proxy,
        }

        if not self.block_ads:
            conditional_fields["blockAds"] = False

        if self.json_options:
            conditional_fields["jsonOptions"] = self.json_options.model_dump(exclude_none=True, by_alias=True)
        if self.actions:
            conditional_fields["actions"] = [action.model_dump(exclude_none=True) for action in self.actions]
        if self.location:
            conditional_fields["location"] = self.location.model_dump(exclude_none=True)

        # Filter out None values and merge with base data
        filtered_fields = {k: v for k, v in conditional_fields.items() if v is not None}
        return {**base_data, **filtered_fields}

    def _format_agent_response(self, url: str, data: dict) -> str:
        """Format the response for agent consumption."""
        sections = [f"<Source URL>\n{url}\n<\\Source URL>"]

        format_mappings = {"content": "Scraped result", "markdown": "Markdown", "html": "HTML", "json": "JSON"}

        for data_key, section_name in format_mappings.items():
            if data_key in data:
                sections.append(f"<{section_name}>\n{data[data_key]}\n<\\{section_name}>")

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

        scrape_data = self._build_scrape_data(url)
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

        data = scrape_result.get("data", {})

        if self.is_optimized_for_agents:
            result = self._format_agent_response(url, data)
        else:
            result = {"success": scrape_result.get("success", False), "url": url, **data}

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
