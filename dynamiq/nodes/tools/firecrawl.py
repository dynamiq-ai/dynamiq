from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import Firecrawl
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class PageOptions(BaseModel):
    """Options for configuring page scraping behavior."""
    headers: dict | None = None
    include_html: bool = False
    only_main_content: bool = False
    remove_tags: list[str] | None = None
    wait_for: int = 0


class ExtractorOptions(BaseModel):
    """Options for configuring content extraction behavior."""
    mode: Literal[
        "markdown",
        "llm-extraction",
        "llm-extraction-from-raw-html",
        "llm-extraction-from-markdown",
    ] = "markdown"
    extraction_prompt: str | None = None
    extraction_schema: dict | None = None


class FirecrawlInputSchema(BaseModel):
    url: str = Field(default="", description="Parameter to specify the url of the page to be scraped.")


class FirecrawlTool(ConnectionNode):
    """A tool for scraping web pages using the Firecrawl service."""
    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Firecrawl Tool"
    description: str = (
        "A tool for scraping web pages, powered by Firecrawl."
        "You can use this tool to scrape the content of a web page."
    )
    connection: Firecrawl
    url: str | None = None
    input_schema: ClassVar[type[FirecrawlInputSchema]] = FirecrawlInputSchema

    # Default parameters
    page_options: PageOptions = Field(
        default_factory=PageOptions, description="The options for scraping the page"
    )
    extractor_options: ExtractorOptions = Field(
        default_factory=ExtractorOptions,
        description="The options for extracting content via service LLM",
    )
    timeout: int = 60000

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @staticmethod
    def to_camel_case(snake_str: str) -> str:
        components = snake_str.split("_")
        return components[0] + "".join(x.title() for x in components[1:])

    @staticmethod
    def dict_to_camel_case(data: dict | list) -> dict | list:
        if isinstance(data, dict):
            return {
                FirecrawlTool.to_camel_case(key): (
                    FirecrawlTool.dict_to_camel_case(value)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [FirecrawlTool.dict_to_camel_case(item) for item in data]
        else:
            return data

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

        scrape_data = {
            "url": url,
            "pageOptions": self.dict_to_camel_case(
                self.page_options.model_dump(exclude_none=True)
            ),
            "extractorOptions": self.dict_to_camel_case(
                self.extractor_options.model_dump(exclude_none=True)
            ),
            "timeout": self.timeout,
        }

        connection_url = self.connection.url + "scrape/"

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
            raise

        if self.is_optimized_for_agents:
            result = f"<Source URL>\n{url}\n<\\Source URL>"
            if scrape_result.get("data", {}).get("content", "") != "":
                result += f"\n\n<Scraped result>\n{scrape_result.get('data', {}).get('content')}\n<\\Scraped result>"
            if scrape_result.get("data", {}).get("markdown", "") != "":
                result += f"\n\n<Markdown>\n{scrape_result.get('data', {}).get('markdown')}\n<\\Markdown>"
            if scrape_result.get("data", {}).get("llm_extraction", "") != "":
                result += (
                    f"\n\n<LLM Extraction>\n{scrape_result.get('data', {}).get('llm_extraction')}\n<\\LLM Extraction>"
                )
        else:
            result = {
                "success": scrape_result.get("success", False),
                "url": url,
                "markdown": scrape_result.get("data", {}).get("markdown", ""),
                "content": scrape_result.get("data", {}).get("content", ""),
                "html": scrape_result.get("data", {}).get("html"),
                "raw_html": scrape_result.get("data", {}).get("rawHtml"),
                "metadata": scrape_result.get("data", {}).get("metadata", {}),
                "llm_extraction": scrape_result.get("data", {}).get("llm_extraction"),
                "warning": scrape_result.get("data", {}).get("warning"),
            }

        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")

        return {"content": result}
