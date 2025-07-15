from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import ZenRows
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_ZENROWS = """Scrapes web content from URLs using ZenRows with advanced anti-bot protection and JavaScript rendering. Handles complex websites with proxy rotation, CAPTCHA solving, and browser automation for reliable data extraction.

Key Capabilities:
- Bypass anti-bot protection and access blocked websites
- JavaScript rendering for dynamic content and SPAs
- Automatic proxy rotation and CAPTCHA solving
- Convert HTML to clean Markdown format for easy processing

Usage Strategy:
Use for websites that block standard scrapers or require JavaScript execution.

Parameter Guide:
- url: Target website URL to scrape

Examples:
- Basic scraping: {"url": "https://example.com"}
- E-commerce data: {"url": "https://shop.example.com/products"}
- News articles: {"url": "https://news.example.com/article/123"}"""  # noqa: E501


class ZenRowsInputSchema(BaseModel):
    url: str = Field(default="", description="Parameter to provide a url of the page to scrape.")


class ZenRowsTool(ConnectionNode):
    """
    A tool for scraping web pages, powered by ZenRows.

    This class is responsible for scraping the content of a web page using ZenRows.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Zenrows Scraper Tool"
    description: str = DESCRIPTION_ZENROWS
    connection: ZenRows
    url: str | None = None
    markdown_response: bool = Field(
        default=True,
        description="If True, the content will be parsed as Markdown instead of HTML.",
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_schema: ClassVar[type[ZenRowsInputSchema]] = ZenRowsInputSchema

    def execute(self, input_data: ZenRowsInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        """
        Executes the web scraping process.

        Args:
            input_data (dict[str, Any]): A dictionary containing 'input' key with the URL to scrape.
            config (RunnableConfig, optional): Configuration for the runnable, including callbacks.
            kwargs: Additional arguments passed to the execution context.

        Returns:
            dict[str, Any]: A dictionary containing the URL and the scraped content.
        """
        logger.info(f"Tool {self.name} - {self.id}: started with input:\n{input_data.model_dump()}")

        # Ensure the config is set up correctly
        config = ensure_config(config)
        self.run_on_node_execute_run(config.callbacks, **kwargs)

        params = {
            "url": input_data.url,
            "markdown_response": str(self.markdown_response).lower(),
        }

        try:
            response = self.client.request(
                method=self.connection.method,
                url=self.connection.url,
                params={**self.connection.params, **params},
            )
            response.raise_for_status()
            scrape_result = response.text
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {e}")
            raise ToolExecutionException(
                f"Tool '{self.name}' failed to execute the requested action. "
                f"Error: {str(e)}. Please analyze the error and take appropriate action.",
                recoverable=True,
            )

        if self.is_optimized_for_agents:
            result = f"## Source URL\n{input_data.url}\n\n## Scraped Result\n\n{scrape_result}\n"
        else:
            result = {"url": input_data.url, "content": scrape_result}
        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return {"content": result}
