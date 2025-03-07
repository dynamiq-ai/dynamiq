from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import ZenRows
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_ZENROWS = """## ZenRows Web Scraper Tool
### Description
A powerful web scraping tool that extracts content from any web page.
### Capabilities
- Extracts complete text from web pages
### When to Use
- To access specific web content not in your knowledge base or requiring up-to-date information.
- To extract information from articles, documentation, or product pages.
- When you need detailed content from a known URL rather than general search results.
### Input Parameters
- `url` (required): The complete URL of the web page to scrape (e.g., "https://www.example.com/article/12345").
### Usage Examples
{"url": "https://www.bbc.com/news/science-environment-12345678"}
### Notes
- Always provide the complete URL including the protocol (http:// or https://).
"""


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
            **kwargs: Additional arguments passed to the execution context.

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
            result = (
                f"<Source URL>\n{input_data.url}\n<\\Source URL>"
                f"\n<Scraped result>\n{scrape_result}\n<\\Scraped result>"
            )
        else:
            result = {"url": input_data.url, "content": scrape_result}
        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return {"content": result}
