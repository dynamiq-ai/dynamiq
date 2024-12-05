from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import ZenRows
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger


class ZenRowsInputSchema(BaseModel):
    url: str = Field(default="", description="Parameter to provide a url of the page to scrape.")


class ZenRowsTool(ConnectionNode):
    """
    A tool for scraping web pages, powered by ZenRows.

    This class is responsible for scraping the content of a web page using ZenRows.
    """

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Zenrows Scraper Tool"
    description: str = (
        "A tool for scraping web pages, powered by ZenRows. "
        "You can use this tool to scrape the content of a web page."
    )
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
            # Perform the HTTP request using the ZenRows connection
            response = self.client.request(
                method=self.connection.method,
                url=self.connection.url,
                params={**self.connection.params, **params},
            )
            response.raise_for_status()
            scrape_result = response.text
        except Exception as e:
            logger.error(f"Tool {self.name} - {self.id}: failed to get results. Error: {e}")

            raise

        if self.is_optimized_for_agents:
            result = (
                f"<Source URL>\n{input_data.url}\n<\\Source URL>"
                f"\n<Scraped result>\n{scrape_result}\n<\\Scraped result>"
            )
        else:
            result = {"url": input_data.url, "content": scrape_result}
        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return {"content": result}
