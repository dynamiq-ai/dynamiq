from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

from dynamiq.connections import ZenRows
from dynamiq.nodes import NodeGroup
from dynamiq.nodes.agents.exceptions import ToolExecutionException
from dynamiq.nodes.node import ConnectionNode, ensure_config
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger

DESCRIPTION_ZENROWS = """## Web Content Extraction Tool
### Purpose
Extract and convert web page content into readable text format for analysis and processing.

### When to Use
- Extract content from specific articles, blogs, or documentation pages
- Retrieve up-to-date information not available in your knowledge base
- Parse content from complex web applications or dynamic pages
- Convert web content to structured text for further processing
- Access content behind forms or interactive elements

### Key Capabilities
- Extract clean text content from any accessible webpage
- Handle JavaScript-heavy sites and dynamic content
- Convert HTML to readable markdown format
- Bypass common anti-scraping measures
- Process single-page applications (SPAs)
- Extract content from password-protected or geo-restricted sites

### Required Parameters
- **url** (string): Complete URL of the webpage to extract content from

### Optional Parameters
- **markdown_response** (boolean): Return content in markdown format (default: true)

### Usage Examples
#### Extract Article Content
```json
{
  "url": "https://www.example.com/blog/latest-technology-trends"
}
```

#### Extract Documentation
```json
{
  "url": "https://docs.example.com/api/authentication"
}
```

#### Extract Product Information
```json
{
  "url": "https://shop.example.com/products/laptop-model-xyz"
}
```

### Best Practices
1. **Use complete URLs** including protocol (https:// or http://)
2. **Verify URL accessibility** before extraction attempts
3. **Respect robots.txt** and website terms of service
4. **Use sparingly** to avoid overwhelming target servers
5. **Prefer official APIs** when available over scraping
6. **Handle sensitive content** with appropriate data protection measures
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
            result = f"## Source URL\n{input_data.url}\n\n## Scraped Result\n\n{scrape_result}\n"
        else:
            result = {"url": input_data.url, "content": scrape_result}
        logger.info(f"Tool {self.name} - {self.id}: finished with result:\n{str(result)[:200]}...")
        return {"content": result}
