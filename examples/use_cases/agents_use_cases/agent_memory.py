from dynamiq.connections import Tavily, Firecrawl
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.tavily import TavilyTool
from dynamiq.nodes.tools.firecrawl import FirecrawlTool

from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm
from dynamiq.memory.inner_memory import InnerMemoryConfig, InnerMemory
from dynamiq.nodes.agents.utils import XMLParser
AGENT_ROLE = (
    "A helpful and general-purpose AI assistant"
)

REQUEST_AWS_PARTNERS = """Visit the AWS Partners directory and retrieve a complete list of all partners and short info about them, at least 100.
Be sure to collect data from all available pages. Use pagination by following the URL pattern:
https://partners.amazonaws.com/search/partners?page=2, page=3, and so on, until there are no more results. No wories about cookies etc, just continue scraping page by page. Start from page2"""

if __name__ == "__main__":
    connection_tavily = Tavily()
    connection_firecrawl = Firecrawl()

    tool_search = TavilyTool(connection=connection_tavily)
    tool_scrape = FirecrawlTool(connection=connection_firecrawl)
    llm = setup_llm(model_provider="claude", model_name="claude-3-5-sonnet-20241022", temperature=0)

    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search, tool_scrape],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        inner_memory_config=InnerMemoryConfig(inner_memory = InnerMemory())
    )

    result = agent.run(input_data={"input": REQUEST_AWS_PARTNERS,
                                    "files": None})

    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
