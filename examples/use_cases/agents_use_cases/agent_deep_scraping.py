from dynamiq.connections import Firecrawl, Tavily
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.file_storage import InMemoryFileStorage
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """A helpful and general-purpose AI assistant with access to filesystem.
 Save useful information page by page in filesystem."""

PROMPT1 = """Parse all aws partners having by scraping pages of
https://clutch.co/developers/artificial-intelligence/generative?page=1
 and generate csv like file with this structure
 Company Name,Rating,Reviews,Location,Minimum Project Size,Hourly Rate,Company Size,Services Focus."""

PROMPT2 = """Create long research on state of AI in EU. Give report for each country."""

if __name__ == "__main__":
    connection_firecrawl = Firecrawl()

    tool_scrape = FirecrawlTool(connection=connection_firecrawl)
    llm = setup_llm(model_provider="claude", model_name="claude-3-7-sonnet-20250219", temperature=0)

    storage = InMemoryFileStorage()

    agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_scrape],
        role=AGENT_ROLE,
        max_loops=30,
        inference_mode=InferenceMode.XML,
        filestorage=storage,
        summarization_config=SummarizationConfig(enabled=True, max_token_context_length=100000),
    )

    result = agent.run(input_data={"input": PROMPT1, "files": None})

    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
