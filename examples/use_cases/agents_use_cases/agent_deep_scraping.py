from dynamiq.connections.connections import E2B, ScaleSerp
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.file import InMemoryFileStore
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """A helpful and general-purpose AI assistant with access to filesystem.
 Save useful information page by page in filesystem."""

PROMPT1 = """Parse all aws partners having by scraping pages of
https://clutch.co/developers/artificial-intelligence/generative?page=1
 and generate csv like file with this structure
 Company Name,Rating,Reviews,Location,Minimum Project Size,Hourly Rate,Company Size,Services Focus."""

PROMPT2 = """Create long research on state of AI in EU. Give report for each country.
Once you saved usefull information you can clean context"""

if __name__ == "__main__":
    connection_scale_serp = ScaleSerp()

    tool_scrape = ScaleSerpTool(connection=connection_scale_serp)
    e2b = E2B()
    tool_code = E2BInterpreterTool(connection=e2b)
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0)

    storage = InMemoryFileStore()

    agent = Agent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_scrape, tool_code],
        role=AGENT_ROLE,
        max_loops=30,
        inference_mode=InferenceMode.XML,
        file_store=FileStoreConfig(enabled=True, backend=storage, agent_file_write_enabled=True),
        summarization_config=SummarizationConfig(enabled=True, max_token_context_length=100000),
    )

    result = agent.run(input_data={"input": PROMPT2, "files": None})

    output_content = result.output.get("content")
    logger.info("RESULT")
    logger.info(output_content)
