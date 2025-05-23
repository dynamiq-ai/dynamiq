import io

from dynamiq.connections import E2B, Firecrawl, ScaleSerp
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = (
    "An assistant with access to the internet and Python coding tools. "
    "Capable of performing preliminary research, scraping data, writing code, and executing it."
    "The agent breaks tasks into smaller parts and solves them sequentially. "
    "It also ensures the quality of the code, refines it, and rechecks all results before final delivery."
)

PROMPT = (
    "Using the input file, for each company, find the company's website, scrape it, and locate the LinkedIn page. "
    "Deliver the final answer in a table with the following columns: Company Name, Company Website, LinkedIn Page. "
    "Ensure this is done for all items in the input file."
)


def read_file_as_bytesio(file_path: str, filename: str = None, description: str = None) -> io.BytesIO:
    """
    Reads the content of a file and returns it as a BytesIO object with custom attributes for filename and description.

    Args:
        file_path (str): The path to the file.
        filename (str, optional): Custom filename for the BytesIO object.
        description (str, optional): Custom description for the BytesIO object.

    Returns:
        io.BytesIO: The file content in a BytesIO object with custom attributes.
    """
    with open(file_path, "rb") as f:
        file_content = f.read()

    file_io = io.BytesIO(file_content)

    file_io.name = filename if filename else "uploaded_file"
    file_io.description = description if description else "No description provided"

    return file_io


def create_agent():
    """
    Create and configure the agent with necessary tools.

    Returns:
        ReActAgent: A configured Dynamiq ReActAgent ready to run.
    """
    tool_code = E2BInterpreterTool(connection=E2B())
    tool_scrape = FirecrawlTool(connection=Firecrawl())
    tool_search = ScaleSerpTool(connection=ScaleSerp())
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0.001)

    agent_software = ReActAgent(
        name="React Agent",
        llm=llm,
        tools=[tool_code, tool_scrape, tool_search],
        role=AGENT_ROLE,
        max_loops=30,
        inference_mode=InferenceMode.XML,
        behaviour_on_max_loops=Behavior.RETURN,
    )

    return agent_software


def run_workflow(prompt: str, files_to_upload: list[io.BytesIO]) -> tuple[str, dict]:
    """
    Main function to set up and run the workflow, handling any exceptions that may occur.

    Args:
        prompt (str): Question/task for the agent to accomplish.
        files_to_upload (List[io.BytesIO]): A list of BytesIO objects representing files to upload.

    Returns:
        tuple[str, dict]: The content generated by the agent and intermediate steps.
    """
    try:
        agent = create_agent()

        result = agent.run(
            input_data={"input": prompt, "files": files_to_upload},
        )

        return result.output.get("content", "")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return "", {}
