from dynamiq.connections import E2B
from dynamiq.nodes.agents.base import FileDataModel
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """
 Senior Data Scientist and Programmer with ability to write a well written
 python code and you have access to python tool
 You have access to web to search for best solutions for a problem.
 Generally, you follow these rules:
    - ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN
    - Use double quotes for property names
    - Make code correct and runnable test code and reiterate if does not work
"""

PROMPT = """
 Write code in Python that fits linear regression model between
 number of bathrooms and bedrooms) and price of a house from the data.
 Count loss. Return this code. Set a seed that results would be reproducable.
 Provide exect result of MSE
"""

# Please use your own csv file path
FILE_PATH = "data.csv"

# Please provide your own csv file description
FILE_DESCRIPTION = """
- It's `data.csv` file
- The CSV file is using , as the delimiter
- It has the following columns (examples included):
    - bedrooms: number of badrooms
    - bathrooms: number of bathrooms
    - price: price of a house
"""


def create_agent():
    """
    Create and configure the agent with necessary tools.

    Returns:
        Workflow: A configured Dynamiq workflow ready to run.
    """
    tool = E2BInterpreterTool(connection=E2B())

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0.001)
    agent_software = ReActAgent(
        name="React Agent",
        llm=llm,
        tools=[tool],
        role=AGENT_ROLE,
        max_loops=10,
        inference_mode=InferenceMode.XML,
    )

    return agent_software


def run_workflow(prompt: str, files_to_upload: list[str], files_description: list[str]) -> tuple[str, dict]:
    """
    Main function to set up and run the workflow, handling any exceptions that may occur.

    This function loads environment variables, creates the workflow, runs it with the
    specified input, and returns the output. Any exceptions are caught and printed.

    Args:
        prompt (str): Question/task for agent to accomplish.
        files_to_upload (List[str]): A list of file paths that have to be uploaded.
        files_description (List[str]): Description of files uploaded
    """
    if len(files_description) != len(files_to_upload):
        raise ValueError("Number of file paths and file descriptions doesn't match")

    try:
        agent = create_agent()
        files = []

        for file_path, file_description in zip(files_to_upload, files_description):
            with open(file_path, "rb") as file:
                file_data = file.read()
                file_model = FileDataModel(file_data=file_data, description=file_description)
                files.append(file_model)

        result = agent.run(
            input_data={"input": prompt, "files": files},
        )
        return result["content"], result.get("intermediate_steps", {})
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    output, steps = run_workflow(prompt=PROMPT, files_to_upload=[FILE_PATH], files_description=[FILE_DESCRIPTION])

    logger.info("---------------------------------Result-------------------------------------")
    logger.info(output)
