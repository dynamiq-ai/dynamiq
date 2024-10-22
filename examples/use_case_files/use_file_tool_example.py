import json
from pathlib import Path

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B
from dynamiq.flows import Flow
from dynamiq.nodes.agents import FileDataModel
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

INPUT_PROMPT = "Calculate the mean values for all columns in the CSV"


def read_file_as_bytes(file_path: str) -> bytes:
    """
    Reads the content of a file and returns it as bytes.

    Args:
        file_path (str): The path to the file.

    Returns:
        bytes: The file content in bytes.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If an I/O error occurs while reading the file.
    """
    file_path_obj = Path(file_path).resolve()
    if not file_path_obj.exists():
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    if not file_path_obj.is_file():
        raise OSError(f"The path {file_path} is not a valid file.")
    with file_path_obj.open("rb") as file:
        file_content = file.read()

    return file_content

def run_workflow(
    agent: ReActAgent,
    input_prompt: str
) -> tuple[str, dict]:
    """
    Execute a workflow using the ReAct agent to process a predefined query.

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.

    Raises:
        Exception: Captures and prints any errors during workflow execution.
    """
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": input_prompt},
            config=RunnableConfig(callbacks=[tracing]),
        )
        # Verify that traces can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}

# Define file paths
CSV_PATH = ".data/sample_regression_data.csv"

# Read files as bytes
csv_bytes = read_file_as_bytes(CSV_PATH)

# Create FileDataModel instances
file_csv_model = FileDataModel(file=csv_bytes, description="CSV file with regression data")

# Initialize tools
python_tool = E2BInterpreterTool(connection=E2B())

# Set up LLM
llm = setup_llm()

# Initialize agent with tools
agent = ReActAgent(
    name="Agent",
    id="Agent",
    llm=llm,
    tools=[python_tool],
)

# Agent execution with input data and files
result = agent.run(input_data={"input": INPUT_PROMPT, "files": [file_csv_model]})

# Print the result content
print(result.output.get("content"))

output, traces = run_workflow(
    agent=agent,
    input_prompt=INPUT_PROMPT
)
print("Agent Output:", output)

