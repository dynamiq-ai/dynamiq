import io
import json
from pathlib import Path

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

INPUT_PROMPT = "Calculate the mean values for all columns in the CSV"
FILE_PATH = ".data/sample_regression_data.csv"


def read_file_as_bytesio(file_path: str, filename: str = None, description: str = None) -> io.BytesIO:
    """
    Reads the content of a file and returns it as a BytesIO object with custom attributes for filename and description.

    Args:
        file_path (str): The path to the file.
        filename (str, optional): Custom filename for the BytesIO object.
        description (str, optional): Custom description for the BytesIO object.

    Returns:
        io.BytesIO: The file content in a BytesIO object with custom attributes.

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

    file_io = io.BytesIO(file_content)

    file_io.name = filename if filename else file_path_obj.name
    file_io.description = description if description else ""

    return file_io


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


csv_bytes_io = read_file_as_bytesio(
    FILE_PATH, filename="custom_regression_data.csv", description="Custom CSV file with regression data"
)

python_tool = E2BInterpreterTool(connection=E2B())

llm = setup_llm()

agent = ReActAgent(
    name="Agent",
    id="Agent",
    llm=llm,
    tools=[python_tool],
)

result = agent.run(input_data={"input": INPUT_PROMPT, "files": [csv_bytes_io]})

print(result.output.get("content"))

output, traces = run_workflow(
    agent=agent,
    input_prompt=INPUT_PROMPT
)
print("Agent Output:", output)
