from pathlib import Path

from dynamiq.connections import E2B
from dynamiq.nodes.agents import FileDataModel
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from examples.llm_setup import setup_llm
from examples.tools.file_reader import FileReadTool


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


# Define file paths
CSV_PATH = "/Users/oleksiibabych/Projects/Product_D/dynamiq/.data/sample_regression_data.csv"

# Read files as bytes
csv_bytes = read_file_as_bytes(CSV_PATH)

# Create FileDataModel instances
file_csv_model = FileDataModel(file_data=csv_bytes, description="CSV file with regression data")

# Initialize tools
tool_csv = FileReadTool(files=[file_csv_model])
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
result = agent.run(
    input_data={"input": "Calculate the mean values for all columns in the CSV", "files": [file_csv_model]}
)

# Print the result content
print(result.output.get("content"))
