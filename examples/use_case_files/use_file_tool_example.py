from pathlib import Path

from dynamiq.connections import E2B
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


CSV_PATH = ".data/sample_regression_data.csv"
TXT_PATH = ".data/company_policies.txt"

txt_bytes = read_file_as_bytes(TXT_PATH)
csv_bytes = read_file_as_bytes(CSV_PATH)

tool_csv = FileReadTool()
python_tool = E2BInterpreterTool(connection=E2B())
files = [(CSV_PATH, "csv file")]
llm = setup_llm()


agent = ReActAgent(
    name="Agent",
    id="Agent",
    llm=llm,
    tools=[python_tool],
)

result = agent.run(
    input_data={"input": "Calculate the mean values for all columns and craft me table of this", "files": files}
)
print(result.output.get("content"))
