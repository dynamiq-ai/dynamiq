import base64
import io
import os

from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic as AnthropicLLM
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.file import InMemorySandbox, SandboxConfig
from dynamiq.utils.logger import logger

AGENT_ROLE = """
Senior Data Scientist and Programmer with the ability to write well-structured Python code.
You have access to Python tools to analyze data and generate statistics from uploaded files.
Generally, you follow these rules:
    - ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN.
    - Use double quotes for property names.
    - Ensure the code is correct and runnable; reiterate if it does not work.
    - When files are provided, use the Python tool to analyze them and generate reports.
"""

PROMPT = """
Analyze the provided CSV file and compute comprehensive statistics using the Python tool.
Generate a detailed markdown report with statistical analysis including
 mean, standard deviation, min, max, and count for all numeric columns.
"""

FILE_PATH = "./examples/use_cases/agents_use_cases/data/house_prices.csv"

FILE_DESCRIPTION = f"""
- The file is `{FILE_PATH}`.
- The CSV file uses a comma (`,`) as the delimiter.
- It contains the following columns (examples included):
    - bedrooms: number of bedrooms
    - bathrooms: number of bathrooms
    - price: price of a house
"""


statistics_requirements = '''
import csv
import io
import statistics

def run(input_data):
    """
    Analyze CSV file and compute statistics, then return results in a file.
    """
    files = input_data.get('files', [])

    if not files:
        return {"error": "No files provided"}

    # Process the first CSV file
    csv_file = files[0]
    csv_file.seek(0)
    content = csv_file.read().decode('utf-8')
    csv_file.seek(0)

    # Read CSV data
    csv_reader = csv.reader(io.StringIO(content))
    rows = list(csv_reader)

    if not rows:
        return {"error": "Empty CSV file"}

    headers = rows[0]
    data_rows = rows[1:]

    # Find numeric columns
    numeric_columns = []
    numeric_data = {}

    for i, header in enumerate(headers):
        try:
            # Try to convert first data value to float
            if data_rows:
                float(data_rows[0][i])
                numeric_columns.append(header)
                # Extract all numeric values for this column
                values = []
                for row in data_rows:
                    try:
                        values.append(float(row[i]))
                    except (ValueError, IndexError):
                        pass
                numeric_data[header] = values
        except (ValueError, IndexError):
            pass

    # Compute statistics for each numeric column
    stats_results = {}
    for column, values in numeric_data.items():
        if values:
            stats_results[column] = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values) if len(values) > 1 else 0,
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }

    # Create result content
    result_lines = ["CSV Statistics Report", "=" * 50, ""]
    result_lines.append(f"File: {getattr(csv_file, 'name', 'unknown')}")
    result_lines.append(f"Total rows: {len(rows)}")
    result_lines.append(f"Data rows: {len(data_rows)}")
    result_lines.append(f"Numeric columns: {', '.join(numeric_columns)}")
    result_lines.append("")

    for column, stats in stats_results.items():
        result_lines.append(f"{column}:")
        result_lines.append(f"  Mean: {stats['mean']:.2f}")
        result_lines.append(f"  Std: {stats['std']:.2f}")
        result_lines.append(f"  Min: {stats['min']:.2f}")
        result_lines.append(f"  Max: {stats['max']:.2f}")
        result_lines.append(f"  Count: {stats['count']}")
        result_lines.append("")

    result_content = "\\n".join(result_lines)
    result_file = io.BytesIO(result_content.encode('utf-8'))
    result_file.name = "statistics_report.txt"
    result_file.description = f"Statistics report for {getattr(csv_file, 'name', 'CSV file')}"

    return {
        "content": f"Successfully analyzed CSV file with {len(numeric_columns)} numeric columns",
        "files": [result_file]
    }
'''


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

    file_io.name = filename if filename else "uploaded_file.csv"
    file_io.description = description if description else "No description provided"

    return file_io


def create_agent():
    """
    Create and configure the agent with necessary tools.

    Returns:
        Agent: A configured Dynamiq agent ready to run.
    """
    statistics_tool = Python(
        name="statistics_tool",
        code=statistics_requirements,
        description="Get required statistics from the file.",
    )

    llm = AnthropicLLM(
        name="claude",
        model="claude-sonnet-4-20250514",
        temperature=1,
        max_tokens=32000,
        thinking_enabled=True,
        budget_tokens=4000,
    )

    sandbox_config = SandboxConfig(enabled=True, backend=InMemorySandbox())

    agent_software = Agent(
        name="Agent",
        llm=llm,
        tools=[statistics_tool],
        role=AGENT_ROLE,
        max_loops=10,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        sandbox=sandbox_config,
    )

    return agent_software


def run_workflow(prompt: str, files_to_upload: list[io.BytesIO]) -> tuple[str, dict, dict]:
    """
    Main function to set up and run the workflow, handling any exceptions that may occur.

    Args:
        prompt (str): Question/task for the agent to accomplish.
        files_to_upload (List[io.BytesIO]): A list of BytesIO objects representing files to upload.

    Returns:
        tuple[str, dict, dict]: The content generated by the agent, intermediate steps, and any files generated.
    """
    try:
        agent = create_agent()

        result = agent.run(
            input_data={"input": prompt, "files": files_to_upload},
        )

        content = result.output.get("content")
        files = result.output.get("files", {})

        return content, files
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return "", {}, {}


def save_agent_files(files: list[io.BytesIO] | dict, output_dir: str = "./agent_outputs") -> None:
    """
    Save files generated by agent, handling both BytesIO objects and legacy dictionary format.

    Args:
        files: List of BytesIO objects or dictionary mapping file paths to content (base64 or data URI)
        output_dir: Directory to save files to
    """
    if not files:
        print("No files to save.")
        return

    print("\n---------------------------------Generated Files-------------------------------------")

    # Handle new BytesIO format
    if isinstance(files, list):
        print(f"Agent generated {len(files)} file(s):")
        os.makedirs(output_dir, exist_ok=True)

        for file_bytesio in files:
            try:
                # Get file metadata from BytesIO object
                file_name = getattr(file_bytesio, "name", f"file_{id(file_bytesio)}.bin")
                file_description = getattr(file_bytesio, "description", "Generated file")
                content_type = getattr(file_bytesio, "content_type", "application/octet-stream")

                # Read content from BytesIO
                file_data = file_bytesio.read()
                file_bytesio.seek(0)  # Reset position for potential future reads

                # Determine file extension from content type if needed
                final_file_name = file_name
                if content_type != "application/octet-stream" and "." not in file_name:
                    extension = get_file_extension_from_mime(content_type)
                    if extension:
                        final_file_name = file_name + extension

                output_path = os.path.join(output_dir, final_file_name)

                with open(output_path, "wb") as f:
                    f.write(file_data)

                print(f"  ✓ Saved: {final_file_name} ({len(file_data):,} bytes) ({content_type}) -> {output_path}")
                print(f"    Description: {file_description}")

            except Exception as e:
                print(f"  ✗ Failed to save {getattr(file_bytesio, 'name', 'unknown')}: {e}")

    # Handle legacy dictionary format
    elif isinstance(files, dict):
        print(f"Agent generated {len(files)} file(s):")
        os.makedirs(output_dir, exist_ok=True)

        for file_path, file_content in files.items():
            original_file_name = file_path.split("/")[-1]

            try:
                base64_content, detected_mime_type = extract_base64_content(file_content)

                file_data = base64.b64decode(base64_content)

                final_file_name = original_file_name

                if detected_mime_type and "." not in original_file_name:
                    extension = get_file_extension_from_mime(detected_mime_type)
                    if extension:
                        final_file_name = original_file_name + extension

                output_path = os.path.join(output_dir, final_file_name)

                with open(output_path, "wb") as f:
                    f.write(file_data)

                mime_info = f" ({detected_mime_type})" if detected_mime_type else ""
                print(f"  ✓ Saved: {final_file_name} ({len(file_data):,} bytes){mime_info} -> {output_path}")

            except Exception as e:
                print(f"  ✗ Failed to save {original_file_name}: {e}")
                preview = file_content[:100] + "..." if len(file_content) > 100 else file_content
                print(f"    Content preview: {preview}")

    else:
        print(f"Unsupported file format: {type(files)}")


def extract_base64_content(file_content: str) -> tuple[str, str | None]:
    """
    Extract base64 content from either data URI or plain base64 string.

    Args:
        file_content: Either a data URI (data:mime/type;base64,content) or plain base64

    Returns:
        Tuple of (base64_content, mime_type)
    """
    if file_content.startswith("data:"):
        try:
            header, base64_content = file_content.split(",", 1)
            mime_type = header.split(";")[0].replace("data:", "")
            return base64_content, mime_type
        except ValueError:
            return file_content, None
    else:
        return file_content, None


def get_file_extension_from_mime(mime_type: str) -> str:
    """
    Get appropriate file extension from MIME type.

    Args:
        mime_type: MIME type string

    Returns:
        File extension with dot (e.g., '.png')
    """
    mime_to_ext = {
        "image/png": ".png",
        "image/jpeg": ".jpg",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
        "image/svg+xml": ".svg",
        "application/pdf": ".pdf",
        "text/html": ".html",
        "text/css": ".css",
        "application/javascript": ".js",
        "text/plain": ".txt",
        "text/csv": ".csv",
        "application/json": ".json",
        "text/markdown": ".md",
    }

    return mime_to_ext.get(mime_type, "")


if __name__ == "__main__":
    csv_file_io = read_file_as_bytesio(
        file_path=FILE_PATH, filename=FILE_PATH.split("/")[-1], description=FILE_DESCRIPTION
    )

    output, files = run_workflow(prompt=PROMPT, files_to_upload=[csv_file_io])

    logger.info("---------------------------------Result-------------------------------------")
    logger.info(output)
    save_agent_files(files)
