import base64
import io
import json
import logging
import os
import uuid

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic as AnthropicLLM
from dynamiq.nodes.tools.python_code_executor import PythonCodeExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger

LOGGER = logging.getLogger(__name__)

AGENT_ROLE = """
Senior Data Scientist and Programmer with the ability to write well-structured Python code.
You have access to the Python Code Executor tool that can read/write files from the shared project workspace.
Generally, you follow these rules:
    - ALWAYS FORMAT YOUR RESPONSE IN MARKDOWN.
    - Use double quotes for property names.
    - Ensure the code is correct and runnable; reiterate if it does not work.
    - Describe which files you touched and include relevant snippets when needed.
    - Every code snippet you send to the executor MUST define a run(...) function as the entrypoint.
    - Use helper functions such as read_file(), write_file(), and list_files() provided by the code executor.
    - Always return structured dictionaries/lists from run();
    rely on the tool's stdout capture for any console-style output.
    - Include a 'markdown_report' entry in your return dict summarizing key insights and recommendations.
"""

PROMPT = """
Get required statistics from the file and compute them on the provided CSV file using the Python code executor tool.
Each execution must provide a `run(...)` function entrypoint. Return a concise markdown report with your findings.
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
    Create and configure the agent with the Python code executor tool.

    Returns:
        Agent: A configured Dynamiq agent ready to run.
    """
    llm = AnthropicLLM(
        name="claude",
        model="claude-sonnet-4-20250514",
        temperature=1,
        max_tokens=32000,
        thinking_enabled=True,
        budget_tokens=4000,
    )

    file_store_backend = InMemoryFileStore()
    file_store_config = FileStoreConfig(enabled=True, backend=file_store_backend, agent_file_write_enabled=True)

    tool = PythonCodeExecutor(name="code-executor", file_store=file_store_backend)

    agent_software = Agent(
        name="Agent",
        llm=llm,
        tools=[tool],
        role=AGENT_ROLE,
        max_loops=10,
        inference_mode=InferenceMode.XML,
        file_store=file_store_config,
    )

    return agent_software


def _build_workflow() -> Workflow:
    agent = create_agent()
    return Workflow(id="agent-coder-workflow", flow=Flow(nodes=[agent]))


def _resolve_trace_runs(callbacks: list[TracingCallbackHandler] | None) -> dict:
    for callback in callbacks or []:
        if isinstance(callback, TracingCallbackHandler):
            return getattr(callback, "runs", {})
    return {}


def run_workflow(
    prompt: str,
    files_to_upload: list[io.BytesIO],
    callbacks: list[TracingCallbackHandler] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
):
    """
    Execute the agent inside a Dynamiq workflow so tracing/UI can capture the run.
    Returns the agent output dictionary and trace runs (if any).
    """
    if callbacks is None:
        callbacks = [TracingCallbackHandler()]

    workflow = _build_workflow()
    user_id = user_id or f"user-{uuid.uuid4().hex[:6]}"
    session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"

    result = workflow.run(
        input_data={
            "input": prompt,
            "files": files_to_upload,
            "user_id": user_id,
            "session_id": session_id,
        },
        config=RunnableConfig(callbacks=callbacks),
    )

    agent_id = workflow.flow.nodes[0].id
    agent_output = result.output.get(agent_id, {}).get("output", {})
    LOGGER.info("Agent coder output preview: %s", agent_output.get("content", "")[:200])

    trace_runs = _resolve_trace_runs(callbacks)
    if trace_runs:
        json.dumps({"runs": [run.to_dict() for run in trace_runs.values()]}, cls=JsonWorkflowEncoder)

    return agent_output, trace_runs


def run_workflow_with_ui_tracing(
    prompt: str,
    files_to_upload: list[io.BytesIO],
    base_url: str = os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://collector.sandbox.getdynamiq.ai"),
    access_key: str | None = os.environ.get(
        "DYNAMIQ_TRACE_ACCESS_KEY",
    ),
    handler_kwargs: dict | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
):
    tracing = DynamiqTracingCallbackHandler(
        base_url=base_url,
        access_key=access_key,
        **(handler_kwargs or {}),
    )
    output, traces = run_workflow(
        prompt=prompt,
        files_to_upload=files_to_upload,
        callbacks=[tracing],
        user_id=user_id,
        session_id=session_id,
    )
    return output, traces, tracing


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


if __name__ == "__main__":
    csv_file_io = read_file_as_bytesio(
        file_path=FILE_PATH, filename=FILE_PATH.split("/")[-1], description=FILE_DESCRIPTION
    )

    agent_output, _, _ = run_workflow_with_ui_tracing(prompt=PROMPT, files_to_upload=[csv_file_io])
    content = agent_output.get("content")
    files = agent_output.get("files")

    logger.info("---------------------------------Result-------------------------------------")
    logger.info(content)
    save_agent_files(files)
