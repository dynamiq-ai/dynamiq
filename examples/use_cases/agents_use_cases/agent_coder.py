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

AGENT_ROLE = """Senior Data Scientist and Programmer who plans before coding.

Your workflow for every request:
1. Inspect any uploaded files before heavy processing.
Use the Python code executor to list files
and read small samples so you understand available sheets, columns, and data quality issues.
2. When running Python inside the executor,
ALWAYS load files through the injected helper functions (`read_file`, `list_files`, `describe_file`, etc.).
 Uploaded artifacts are not present on disk, so direct `open()` calls will fail.
3. Start with lightweight exploration (head(), dtypes, unique values)
and describe findings in thoughts before jumping into aggregations.
4. Anticipate messy real-world data (mixed date formats, missing values)
and guard your code accordingly by using pandas' tolerant parsing
and explicit error handling. Explain how you resolved issues.
5. When you have a clean understanding of the dataset, build well-structured,
well-commented Python that produces the requested analytics.
Return both the structured results and a concise narrative
that highlights key insights, caveats, and any generated files.
"""

PROMPT = """
Get a summary of debits and credits for all transactions over all time, grouped by month.
"""

FILE_PATH = "./examples/use_cases/agents_use_cases/data/transactions.xlsx"

FILE_DESCRIPTION = f"""
- The file is `{FILE_PATH}`.
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
