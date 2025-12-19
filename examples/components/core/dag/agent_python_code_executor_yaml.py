import io
import json
import os
import uuid
from pathlib import Path

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections.managers import get_connection_manager
from dynamiq.nodes.tools.python_code_executor import PythonCodeExecutor
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger

PROMPT = """
Summarize the dataset with:
- row count and column names
- basic stats for numeric columns
- save a CSV named `summary_stats.csv` with the numeric column stats
"""

ROOT_DIR = Path(__file__).resolve().parents[4]
DATA_PATH = ROOT_DIR / "examples/use_cases/agents_use_cases/data/product_feedback.csv"
DATA_DESCRIPTION = "Sample product feedback CSV data."


def _read_file_as_bytesio(file_path: Path, description: str, content_type: str = "text/csv") -> io.BytesIO:
    with open(file_path, "rb") as f:
        payload = f.read()
    file_obj = io.BytesIO(payload)
    file_obj.name = file_path.name
    file_obj.description = description.strip()
    file_obj.content_type = content_type
    return file_obj


def _build_workflow() -> Workflow:
    dag_yaml_file_path = Path(__file__).with_name("agent_python_code_executor.yaml")
    with get_connection_manager() as cm:
        wf = Workflow.from_yaml_file(file_path=str(dag_yaml_file_path), connection_manager=cm, init_components=True)
    _share_file_store_with_python_executor(wf)
    return wf


def _share_file_store_with_python_executor(workflow: Workflow) -> None:
    """
    Ensure PythonCodeExecutor tools share the agent's file store backend.
    """
    for node in workflow.flow.nodes:
        agent_file_store = getattr(node, "file_store_backend", None)
        if not agent_file_store:
            continue
        for tool in getattr(node, "tools", []):
            if isinstance(tool, PythonCodeExecutor):
                tool.file_store = agent_file_store


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
    logger.info("Python executor agent output preview: %s", agent_output.get("content", "")[:200])

    trace_runs = _resolve_trace_runs(callbacks)
    if trace_runs:
        json.dumps({"runs": [run.to_dict() for run in trace_runs.values()]}, cls=JsonWorkflowEncoder)
    return agent_output, trace_runs


def run_workflow_with_ui_tracing(
    prompt: str,
    files_to_upload: list[io.BytesIO],
    base_url: str = os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://collector.sandbox.getdynamiq.ai"),
    access_key: str | None = os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY"),
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


def main() -> None:
    data_file = _read_file_as_bytesio(DATA_PATH, DATA_DESCRIPTION)
    agent_output, _, _ = run_workflow_with_ui_tracing(prompt=PROMPT, files_to_upload=[data_file])

    content = agent_output.get("content")
    files = agent_output.get("files", [])

    logger.info("---------------------------------Result-------------------------------------")
    logger.info(content)

    if files:
        os.makedirs("./agent_outputs", exist_ok=True)
        for idx, f in enumerate(files, start=1):
            file_name = getattr(f, "name", f"file_{idx}")
            path = Path("./agent_outputs") / file_name
            with open(path, "wb") as output_file:
                output_file.write(f.read())
                f.seek(0)
            logger.info("Saved generated file to %s", path)
    else:
        logger.info("No files returned by the agent.")


if __name__ == "__main__":
    main()
