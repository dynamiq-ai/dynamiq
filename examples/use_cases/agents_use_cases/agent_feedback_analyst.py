import io
import json
import logging
import os
import uuid
from pathlib import Path

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections import TogetherAI as TogetherAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.togetherai import TogetherAI
from dynamiq.nodes.tools.python_code_executor import PythonCodeExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import InMemorySandbox, SandboxConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger

LOGGER = logging.getLogger(__name__)

AGENT_ROLE = """
You are a Senior Product Analytics Engineer. Use the Python code executor to inspect structured feedback datasets,
surface key trends, and propose actionable follow-ups. Always:
- Summarize the methodology and findings in markdown.
"""

PROMPT = """
Plot a chart that describes the data
"""

DATA_DIR = Path(__file__).resolve().parent / "data"
FEEDBACK_PATH = DATA_DIR / "product_feedback.csv"
FEEDBACK_DESCRIPTION = """Some csv data"""


def _read_file_as_bytesio(file_path: Path, description: str, content_type: str = "text/csv") -> io.BytesIO:
    with open(file_path, "rb") as f:
        payload = f.read()
    file_obj = io.BytesIO(payload)
    file_obj.name = file_path.name
    file_obj.description = description.strip()
    file_obj.content_type = content_type
    return file_obj


def _create_agent() -> Agent:
    TOGETHER_API_KEY = os.environ.get(
        "TOGETHER_API_KEY",
    )

    model_name = "zai-org/GLM-4.5-Air-FP8"  # "openai/gpt-oss-120b"
    connection = TogetherAIConnection(api_key=TOGETHER_API_KEY)
    llm = TogetherAI(
        model=model_name,
        connection=connection,
        max_tokens=4096,
        temperature=0.1,
    )
    sandbox_backend = InMemorySandbox()
    sandbox_config = SandboxConfig(enabled=True, backend=sandbox_backend, agent_file_write_enabled=True)

    code_tool = PythonCodeExecutor(name="code-executor")

    return Agent(
        name="Feedback Analyst",
        llm=llm,
        tools=[code_tool],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        max_loops=8,
        sandbox=sandbox_config,
    )


def _build_workflow() -> Workflow:
    agent = _create_agent()
    return Workflow(id="agent-feedback-analyst-workflow", flow=Flow(nodes=[agent]))


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
    LOGGER.info("Feedback analyst output preview: %s", agent_output.get("content", "")[:200])

    trace_runs = _resolve_trace_runs(callbacks)
    if trace_runs:
        json.dumps({"runs": [run.to_dict() for run in trace_runs.values()]}, cls=JsonWorkflowEncoder)
    return agent_output, trace_runs


def run_workflow_with_ui_tracing(
    prompt: str,
    files_to_upload: list[io.BytesIO],
    base_url: str = os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://ui.sandbox.getdynamiq.ai"),
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


def main():
    feedback_file = _read_file_as_bytesio(FEEDBACK_PATH, FEEDBACK_DESCRIPTION)
    agent_output, _, _ = run_workflow_with_ui_tracing(prompt=PROMPT, files_to_upload=[feedback_file])

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
