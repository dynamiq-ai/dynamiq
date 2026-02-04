import json
import logging
import os
import uuid
from pathlib import Path

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections import Exa as ExaConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.tools.python_code_executor import PythonCodeExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file import InMemorySandbox, SandboxConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

LOGGER = logging.getLogger(__name__)
AGENT_ROLE = """
You are a Senior Research Strategist preparing a publication-ready sustainability brief.
ts.

Operating protocol:
1. Begin by drafting a plan (sections, data you plan to gather) and log it in your thoughts.
2. Use Exa search iteratively.
Each cited insight must include the source URL, e.g., [AWS Circularity 2025](https://...).
3. Maintain the evolving report in `web_research_report.md` using read_file()/write_file(). Treat it like a real doc:
   - Keep a table of KPIs
   - Track a version header with timestamp (ISO format) and total sources referenced.
4. Whenever data lends itself to visualization,
generate a quick chart (PNG) with Python and save it (e.g., `sustainability_gap.png`).
5. Final deliverable: attach the markdown report + any generated assets, and summarize key findings and next steps.
6. Always ensure your Python code defines `run(...)` and returns structured data.
"""

PROMPT = """
Research "Top sustainability initiatives launched by major cloud providers in 2025".
Your deliverable is a concise markdown report saved as `web_research_report.md`
with sections: Executive Summary, Provider Highlights (AWS, Azure, Google Cloud),
Open Questions, and References.
"""


def _create_agent() -> Agent:
    llm = setup_llm(model_provider="gpt", model_name="o4-mini", temperature=0.4)
    sandbox_backend = InMemorySandbox()
    sandbox_config = SandboxConfig(enabled=True, backend=sandbox_backend, agent_file_write_enabled=True)

    exa_tool = ExaTool(connection=ExaConnection(), name="exa-search")
    code_tool = PythonCodeExecutor(name="code-executor")

    return Agent(
        name="Cloud Sustainability Researcher",
        llm=llm,
        tools=[exa_tool, code_tool],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        max_loops=15,
        sandbox=sandbox_config,
    )


def _build_workflow() -> Workflow:
    agent = _create_agent()
    return Workflow(id="agent-web-researcher-workflow", flow=Flow(nodes=[agent]))


def _resolve_trace_runs(callbacks: list[TracingCallbackHandler] | None) -> dict:
    for callback in callbacks or []:
        if isinstance(callback, TracingCallbackHandler):
            return getattr(callback, "runs", {})
    return {}


def run_workflow(
    prompt: str,
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
        input_data={"input": prompt, "user_id": user_id, "session_id": session_id},
        config=RunnableConfig(callbacks=callbacks),
    )

    agent_id = workflow.flow.nodes[0].id
    agent_output = result.output.get(agent_id, {}).get("output", {})
    LOGGER.info("Web researcher output preview: %s", agent_output.get("content", "")[:200])

    trace_runs = _resolve_trace_runs(callbacks)
    if trace_runs:
        json.dumps({"runs": [run.to_dict() for run in trace_runs.values()]}, cls=JsonWorkflowEncoder)
    return agent_output, trace_runs


def run_workflow_with_ui_tracing(
    prompt: str,
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
        callbacks=[tracing],
        user_id=user_id,
        session_id=session_id,
    )
    return output, traces, tracing


def main():
    agent_output, _, _ = run_workflow_with_ui_tracing(prompt=PROMPT)

    content = agent_output.get("content")
    files = agent_output.get("files", [])

    logger.info("---------------------------------Result-------------------------------------")
    logger.info(content)

    if files:
        output_dir = Path("./agent_outputs")
        output_dir.mkdir(exist_ok=True)
        for idx, file_obj in enumerate(files, start=1):
            file_name = getattr(file_obj, "name", f"file_{idx}")
            target_path = output_dir / file_name
            with open(target_path, "wb") as f:
                f.write(file_obj.read())
                file_obj.seek(0)
            logger.info("Saved generated file to %s", target_path)
    else:
        logger.info("No files returned by the agent.")


if __name__ == "__main__":
    main()
