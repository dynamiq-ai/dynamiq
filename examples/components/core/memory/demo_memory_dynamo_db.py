from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Iterable

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections import AWS
from dynamiq.flows import Flow
from dynamiq.memory import Memory
from dynamiq.memory.backends.dynamo_db import DynamoDB
from dynamiq.nodes.agents import Agent
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

LOGGER = logging.getLogger(__name__)

WORKFLOW_ID = "dynamodb-memory-workflow"
DEFAULT_HELLO_PROMPT = "Hello, I'm Alex and I love reading science fiction books."
DEFAULT_FACT_PROMPT = "What do you remember about me?"


def setup_agent() -> Agent:
    """Initialise the chat agent with DynamoDB-backed memory."""
    llm = setup_llm()

    try:
        aws_connection = AWS()
        session = aws_connection.get_boto3_session()
        credentials = session.get_credentials()
        frozen = credentials.get_frozen_credentials() if credentials else None
        LOGGER.debug(
            "AWS session initialised (access key prefix=%s, token=%s)",
            (frozen.access_key[:4] + "***") if frozen else "<missing>",
            "present" if frozen and frozen.token else "missing",
        )
        dynamo_backend = DynamoDB(
            connection=aws_connection,
            table_name=os.getenv("DYNAMODB_TABLE", "dynamiq-chat-memory"),
            create_if_not_exist=bool(os.getenv("DYNAMODB_CREATE_TABLE", "true").lower() in {"1", "true", "yes"}),
        )
        LOGGER.info("DynamoDB backend initialised and connection verified.")
    except Exception as error:
        LOGGER.error("Failed to initialise DynamoDB backend: %s", error)
        raise

    memory = Memory(backend=dynamo_backend, message_limit=50)
    return Agent(
        name="ChatAgent",
        llm=llm,
        role="Helpful assistant that remembers the conversation using DynamoDB.",
        id="chat-memory-agent",
        memory=memory,
    )


def build_workflow() -> Workflow:
    """Create a workflow with a single DynamoDB-enabled agent node."""
    agent = setup_agent()
    return Workflow(id=WORKFLOW_ID, flow=Flow(nodes=[agent]))


def _resolve_trace_runs(callbacks: Iterable[object] | None) -> dict:
    for callback in callbacks or []:
        if isinstance(callback, TracingCallbackHandler):
            return getattr(callback, "runs", {})
    return {}


def _get_agent_output(workflow: Workflow, wf_result) -> dict:
    """Extract the single-agent output dictionary from the workflow result."""
    if not workflow.flow.nodes:
        return {}

    agent_id = workflow.flow.nodes[0].id
    agent_result = wf_result.output.get(agent_id, {})
    return agent_result.get("output", {})


def run_workflow(
    callbacks: list[TracingCallbackHandler] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    input_text: str | None = None,
):
    """Run the DynamoDB memory workflow once and return the agent output with trace runs."""

    if input_text is None:
        raise ValueError("input_text must be provided for the workflow run.")

    if callbacks is None:
        callbacks = [TracingCallbackHandler()]

    user_id = user_id or f"user-{uuid.uuid4().hex[:6]}"
    session_id = session_id or f"session-{uuid.uuid4().hex[:8]}"

    LOGGER.info("Running workflow '%s' for user=%s session=%s", WORKFLOW_ID, user_id, session_id)

    workflow = build_workflow()

    result = workflow.run(
        input_data={
            "input": input_text,
            "user_id": user_id,
            "session_id": session_id,
        },
        config=RunnableConfig(callbacks=callbacks),
    )

    agent_output = _get_agent_output(workflow, result)
    LOGGER.info("Workflow output: %s", agent_output.get("content"))

    trace_runs = _resolve_trace_runs(callbacks)
    if trace_runs:
        json.dumps({"runs": [run.to_dict() for run in trace_runs.values()]}, cls=JsonWorkflowEncoder)

    return agent_output, trace_runs


def run_workflow_with_ui_tracing(
    input_text: str,
    base_url: str = os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://collector.sandbox.getdynamiq.ai"),
    access_key: str | None = os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY"),
    handler_kwargs: dict | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
):
    """Execute the workflow once with Dynamiq UI tracing enabled."""

    tracing = DynamiqTracingCallbackHandler(
        base_url=base_url,
        access_key=access_key,
        **(handler_kwargs or {}),
    )
    output, traces = run_workflow(
        callbacks=[tracing],
        user_id=user_id,
        session_id=session_id,
        input_text=input_text,
    )
    return output, traces, tracing


if __name__ == "__main__":
    shared_user = "def"
    shared_session = "def"

    first_output, _, _ = run_workflow_with_ui_tracing(
        input_text=DEFAULT_HELLO_PROMPT,
        user_id=shared_user,
        session_id=shared_session,
    )

    second_output, _, _ = run_workflow_with_ui_tracing(
        input_text=DEFAULT_FACT_PROMPT,
        user_id=shared_user,
        session_id=shared_session,
    )

    LOGGER.info("=== DYNAMODB MEMORY WORKFLOW OUTPUT ===")
    LOGGER.info("First turn: %s", first_output.get("content"))
    LOGGER.info("Second turn: %s", second_output.get("content"))
