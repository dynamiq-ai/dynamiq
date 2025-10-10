from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Iterable

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import PostgreSQL as PostgreSQLConnection
from dynamiq.flows import Flow
from dynamiq.memory import Memory
from dynamiq.memory.backends.postgresql import PostgreSQL as PostgreSQLMemoryBackend
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI as OpenAILLM
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder

LOGGER = logging.getLogger(__name__)

WORKFLOW_ID = "postgresql-memory-workflow"
DEFAULT_GREETING_PROMPT = "Hi, I'm Alex and Lisbon is my favourite city."
DEFAULT_RECALL_PROMPT = "Do you remember what city I like the most?"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
POSTGRES_TABLE_NAME = os.getenv("POSTGRESQL_TABLE", "dynamiq_chat_memory")
MEMORY_MESSAGE_LIMIT = int(os.getenv("POSTGRESQL_MESSAGE_LIMIT", "50"))


def setup_agent() -> Agent:
    """Initialise the chat agent with PostgreSQL-backed memory."""
    try:
        openai_connection = OpenAIConnection()
        llm = OpenAILLM(connection=openai_connection, model=OPENAI_MODEL)
        LOGGER.info("OpenAI LLM (%s) initialised.", llm.name)
    except Exception as error:
        LOGGER.error("Failed to initialise OpenAI LLM: %s", error)
        raise

    try:
        pg_connection = PostgreSQLConnection()
        memory_backend = PostgreSQLMemoryBackend(
            connection=pg_connection,
            table_name=POSTGRES_TABLE_NAME,
            create_if_not_exist=True,
        )
        LOGGER.info(
            "PostgreSQL backend initialised for table '%s'.",
            POSTGRES_TABLE_NAME,
        )
    except Exception as error:
        LOGGER.error("Failed to initialise PostgreSQL backend: %s", error)
        raise

    memory = Memory(backend=memory_backend, message_limit=MEMORY_MESSAGE_LIMIT)
    return Agent(
        name="ChatAgentPostgres",
        llm=llm,
        role="Helpful assistant that remembers the conversation using PostgreSQL.",
        id="chat-memory-agent",
        memory=memory,
    )


def build_workflow() -> Workflow:
    """Create a workflow with a single PostgreSQL-enabled agent node."""
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
    """Run the PostgreSQL memory workflow once and return the agent output with traces."""

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
        input_text=DEFAULT_GREETING_PROMPT,
        user_id=shared_user,
        session_id=shared_session,
    )

    second_output, _, _ = run_workflow_with_ui_tracing(
        input_text=DEFAULT_RECALL_PROMPT,
        user_id=shared_user,
        session_id=shared_session,
    )

    LOGGER.info("=== POSTGRESQL MEMORY WORKFLOW OUTPUT ===")
    LOGGER.info("First turn: %s", first_output.get("content"))
    LOGGER.info("Second turn: %s", second_output.get("content"))
