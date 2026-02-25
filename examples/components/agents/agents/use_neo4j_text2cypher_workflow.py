import os

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections import Neo4j
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools import CypherExecutor
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

INGEST_ROLE = (
    "You can ingest data into Neo4j and then answer questions. "
    "Always use XML actions: <action>cypher_executor</action> with JSON in <action_input>. "
    "Do not use self-closing tags like <cypher_executor/>. "
    "If schema is unknown, call cypher_executor with mode=introspect. "
    "For new raw text, extract entities/relations, write them via cypher_executor with allow_writes=true, "
    "then query with allow_writes=false and provide the final answer. "
    "Use only mode=execute or mode=introspect; do not use mode=r/w. "
    "Use routing='r' for read queries when supported. "
    "Avoid comma-separated MATCH patterns (cartesian products). Prefer MATCH ... WITH ... MATCH or "
    "single MATCH with relationship patterns. "
    "For write-then-read flows, you may send a list of queries in one call."
)


def build_ingest_agent() -> Agent:
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0, max_tokens=2048)
    connection = Neo4j()

    cypher_executor = CypherExecutor(connection=connection, name="cypher_executor")

    return Agent(
        name="neo4j_ingest_agent",
        description="Writes facts to Neo4j and answers follow-up questions.",
        role=INGEST_ROLE,
        llm=llm,
        tools=[cypher_executor],
        inference_mode=InferenceMode.FUNCTION_CALLING,
        max_loops=8,
    )


def run_workflow(callbacks: list | None = None):
    agent = build_ingest_agent()
    wf = Workflow(flow=Flow(nodes=[agent]))

    tracing = callbacks or [TracingCallbackHandler()]
    result = wf.run(
        input_data={
            "input": "Add that Charlie works at Dynamiq as a Product Manager, then list everyone who works at Dynamiq."
        },
        config=RunnableConfig(callbacks=tracing),
    )
    agent_output = result.output.get(agent.id, {}).get("output", {})
    content = agent_output.get("content")
    if content is None:
        raise RuntimeError("Agent returned no content. Check logs for XML/tool parsing errors.")
    return content


def run_with_ui_tracing(
    base_url: str = os.environ.get("DYNAMIQ_TRACE_BASE_URL", "https://collector.sandbox.getdynamiq.ai"),
    access_key: str | None = os.environ.get("DYNAMIQ_TRACE_ACCESS_KEY"),
    **handler_kwargs,
):
    tracing = DynamiqTracingCallbackHandler(base_url=base_url, access_key=access_key, **handler_kwargs)
    content = run_workflow(callbacks=[tracing])
    return content, tracing


if __name__ == "__main__":
    output, tracing_handler = run_with_ui_tracing()
    logger.info("=== NEO4J INGEST AGENT OUTPUT ===")
    logger.info(output)
