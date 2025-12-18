import os

from dynamiq import Workflow
from dynamiq.callbacks import DynamiqTracingCallbackHandler, TracingCallbackHandler
from dynamiq.connections import Neo4j
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools import Neo4jCypherExecutor, Neo4jGraphWriter, Neo4jSchemaIntrospector, Neo4jText2Cypher
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

INGEST_ROLE = (
    "You can ingest data into Neo4j (graph_writer) and then answer questions. "
    "Always use XML actions: <action>schema_introspector</action>, <action>graph_writer</action>, "
    "<action>text2cypher_with_writes</action>, or <action>cypher_executor</action> with JSON in <action_input>. "
    "For new raw text, extract entities/relations, write them via graph_writer "
    '{"nodes": [{"labels": ["Label"], "identity_key": "id", "properties": {"id": "x", "name": "..."}}], '
    '"relationships": [{"type": "REL", "start_label": "Label", "start_identity_key": "id", "start_identity": "x", '
    '"end_label": "LabelB", "end_identity_key": "id", "end_identity": "y", "properties": {}}]}. '
    "Then generate Cypher with text2cypher_with_writes (allow_writes=true) and execute with cypher_executor."
)


def build_ingest_agent() -> Agent:
    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0, max_tokens=2048)
    connection = Neo4j()

    schema = Neo4jSchemaIntrospector(connection=connection, name="schema_introspector")
    graph_writer = Neo4jGraphWriter(connection=connection, name="graph_writer")
    text2cypher = Neo4jText2Cypher(llm=llm, name="text2cypher_with_writes")
    cypher_executor = Neo4jCypherExecutor(connection=connection, name="cypher_executor")

    return Agent(
        name="neo4j_ingest_agent",
        description="Writes facts to Neo4j and answers follow-up questions.",
        role=INGEST_ROLE,
        llm=llm,
        tools=[schema, graph_writer, text2cypher, cypher_executor],
        inference_mode=InferenceMode.XML,
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
    content = result.output[agent.id]["output"]["content"]
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
