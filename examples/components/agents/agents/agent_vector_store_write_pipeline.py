import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.components.splitters.document import DocumentSplitBy
from dynamiq.connections import Firecrawl as FirecrawlConnection
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.tools import FirecrawlTool, PreprocessTool, TavilyTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.nodes.writers import WeaviateDocumentWriter
from dynamiq.nodes.writers.writer import VectorStoreWriter
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import WeaviateVectorStore
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are an AI assistant that assists the users with their questions.
If the user asks you to ingest a document, \
    you should process your response with the PreprocessTool \
    and write the results to the vector store in a structured format.
"""


def setup_agent() -> Agent:
    document_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
    writer = WeaviateDocumentWriter(
        vector_store=WeaviateVectorStore(index_name="ArticlesData", create_if_not_exist=True)
    )

    tool_writer = VectorStoreWriter(
        name="Articles Data Vector Store Writer",
        document_embedder=document_embedder,
        document_writer=writer,
        is_optimized_for_agents=True,
    )

    preprocess_tool = PreprocessTool(
        split_by=DocumentSplitBy.SENTENCE,
        split_length=10,
        split_overlap=0,
    )

    tavily_tool = TavilyTool(
        connection=TavilyConnection(),
        include_raw_content=True,
    )

    scraping_tool = FirecrawlTool(
        connection=FirecrawlConnection(),
        include_raw_content=True,
    )

    llm = setup_llm(model_provider="gemini", model_name="gemini-2.5-flash-preview-05-20")

    agent = Agent(
        name="React Agent",
        llm=llm,
        tools=[tool_writer, preprocess_tool, tavily_tool, scraping_tool],
        inference_mode=InferenceMode.XML,
        parallel_tool_calls_enabled=True,
        role=AGENT_ROLE,
        max_loops=50,
    )
    return agent


AGENT_QUERY = """Ingest Paul Graham's "Life is Short" essay into the vector store."""


def run_workflow(agent: Agent = setup_agent(), input_prompt: str = AGENT_QUERY) -> tuple[str, dict]:
    """
    Execute a workflow using the ReAct agent to process a predefined query.

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.

    Raises:
        Exception: Captures and prints any errors during workflow execution.
    """
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": input_prompt},
            config=RunnableConfig(callbacks=[tracing]),
        )

        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    output, traces = run_workflow()
    print("Agent Output:", output)
