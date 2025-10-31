import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.embedders import OpenAIDocumentEmbedder
from dynamiq.nodes.types import InferenceMode
from dynamiq.nodes.writers import WeaviateDocumentWriter
from dynamiq.nodes.writers.writer import VectorStoreWriter
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import WeaviateVectorStore
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm


def setup_react_agent_rag() -> Agent:
    document_embedder = OpenAIDocumentEmbedder(model="text-embedding-3-small")
    vector_store_writer = WeaviateDocumentWriter(
        vector_store=WeaviateVectorStore(index_name="TravelData", create_if_not_exist=True)
    )

    tool_vector_store_writer = VectorStoreWriter(
        name="Travel Data Vector Store Writer",
        document_embedder=document_embedder,
        document_writer=vector_store_writer,
        is_optimized_for_agents=True,
    )

    llm = setup_llm(model_name="gpt-5-mini")

    agent = Agent(
        name="React Agent",
        llm=llm,
        tools=[tool_vector_store_writer],
        inference_mode=InferenceMode.XML,
        role="AI assistant with deep knowledge about various travel destinations. Your goal is to provide well explained answers and write them to the vector store using the Travel Data Vector Store Writer tool",  # noqa: E501
        max_loops=7,
    )
    return agent


AGENT_QUERY = """What are top 10 places to visit in Rome, Italy?"""


def run_workflow(agent: Agent = setup_react_agent_rag(), input_prompt: str = AGENT_QUERY) -> tuple[str, dict]:
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
