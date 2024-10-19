import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B
from dynamiq.connections import Http as HttpConnection
from dynamiq.connections import ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.embedders import OpenAITextEmbedder
from dynamiq.nodes.retrievers import WeaviateDocumentRetriever
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.tools.retriever import RetrievalTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.vector import WeaviateVectorStore
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

# Constants
AGENT_ROLE = (
    "teacher for children, goal is to craft a well-structured and simple final answer"
    "with a lot of emojis to empathize with the children."
)
QUERY = "Who won the Euro 2024?"
QUERY_FOR_CODING_SIMPLE = "Add the first 10 numbers and tell if the result is prime"
QUERY_FOR_CODING_COMPLEX = (
    "generate a sample for linear regression analysis and print the results, just textual representations"
)
QUERY_FOR_PYTHON_TOOL = "Show me content of example.com"
QUERY_FOR_HTTP_TOOL = "Show me some random fact about cat"


def setup_react_agent() -> ReActAgent:
    """
    Set up and return a ReAct agent with specified LLM and tools.

    Returns:
        ReActAgent: Configured ReAct agent.
    """
    llm = setup_llm()

    # Create tools
    tool_search = ScaleSerpTool(connection=ScaleSerp())

    return ReActAgent(
        name="ReAct Agent - Children Teacher",
        id="react",
        llm=llm,
        tools=[tool_search],
        role=AGENT_ROLE,
    )


def setup_react_agent_coding() -> ReActAgent:
    """
    Set up and return a ReAct agent with specified LLM and tools.

    Returns:
        ReActAgent: Configured ReAct agent.
    """
    llm = setup_llm()

    # Create tools
    tool = E2BInterpreterTool(
        connection=E2B(),
        default_packages="",
    )

    return ReActAgent(
        name="ReAct Agent - Children Teacher",
        id="react",
        llm=llm,
        tools=[tool],
        role=AGENT_ROLE,
    )


def setup_react_agent_http_python() -> ReActAgent:
    """
    Set up and return a ReAct agent with specified LLM and tools.

    Returns:
        ReActAgent: Configured ReAct agent.
    """
    llm = setup_llm()

    # Create the PythonTool for web requests
    web_request_tool = Python(
        name="WebRequestTool",
        description="Makes a GET request to example.com and returns the response text",
        code="""
    def run(input_data):
        import requests
        response = requests.get("https://example.com")
        return {"content": response.text}
    """,
    )

    connection = HttpConnection(
        method="GET",
        url="https://catfact.ninja/fact",
    )

    # Create an instance of HttpApiCall
    api_call = HttpApiCall(
        connection=connection,
        success_codes=[200, 201],
        timeout=60,
        response_type=ResponseType.JSON,
        params={"limit": 10},
        name="CatFactApi",
        description="Gets a random cat fact from the CatFact API",
    )

    # Create the agent with the PythonTool
    agent = ReActAgent(
        name="AI Agent",
        llm=llm,
        tools=[web_request_tool, api_call],
        role="is to help user with various tasks",
    )
    return agent


def setup_react_agent_rag() -> ReActAgent:
    text_embedder = OpenAITextEmbedder(model="text-embedding-3-small")
    retriever_dubai = WeaviateDocumentRetriever(top_k=2, vector_store=WeaviateVectorStore(index_name="Dubai"))
    retriever_customs = WeaviateDocumentRetriever(
        top_k=2, vector_store=WeaviateVectorStore(index_name="Dubai_customs_policies_and_notices")
    )

    tool_retrieval_sports = RetrievalTool(
        name="visit DUBAI data",
        text_embedder=text_embedder,
        document_retriever=retriever_dubai,
    )
    tool_retrieval_rta = RetrievalTool(
        name="visit DUBAI data",
        text_embedder=text_embedder,
        document_retriever=retriever_customs,
    )

    llm = setup_llm()

    # Create the agent with tools and configuration
    agent = ReActAgent(
        name="React Agent",
        llm=llm,
        tools=[tool_retrieval_rta, tool_retrieval_sports],
        role="AI assistant with knowledge about Dubai city, goal is provide well explained final answers, you can tune user search to be more accurate with RAG search",  # noqa: E501
        max_loops=7,
    )
    return agent


def run_workflow(
    agent: ReActAgent = setup_react_agent_http_python(), input_prompt: str = QUERY_FOR_HTTP_TOOL
) -> tuple[str, dict]:
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
        # Verify that traces can be serialized to JSON
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
