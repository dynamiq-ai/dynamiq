import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Http as HttpConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from examples.llm_setup import setup_llm

# Constants
AGENT_ROLE = "Helpful assistant with the goal of providing useful information and answering questions."
INPUT_QUESTION = "What is the capital of France?"
QUERY_FOR_PYTHON_AND_HTTP_TOOL = "Show me the content of example.com and Tell me a random fact about cats."
PYTHON_TOOL_CODE = """
def run(inputs):
    import requests
    response = requests.get("https://example.com")
    return {"content": response.text}
"""


def run_simple_workflow() -> tuple[str, dict]:
    """
    Execute a simple workflow using an OpenAI agent to process a predefined question.

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.

    Raises:
        Exception: Propagates any exceptions raised during workflow execution.
    """
    llm = setup_llm()

    agent = Agent(
        name="Simple Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent-simple",
        verbose=True,
    )
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": INPUT_QUESTION},
            config=RunnableConfig(callbacks=[tracing]),
        )

        # Ensure trace logs can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs

    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


def run_react_workflow() -> tuple[str, dict]:
    """
    Execute a workflow using a ReAct agent with Python and HTTP tools.

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.
    """
    llm = setup_llm()

    # Define a Python tool to fetch content from example.com
    web_request_tool = Python(
        name="WebRequestTool",
        description="Makes a GET request to example.com and returns the response text",
        code=PYTHON_TOOL_CODE,
    )

    # Set up an HTTP connection to retrieve a random cat fact
    connection = HttpConnection(
        method="GET",
        url="https://catfact.ninja/fact",
    )
    api_call = HttpApiCall(
        connection=connection,
        success_codes=[200, 201],
        timeout=60,
        response_type=ResponseType.JSON,
        params={"limit": 10},
        name="CatFactApi",
        description="Gets a random cat fact from the CatFact API",
    )

    # Create the agent with both the Python tool and the HTTP API call
    agent = Agent(
        name="AI Agent",
        llm=llm,
        tools=[web_request_tool, api_call],
        role="is to help user with various tasks",
    )

    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": QUERY_FOR_PYTHON_AND_HTTP_TOOL},
            config=RunnableConfig(callbacks=[tracing]),
        )

        # Ensure trace logs can be serialized to JSON
        json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        return result.output[agent.id]["output"]["content"], tracing.runs

    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    # result, trace = run_simple_workflow()
    result, trace = run_react_workflow()
