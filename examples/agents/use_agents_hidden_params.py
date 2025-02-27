from dynamiq.connections import Http as HttpConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

PYTHON_TOOL_CODE = """
import requests

def run(inputs):
    response = requests.get("https://example.com")
    return response.text[:500]  # Return first 500 chars of the response
"""


def setup_react_agent_http_python() -> ReActAgent:
    """
    Set up and return a ReAct agent with specified LLM and tools.

    Returns:
        ReActAgent: Configured ReAct agent.
    """
    llm = setup_llm()

    web_request_tool = Python(
        name="WebRequestTool",
        description="Makes a GET request to example.com and returns the response text",
        code=PYTHON_TOOL_CODE,
    )

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

    original_request = api_call.client.request

    def request_with_logging(*args, **kwargs):
        print(f"Headers being sent: {kwargs.get('headers', {})}")
        return original_request(*args, **kwargs)

    api_call.client.request = request_with_logging

    agent = ReActAgent(
        name="AI Agent",
        llm=llm,
        tools=[web_request_tool, api_call],
        role="is to help user with various tasks",
        inference_mode=InferenceMode.DEFAULT,
    )
    return agent


def run_agent_with_token():
    agent = setup_react_agent_http_python()

    token = "your_auth_token_12345"  # nosec B105

    result = agent.run(
        input_data={
            "input": "Get me a cat fact using the CatFactApi tool",
            "tool_params": {"headers": {"Authorization": f"Bearer {token}", "X-Custom-Header": "CustomValue"}},
        }
    )

    print("Agent Result:")
    print(result.output.get("content"))


if __name__ == "__main__":
    run_agent_with_token()
