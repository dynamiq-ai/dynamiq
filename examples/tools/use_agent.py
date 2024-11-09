from dynamiq.connections import Http as HttpConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType
from dynamiq.nodes.tools.python import Python
from examples.llm_setup import setup_llm

PYTHON_TOOL_CODE = """
def run(inputs):
    import requests
    response = requests.get("https://example.com")
    return {"content": response.text}
"""


if __name__ == "__main__":
    llm = setup_llm()

    # Create the PythonTool for web requests
    web_request_tool = Python(
        name="WebRequestTool",
        description="Makes a GET request to example.com and returns the response text",
        code=PYTHON_TOOL_CODE,
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
        role="is to help user with various tasks, goal is to provide best of possible answers to user queries",  # noqa: E501
    )

    result = agent.run(
        input_data={"input": "Show me content of example.com"}, config=None
    )

    print("Agent's response:")

    print(result.output)

    result = agent.run(
        input_data={"input": "Show me some random fact about cat"}, config=None
    )

    print("Agent's response:")

    print(result.output)
