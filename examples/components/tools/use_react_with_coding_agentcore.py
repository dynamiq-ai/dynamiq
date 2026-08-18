"""ReAct agent that writes and runs Python in an AWS Bedrock AgentCore sandbox.

The AgentCore counterpart of ``use_react_with_coding.py``: same agent, same role, with
``BedrockAgentCoreInterpreterTool`` in place of ``E2BInterpreterTool``.

Prerequisites:
- AWS credentials resolvable by boto3 (env vars, ``AWS_DEFAULT_PROFILE``, or IMDS) with
  permission to call ``bedrock-agentcore`` and a region set via ``AWS_DEFAULT_REGION``.

Note: the built-in ``aws.codeinterpreter.v1`` interpreter runs with no network egress, so
tasks must be solvable offline -- no web requests and no ``pip install``. Point
``code_interpreter_identifier`` at a custom interpreter created with
``networkMode: PUBLIC`` if the agent needs to reach the internet.
"""

from dynamiq.connections import AWS
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.bedrock_agentcore_sandbox import BedrockAgentCoreInterpreterTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

AGENT_ROLE = """
Senior Data Scientist and Programmer with ability to write a well written python code inside
<code> tags and you have access to python tool.
Goal is to provide well explained answer to requests
"""  # noqa: E501
if __name__ == "__main__":
    # persistent_sandbox=True keeps one AgentCore session across tool calls. Without it
    # (the default) every call starts a fresh session with an empty filesystem, so a file
    # written by one call is gone by the next -- see the file round-trip task below.
    tool = BedrockAgentCoreInterpreterTool(
        connection=AWS(),
        persistent_sandbox=True,
    )

    llm = setup_llm(model_provider="claude", model_name="claude-sonnet-4-5", temperature=0.1)

    # Create the agent with tools and configuration
    agent = Agent(name="React Agent", inference_mode=InferenceMode.XML, llm=llm, tools=[tool], role=AGENT_ROLE)

    try:
        result_prime = agent.run(
            input_data={
                "input": "Add the first 10 numbers and tell if the result is prime, use functions",
            },
            config=None,
        )

        result_fibonacci = agent.run(
            input_data={
                "input": "generate the first 100 numbers in the Fibonacci sequence, and print me each odd number",
            },
            config=None,
        )
        # The E2B version of this example scrapes a website here. The built-in AgentCore
        # interpreter has no egress, so this reads back a file the agent wrote instead.
        # Only works because the tool holds a persistent session.
        result_files = agent.run(
            input_data={
                "input": "write a CSV of 5 cities and their populations to output/cities.csv, then read it back and print it",  # noqa: E501
            },
            config=None,
        )
        result_data_analysis = agent.run(
            input_data={
                "input": "generate a sample for linear regression analysis and print the results, just textual representations",  # noqa: E501
            },
            config=None,
        )
    finally:
        # A persistent session bills until it is stopped or hits its TTL, so always close it.
        tool.close()

    print("\nResult with prime check:")
    print(result_prime.output.get("content"))
    print("\nResult with Fibonacci sequence:")
    print(result_fibonacci.output.get("content"))
    print("\nResult with file round-trip:")
    print(result_files.output.get("content"))
    print("\nResult with data analysis:")
    print(result_data_analysis.output.get("content"))
