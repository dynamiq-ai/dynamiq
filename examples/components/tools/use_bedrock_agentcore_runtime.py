"""Example demonstrating BedrockAgentCoreRuntimeInterpreterTool with a ReAct agent.

The companion example ``agents/agents/agent_bedrock_agentcore_runtime_sandbox.py`` wires the
backend in as an agent *sandbox* (filesystem + shell tools). This one wires it in as a code
*interpreter tool*, which is the other integration path and mirrors ``use_react_with_coding.py``.
The first two tasks are identical to that E2B example so the two backends can be compared
directly.

The third task exercises what AgentCore does that E2B does not: session storage survives an
explicit stop, so reusing a ``session_seed`` reconnects a *new* tool instance to the same
workspace. The seed is typically a conversation ID; the session ID is derived from it rather
than stored, so reconnection survives a process restart with no mapping table.

Prerequisites:
- AWS credentials with ``bedrock-agentcore:InvokeAgentRuntimeCommand`` and
  ``bedrock-agentcore:StopRuntimeSession`` (env vars, an AWS profile, SSO or IMDS)
- ``AWS_DEFAULT_REGION`` (or a region on the resolved profile)
- ``BEDROCK_AGENTCORE_RUNTIME_ARN`` — the ARN of a provisioned agent runtime
"""

import os

from dynamiq.connections import AWS as AWSConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.bedrock_agentcore_runtime_sandbox import BedrockAgentCoreRuntimeInterpreterTool
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm

AGENT_ROLE = """
Senior Data Scientist and Programmer with ability to write a well written python code inside
<code> tags and you have access to python tool.
Goal is to provide well explained answer to requests
"""  # noqa: E501

SESSION_SEED = "agentcore-runtime-example-conversation"


def build_agent(tool: BedrockAgentCoreRuntimeInterpreterTool) -> Agent:
    """Build a ReAct agent backed by the given interpreter tool."""
    llm = setup_llm(model_provider="claude", model_name="claude-sonnet-4-5", temperature=0.1)
    return Agent(name="React Agent", inference_mode=InferenceMode.XML, llm=llm, tools=[tool], role=AGENT_ROLE)


def make_tool(session_seed: str, runtime_arn: str, region: str) -> BedrockAgentCoreRuntimeInterpreterTool:
    """Create an interpreter tool bound to the session derived from ``session_seed``."""
    return BedrockAgentCoreRuntimeInterpreterTool(
        connection=AWSConnection(region=region),
        agent_runtime_arn=runtime_arn,
        session_seed=session_seed,
        s3_relay_bucket=os.getenv("BEDROCK_AGENTCORE_RELAY_BUCKET"),
        # Hold one session open across the agent's tool calls instead of creating and
        # stopping one per execution. close() is a no-op unless this is set.
        persistent_sandbox=True,
    )


if __name__ == "__main__":
    missing = [var for var in ("BEDROCK_AGENTCORE_RUNTIME_ARN", "AWS_DEFAULT_REGION") if not os.getenv(var)]
    if missing:
        print(f"Warning: {', '.join(missing)} environment variable(s) not set.")
        print("Please set them before running this example:")
        print("  export AWS_DEFAULT_REGION='us-east-1'")
        print("  export BEDROCK_AGENTCORE_RUNTIME_ARN='arn:aws:bedrock-agentcore:...:runtime/...'\n")
        exit(1)

    arn = os.environ["BEDROCK_AGENTCORE_RUNTIME_ARN"]
    aws_region = os.environ["AWS_DEFAULT_REGION"]

    tool = make_tool(SESSION_SEED, arn, aws_region)
    agent = build_agent(tool)

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

    # Write a file, then stop the session outright. Session storage is durable across a stop.
    result_write = agent.run(
        input_data={
            "input": (
                "Write a file at /mnt/workspace/notes.txt containing exactly the text "
                "'survived the stop', then confirm it exists."
            ),
        },
        config=None,
    )
    tool.close()

    # A brand-new tool with the same seed derives the same session ID and finds the file.
    reconnected_tool = make_tool(SESSION_SEED, arn, aws_region)
    reconnected_agent = build_agent(reconnected_tool)
    result_reconnect = reconnected_agent.run(
        input_data={
            "input": "Read /mnt/workspace/notes.txt and tell me exactly what it contains.",
        },
        config=None,
    )
    reconnected_tool.close()

    print("\nResult with prime check:")
    print(result_prime.output.get("content"))
    print("\nResult with Fibonacci sequence:")
    print(result_fibonacci.output.get("content"))
    print("\nResult with workspace write:")
    print(result_write.output.get("content"))
    print("\nResult after reconnecting to the same session (expect 'survived the stop'):")
    print(result_reconnect.output.get("content"))
