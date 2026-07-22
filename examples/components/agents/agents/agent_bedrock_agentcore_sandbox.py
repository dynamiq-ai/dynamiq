"""Example demonstrating the use of BedrockAgentCoreSandbox with an Agent.

This example shows how to configure an agent with an AWS Bedrock AgentCore
Code Interpreter sandbox for:
- Remote file storage in an isolated AWS-managed microVM session
- File read/write operations via sandbox tools
- Shell command execution via SandboxShellTool

Prerequisites:
- AWS credentials with bedrock-agentcore permissions
  (set via AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION
  environment variables, an AWS profile, or pass directly)
"""

import os

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import AWS as AWSConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.bedrock_agentcore import BedrockAgentCoreSandbox
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are a helpful assistant that can execute shell commands in a remote sandbox environment.
 Use write tool to write to the sandbox
"""

EXAMPLE_QUERY = """
1. Create a file called 'hello.py' with a Python script that prints "Hello from AgentCore!"
2. Edit it with edit action, by changing it to 'Hello from AgentCore!!!'
3. Run the script: python hello.py
4. Show the output
"""


def setup_agent(region: str = None) -> tuple[Agent, BedrockAgentCoreSandbox]:
    """
    Set up and return an agent configured with an AgentCore sandbox.

    Args:
        region: AWS region. If None, will use AWS_DEFAULT_REGION environment variable.

    Returns:
        tuple: (Agent, BedrockAgentCoreSandbox) - Configured agent and sandbox backend.
    """
    region = region or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError("AWS region is required. Set AWS_DEFAULT_REGION environment variable or pass it directly.")

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.2)

    aws_connection = AWSConnection(region=region)

    # Create AgentCore sandbox backend
    agentcore_sandbox = BedrockAgentCoreSandbox(
        connection=aws_connection,
        timeout=3600,
    )

    agent = Agent(
        name="AgentCore Sandbox Agent",
        id="agentcore_sandbox_agent",
        llm=llm,
        role=AGENT_ROLE,
        sandbox=SandboxConfig(enabled=True, backend=agentcore_sandbox),
        inference_mode=InferenceMode.XML,
    )

    return agent, agentcore_sandbox


def run_workflow(query: str = EXAMPLE_QUERY) -> str:
    """Run the agent workflow with the given query."""
    agent, sandbox = setup_agent()

    workflow = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = workflow.run(
            input_data={"input": query},
            config=RunnableConfig(callbacks=[TracingCallbackHandler()]),
        )
        output = result.output[agent.id]["output"]["content"]
        logger.info(f"Agent output: {output}")
        return output
    finally:
        sandbox.close(kill=True)


if __name__ == "__main__":
    run_workflow()
