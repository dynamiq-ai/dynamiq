"""Example demonstrating the use of BedrockAgentCoreSandbox with an Agent.

This example shows how to configure an agent with an AWS Bedrock AgentCore sandbox for:
- Remote file storage in an isolated AgentCore microVM session
- File read/write/search/list operations via sandbox tools
- Shell command execution via SandboxShellTool

It is the AgentCore counterpart of ``agent_e2b_sandbox.py`` -- same agent, same tools, with
``BedrockAgentCoreSandbox`` as the backend instead of ``E2BSandbox``.

Prerequisites:
- AWS credentials resolvable by boto3 (env vars, ``AWS_DEFAULT_PROFILE``, or IMDS) with
  permission to call ``bedrock-agentcore``
- A region set via ``AWS_DEFAULT_REGION`` (AgentCore is not available in every region)

Differences from the E2B backend, all inherited from the Code Interpreter API:
- Paths are workspace-relative, not absolute (``base_path="workspace"``, not ``/home/user``)
- The built-in ``aws.codeinterpreter.v1`` interpreter has no network egress, so the agent
  cannot fetch URLs or ``pip install``. Use a custom interpreter with ``networkMode: PUBLIC``
  for that.
- No public ports or preview URLs are exposed, so ``get_sandbox_info(port=...)`` reports an
  error instead of a host.
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
    Set up and return an agent configured with a Bedrock AgentCore sandbox.

    Args:
        region: AWS region. If None, will use the AWS_DEFAULT_REGION environment variable.

    Returns:
        tuple: (Agent, BedrockAgentCoreSandbox) - Configured agent and sandbox backend.
    """
    aws_connection = AWSConnection(region=region) if region else AWSConnection()
    if not aws_connection.region:
        raise ValueError("AWS region is required. Set AWS_DEFAULT_REGION environment variable or pass it directly.")

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.2)

    # Create AgentCore sandbox backend
    agentcore_sandbox = BedrockAgentCoreSandbox(
        connection=aws_connection,
        timeout=3600,  # 1 hour session lifetime, fixed at session start
        base_path="workspace",  # workspace-relative, unlike E2B's absolute /home/user
    )

    sandbox_config = SandboxConfig(
        enabled=True,
        backend=agentcore_sandbox,
    )

    agent = Agent(
        name="BedrockAgentCoreSandboxAgent",
        id="bedrock-agentcore-sandbox-agent",
        llm=llm,
        sandbox=sandbox_config,
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        max_loops=10,
    )

    return agent, agentcore_sandbox


def run_workflow(
    agent: Agent = None,
    sandbox: BedrockAgentCoreSandbox = None,
    input_prompt: str = EXAMPLE_QUERY,
) -> str:
    """
    Run the agent workflow with the Bedrock AgentCore sandbox.

    Args:
        agent: The agent to use. If None, will create a new one.
        sandbox: The AgentCore sandbox backend. If None, will be created with agent.
        input_prompt: The input prompt for the agent.

    Returns:
        str: The agent's output content.
    """
    if agent is None:
        agent, sandbox = setup_agent()

    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": input_prompt},
            config=RunnableConfig(callbacks=[tracing]),
        )

        output = result.output[agent.id]["output"]["content"]
        print(f"\n=== Agent Output ===\n{output}")

        return output

    except Exception as e:
        logger.exception(f"An error occurred: {e}")
        return f"Error: {e}"

    finally:
        if sandbox:
            try:
                # kill=True stops the AgentCore session; kill=False would leave it running
                # until its timeout so it can be reattached later via session_id.
                sandbox.close(kill=True)
                print("\nAgentCore sandbox closed successfully.")
            except Exception as e:
                logger.warning(f"Failed to close AgentCore sandbox: {e}")


if __name__ == "__main__":
    print("=== Bedrock AgentCore Sandbox Agent Example ===\n")

    if not os.getenv("AWS_DEFAULT_REGION"):
        print("Warning: AWS_DEFAULT_REGION environment variable not set.")
        print("Please set it before running this example:")
        print("  export AWS_DEFAULT_REGION='us-east-1'\n")
        exit(1)

    result = run_workflow()
