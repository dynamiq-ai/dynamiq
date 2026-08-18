"""Example demonstrating the use of BedrockAgentCoreRuntimeSandbox with an Agent.

This is the AWS counterpart of ``agent_e2b_sandbox.py`` and deliberately mirrors it, so the
two backends can be run against the same task and compared. It shows how to configure an
agent with an AWS Bedrock AgentCore Runtime sandbox for:
- Remote file storage in an isolated AWS-managed microVM session
- File read/write/search/list operations via sandbox tools
- Shell command execution via SandboxShellTool

Prerequisites:
- AWS credentials with ``bedrock-agentcore:InvokeAgentRuntimeCommand`` and
  ``bedrock-agentcore:StopRuntimeSession`` (env vars, an AWS profile, SSO or IMDS)
- ``AWS_DEFAULT_REGION`` (or a region on the resolved profile)
- ``BEDROCK_AGENTCORE_RUNTIME_ARN`` — the ARN of a provisioned agent runtime. Unlike E2B,
  the runtime is a deployment-time resource (an image plus its configuration, closer to an
  E2B *template* than to a sandbox) and is never created by this example. Its container
  must serve the AgentCore HTTP contract, or every invocation returns a 500.
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
from dynamiq.sandboxes.bedrock_agentcore_runtime import BedrockAgentCoreRuntimeSandbox
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are a helpful assistant that can execute shell commands in a remote sandbox environment.
 Use write tool to write to the sandbox
"""

EXAMPLE_QUERY = """
1. Create a file called 'hello.py' with a Python script that prints "Hello from AgentCore!"
2. Edit it with edit action, by changing it to 'Hello from AgentCore!!!'
3. Run the script: python3 hello.py
4. Show the output
"""


def setup_agent(
    agent_runtime_arn: str | None = None,
    region: str | None = None,
    session_seed: str | None = None,
) -> tuple[Agent, BedrockAgentCoreRuntimeSandbox]:
    """
    Set up and return an agent configured with an AgentCore Runtime sandbox.

    Args:
        agent_runtime_arn: ARN of a provisioned agent runtime. If None, will use the
            BEDROCK_AGENTCORE_RUNTIME_ARN environment variable.
        region: AWS region. If None, will use the AWS_DEFAULT_REGION environment variable.
        session_seed: Seed for deterministic session-ID derivation, typically a conversation
            ID. Reusing a seed reconnects to the same session and the same workspace.

    Returns:
        tuple: (Agent, BedrockAgentCoreRuntimeSandbox) - Configured agent and sandbox backend.
    """
    runtime_arn = agent_runtime_arn or os.getenv("BEDROCK_AGENTCORE_RUNTIME_ARN")
    if not runtime_arn:
        raise ValueError("An agent runtime ARN is required. Set BEDROCK_AGENTCORE_RUNTIME_ARN or pass it directly.")

    region = region or os.getenv("AWS_DEFAULT_REGION")
    if not region:
        raise ValueError("AWS region is required. Set AWS_DEFAULT_REGION environment variable or pass it directly.")

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.2)

    aws_connection = AWSConnection(region=region)

    # Create the AgentCore Runtime sandbox backend. `s3_relay_bucket` is optional: without
    # it, files above `inline_transfer_max_bytes` (40 KB) cannot be transferred, because the
    # 64 KB command limit leaves no room to inline them.
    agentcore_sandbox = BedrockAgentCoreRuntimeSandbox(
        connection=aws_connection,
        agent_runtime_arn=runtime_arn,
        session_seed=session_seed,
        timeout=300,  # per-command, AgentCore allows 1..3600
        base_path="/mnt/workspace",
        s3_relay_bucket=os.getenv("BEDROCK_AGENTCORE_RELAY_BUCKET"),
    )

    sandbox_config = SandboxConfig(
        enabled=True,
        backend=agentcore_sandbox,
    )

    agent = Agent(
        name="AgentCoreRuntimeSandboxAgent",
        id="agentcore-runtime-sandbox-agent",
        llm=llm,
        sandbox=sandbox_config,
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        max_loops=10,
    )

    return agent, agentcore_sandbox


def run_workflow(
    agent: Agent | None = None,
    sandbox: BedrockAgentCoreRuntimeSandbox | None = None,
    input_prompt: str = EXAMPLE_QUERY,
) -> str:
    """
    Run the agent workflow with the AgentCore Runtime sandbox.

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
                # kill=True calls StopRuntimeSession, which flushes session storage durably
                # but blocks for several seconds. Pass kill=False to leave the session
                # reconnectable; its compute stops on the idle timeout and is not billed.
                sandbox.close(kill=True)
                print("\nAgentCore Runtime session stopped successfully.")
            except Exception as e:
                logger.warning(f"Failed to close AgentCore Runtime sandbox: {e}")


if __name__ == "__main__":
    print("=== AgentCore Runtime Sandbox Agent Example ===\n")

    missing = [var for var in ("BEDROCK_AGENTCORE_RUNTIME_ARN", "AWS_DEFAULT_REGION") if not os.getenv(var)]
    if missing:
        print(f"Warning: {', '.join(missing)} environment variable(s) not set.")
        print("Please set them before running this example:")
        print("  export AWS_DEFAULT_REGION='us-east-1'")
        print("  export BEDROCK_AGENTCORE_RUNTIME_ARN='arn:aws:bedrock-agentcore:...:runtime/...'\n")
        exit(1)

    result = run_workflow()
