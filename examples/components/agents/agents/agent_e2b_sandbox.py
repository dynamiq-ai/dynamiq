"""Example demonstrating the use of E2BSandbox with an Agent.

This example shows how to configure an agent with E2B sandbox for:
- Remote file storage in an isolated E2B environment
- File read/write/search/list operations via sandbox tools
- Shell command execution via SandboxShellTool

Prerequisites:
- E2B API key (set via E2B_API_KEY environment variable or pass directly)
- Install e2b-desktop: pip install e2b-desktop
"""

import os

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import E2B as E2BConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.e2b import E2BSandbox
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are a helpful assistant that can execute shell commands in a remote sandbox environment.
 Use write tool to write to the sandox
"""

EXAMPLE_QUERY = """
1. Create a file called 'hello.py' with a Python script that prints "Hello from E2B!"
2. Run the script: python hello.py
3. Show the output
"""


def setup_agent(e2b_api_key: str = None) -> tuple[Agent, E2BSandbox]:
    """
    Set up and return an agent configured with E2B sandbox.

    Args:
        e2b_api_key: E2B API key. If None, will use E2B_API_KEY environment variable.

    Returns:
        tuple: (Agent, E2BSandbox) - Configured agent and sandbox backend.
    """
    api_key = e2b_api_key or os.getenv("E2B_API_KEY")
    if not api_key:
        raise ValueError("E2B API key is required. Set E2B_API_KEY environment variable or pass it directly.")

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.2)

    e2b_connection = E2BConnection(api_key=api_key)

    # Create E2B sandbox backend
    e2b_sandbox = E2BSandbox(
        connection=e2b_connection,
        timeout=3600,  # 1 hour timeout
        base_path="/home/user",
    )

    sandbox_config = SandboxConfig(
        enabled=True,
        backend=e2b_sandbox,
    )

    agent = Agent(
        name="E2BSandboxAgent",
        id="e2b-sandbox-agent",
        llm=llm,
        sandbox=sandbox_config,
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        max_loops=10,
    )

    return agent, e2b_sandbox


def run_workflow(
    agent: Agent = None,
    sandbox: E2BSandbox = None,
    input_prompt: str = EXAMPLE_QUERY,
) -> str:
    """
    Run the agent workflow with E2B sandbox.

    Args:
        agent: The agent to use. If None, will create a new one.
        sandbox: The E2B sandbox backend. If None, will be created with agent.
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
                sandbox.close(kill=True)
                print("\nE2B sandbox closed successfully.")
            except Exception as e:
                logger.warning(f"Failed to close E2B sandbox: {e}")


if __name__ == "__main__":
    print("=== E2B Sandbox Agent Example ===\n")

    if not os.getenv("E2B_API_KEY"):
        print("Warning: E2B_API_KEY environment variable not set.")
        print("Please set it before running this example:")
        print("  export E2B_API_KEY='your-api-key'\n")
        exit(1)

    result = run_workflow()
