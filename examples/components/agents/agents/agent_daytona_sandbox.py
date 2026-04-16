"""Example demonstrating the use of DaytonaSandbox with an Agent.

This example shows how to configure an agent with Daytona sandbox for:
- Remote file storage in an isolated Daytona environment
- File read/write/search/list operations via sandbox tools
- Shell command execution via SandboxShellTool

Prerequisites:
- Daytona API key (set via DAYTONA_API_KEY environment variable or pass directly)
- Install daytona-desktop: pip install daytona-desktop
"""

import os

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Daytona as DaytonaConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.daytona import DaytonaSandbox

# from tests.unit.sandboxes.test_daytona_sandbox_provider import daytona_sandbox
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are a helpful assistant that can execute shell commands in a remote sandbox environment.
 Use write tool to write to the sandox
"""

EXAMPLE_QUERY = """
1. Create a file called 'hello.py' with a Python script that prints "Hello from Daytona!"
2. Edit it with edit action, by changing it to 'Hello from Daytona!!!'
3. Run the script: python hello.py
4. Show the output
"""


def setup_agent(daytona_api_key: str = None) -> tuple[Agent, DaytonaSandbox]:
    """
    Set up and return an agent configured with Daytona sandbox.

    Args:
        daytona_api_key: Daytona API key. If None, will use DAYTONA_API_KEY environment variable.

    Returns:
        tuple: (Agent, DaytonaSandbox) - Configured agent and sandbox backend.
    """
    api_key = daytona_api_key or os.getenv("DAYTONA_API_KEY")
    if not api_key:
        raise ValueError("Daytona API key is required. Set DAYTONA_API_KEY environment variable or pass it directly.")

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.2)

    daytona_connection = DaytonaConnection(api_key=api_key)

    # Create Daytona sandbox backend
    daytona_sandbox = DaytonaSandbox(
        connection=daytona_connection,
        timeout=3600,  # 1 hour timeout
        base_path="/home/daytona",
    )

    sandbox_config = SandboxConfig(
        enabled=True,
        backend=daytona_sandbox,
    )

    agent = Agent(
        name="DaytonaSandboxAgent",
        id="daytona-sandbox-agent",
        llm=llm,
        sandbox=sandbox_config,
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        max_loops=10,
    )

    return agent, daytona_sandbox


def run_workflow(
    agent: Agent = None,
    sandbox: DaytonaSandbox = None,
    input_prompt: str = EXAMPLE_QUERY,
) -> str:
    """
    Run the agent workflow with Daytona sandbox.

    Args:
        agent: The agent to use. If None, will create a new one.
        sandbox: The Daytona sandbox backend. If None, will be created with agent.
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
                print("\nDaytona sandbox closed successfully.")
            except Exception as e:
                logger.warning(f"Failed to close Daytona sandbox: {e}")


if __name__ == "__main__":
    print("=== Daytona Sandbox Agent Example ===\n")

    if not os.getenv("DAYTONA_API_KEY"):
        print("Warning: DAYTONA_API_KEY environment variable not set.")
        print("Please set it before running this example:")
        print("  export DAYTONA_API_KEY='your-api-key'\n")
        exit(1)

    result = run_workflow()
