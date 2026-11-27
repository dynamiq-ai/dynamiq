"""Example demonstrating the use of BoxdSandbox with an Agent.

This example shows how to configure an agent with a boxd sandbox for:
- Remote file storage on an isolated boxd machine (a full Ubuntu VM)
- File read/write/search/list operations via sandbox tools
- Shell command execution, including pip installs, via SandboxShellTool

Prerequisites:
- boxd API key (set via BOXD_API_KEY environment variable or pass directly), from https://boxd.sh?ref=dynamiq
- Anthropic API key (set via ANTHROPIC_API_KEY)
- Install the boxd extra: pip install "dynamiq[boxd]"
"""

import os

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Boxd as BoxdConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.boxd import BoxdSandbox
from dynamiq.utils.logger import logger
from examples.llm_setup import MODEL_NAME_CLAUDE, setup_llm

AGENT_ROLE = """
You are a helpful assistant that can execute shell commands in a remote sandbox environment.
 Use write tool to write to the sandbox
"""

EXAMPLE_QUERY = """
1. Create a file called 'hello.py' with a Python script that prints "Hello from boxd!" in bold with the rich package
2. Install rich with pip
3. Run the script: python hello.py
4. Show the output
"""


def setup_agent(boxd_api_key: str = None) -> tuple[Agent, BoxdSandbox]:
    """
    Set up and return an agent configured with a boxd sandbox.

    Args:
        boxd_api_key: boxd API key. If None, will use BOXD_API_KEY environment variable.

    Returns:
        tuple: (Agent, BoxdSandbox) - Configured agent and sandbox backend.
    """
    api_key = boxd_api_key or os.getenv("BOXD_API_KEY")
    if not api_key:
        raise ValueError("boxd API key is required. Set BOXD_API_KEY environment variable or pass it directly.")

    llm = setup_llm(model_provider="claude", model_name=MODEL_NAME_CLAUDE, temperature=0.2)

    boxd_connection = BoxdConnection(api_key=api_key)

    # Create the boxd sandbox backend. The machine is created on first use.
    boxd_sandbox = BoxdSandbox(
        connection=boxd_connection,
        base_path="/home/boxd/workspace",
    )

    sandbox_config = SandboxConfig(
        enabled=True,
        backend=boxd_sandbox,
    )

    agent = Agent(
        name="BoxdSandboxAgent",
        id="boxd-sandbox-agent",
        llm=llm,
        sandbox=sandbox_config,
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        max_loops=10,
    )

    return agent, boxd_sandbox


def run_workflow(
    agent: Agent = None,
    sandbox: BoxdSandbox = None,
    input_prompt: str = EXAMPLE_QUERY,
) -> str:
    """
    Run the agent workflow with a boxd sandbox.

    Args:
        agent: The agent to use. If None, will create a new one.
        sandbox: The boxd sandbox backend. If None, will be created with agent.
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
                print("\nboxd machine deleted.")
            except Exception as e:
                logger.warning(f"Failed to delete the boxd machine: {e}")


if __name__ == "__main__":
    print("=== boxd Sandbox Agent Example ===\n")

    if not os.getenv("BOXD_API_KEY"):
        print("Warning: BOXD_API_KEY environment variable not set.")
        print("Please set it before running this example:")
        print("  export BOXD_API_KEY='your-api-key'\n")
        exit(1)

    result = run_workflow()
