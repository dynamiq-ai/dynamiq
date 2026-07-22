"""Example demonstrating the use of CloudflareSandbox with an Agent.

This example shows how to configure an agent with a Cloudflare sandbox for:
- Remote file storage in an isolated Cloudflare container (/workspace)
- File read/write/list operations via sandbox tools
- Shell command execution via SandboxShellTool

Prerequisites:
- A deployed Cloudflare sandbox bridge Worker
  (https://developers.cloudflare.com/sandbox/bridge/):
    npm create cloudflare -- sandbox-bridge --template=cloudflare/sandbox-sdk/bridge/worker
    npx wrangler secret put SANDBOX_API_KEY
    npx wrangler deploy
- CLOUDFLARE_SANDBOX_API_URL set to the deployed Worker URL
- CLOUDFLARE_SANDBOX_API_KEY set to the Worker's SANDBOX_API_KEY secret

For low-latency sandbox startup, set WARM_POOL_TARGET > 0 in the Worker's vars so
new sandboxes are assigned pre-booted containers instead of cold-starting (~1-3s).
Each sandbox runs in its own VM; warm-pool containers are assigned exclusively to
one sandbox id and never reused across sandboxes.
"""

import os

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Cloudflare as CloudflareConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.sandboxes import SandboxConfig
from dynamiq.sandboxes.cloudflare import CloudflareSandbox
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are a helpful assistant that can execute shell commands in a remote sandbox environment.
 Use write tool to write to the sandbox
"""

EXAMPLE_QUERY = """
1. Create a file called 'hello.py' with a Python script that prints "Hello from Cloudflare!"
2. Edit it with edit action, by changing it to 'Hello from Cloudflare!!!'
3. Run the script: python3 hello.py
4. Show the output
"""


def setup_agent(api_url: str = None, api_key: str = None) -> tuple[Agent, CloudflareSandbox]:
    """
    Set up and return an agent configured with a Cloudflare sandbox.

    Args:
        api_url: Deployed bridge Worker URL. Defaults to CLOUDFLARE_SANDBOX_API_URL.
        api_key: Bridge SANDBOX_API_KEY secret. Defaults to CLOUDFLARE_SANDBOX_API_KEY.

    Returns:
        tuple: (Agent, CloudflareSandbox) - Configured agent and sandbox backend.
    """
    url = api_url or os.getenv("CLOUDFLARE_SANDBOX_API_URL")
    key = api_key or os.getenv("CLOUDFLARE_SANDBOX_API_KEY")
    if not url or not key:
        raise ValueError(
            "Cloudflare sandbox bridge URL and API key are required. Set the "
            "CLOUDFLARE_SANDBOX_API_URL and CLOUDFLARE_SANDBOX_API_KEY environment variables."
        )

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.2)

    cloudflare_connection = CloudflareConnection(url=url, api_key=key)

    # Create Cloudflare sandbox backend (all files live under /workspace)
    cloudflare_sandbox = CloudflareSandbox(
        connection=cloudflare_connection,
        base_path="/workspace",
    )

    sandbox_config = SandboxConfig(
        enabled=True,
        backend=cloudflare_sandbox,
    )

    agent = Agent(
        name="CloudflareSandboxAgent",
        id="cloudflare-sandbox-agent",
        llm=llm,
        sandbox=sandbox_config,
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        max_loops=10,
    )

    return agent, cloudflare_sandbox


def run_workflow(
    agent: Agent = None,
    sandbox: CloudflareSandbox = None,
    input_prompt: str = EXAMPLE_QUERY,
) -> str:
    """
    Run the agent workflow with a Cloudflare sandbox.

    Args:
        agent: The agent to use. If None, will create a new one.
        sandbox: The Cloudflare sandbox backend. If None, will be created with agent.
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
                print("\nCloudflare sandbox closed successfully.")
            except Exception as e:
                logger.warning(f"Failed to close Cloudflare sandbox: {e}")


if __name__ == "__main__":
    print("=== Cloudflare Sandbox Agent Example ===\n")

    if not os.getenv("CLOUDFLARE_SANDBOX_API_URL") or not os.getenv("CLOUDFLARE_SANDBOX_API_KEY"):
        print("Warning: CLOUDFLARE_SANDBOX_API_URL / CLOUDFLARE_SANDBOX_API_KEY not set.")
        print("Deploy the bridge Worker and set them before running this example:")
        print("  export CLOUDFLARE_SANDBOX_API_URL='https://your-sandbox-bridge.workers.dev'")
        print("  export CLOUDFLARE_SANDBOX_API_KEY='your-sandbox-api-key'\n")
        exit(1)

    result = run_workflow()
