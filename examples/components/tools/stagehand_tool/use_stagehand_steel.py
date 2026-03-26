import os

from dynamiq import Workflow
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.anthropic import Anthropic

from dynamiq.connections import SteelBrowser as SteelBrowserConnection
from dynamiq.connections import SteelBrowserEnvironment
from dynamiq.nodes.tools import Stagehand

USE_CLOUD = True


def set_wf_with_steeldev_agent(cm, use_cloud: bool = True):
    """Create workflow with Steel.dev-powered Stagehand tool.

    SteelDev supports two environments:
    1. Self-hosted: Use environment='self-hosted' with base_url to connect to your local Steel instance
    2. Cloud: Use environment='cloud' with api_key to connect to Steel.dev cloud service
    """
    if use_cloud:
        # Steel.dev cloud configuration
        connection = SteelBrowserConnection(
            environment=SteelBrowserEnvironment.CLOUD,
            api_key=os.getenv("STEEL_API_KEY"),
            model_api_key=os.getenv("ANTHROPIC_API_KEY"),
            session_config={"block_ads": True, "dimensions": {"width": 1280, "height": 800}},
        )
    else:
        # Self-hosted Steel configuration
        connection = SteelBrowserConnection(
            environment=SteelBrowserEnvironment.SELF_HOSTED,
            base_url="http://localhost:3000",
            model_api_key=os.getenv("ANTHROPIC_API_KEY"),
            session_config={"block_ads": True, "dimensions": {"width": 1280, "height": 800}},
        )

    stagehand_tool = Stagehand(
        connection=connection,
        model_name="claude-sonnet-4-20250514",
        is_postponed_component_init=True,
    )

    llm = Anthropic(
        id="anthropic",
        connection=AnthropicConnection(),
        model="claude-sonnet-4-20250514",
        temperature=0.3,
        max_tokens=1000,
        is_postponed_component_init=True,
    )

    agent = Agent(
        id="agent",
        llm=llm,
        tools=[stagehand_tool],
        role="assistant",
        max_loops=25,
        is_postponed_component_init=True,
    )

    wf = Workflow(
        flow=Flow(
            connection_manager=cm,
            init_components=True,
            nodes=[agent],
        )
    )
    return wf


def main():
    with get_connection_manager() as cm:
        wf = set_wf_with_steeldev_agent(cm, use_cloud=USE_CLOUD)

        result = wf.run(
            input_data={
                "input": (
                    "Use the Stagehand tool to open YouTube, search for 'Eurovision', "
                    "and extract the title of the first 10 videos shown in the results."
                )
            }
        )
        print("Agent result:")
        print(result.output.get("agent", {}).get("output", {}).get("content"))


if __name__ == "__main__":
    main()
