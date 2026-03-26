import os

from dynamiq import Workflow
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import CuaDesktop as CuaDesktopConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.llms.anthropic import Anthropic
from dynamiq.nodes.tools.cua_desktop import CuaDesktopTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.storages.file import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore


def set_wf_with_agent(cm):
    """Set up workflow with ReAct agent using CuaDesktop tool."""
    cua_tool = CuaDesktopTool(
        connection=CuaDesktopConnection(computer_name=os.getenv("CUA_COMPUTER_NAME")),
        is_postponed_component_init=True,
    )

    llm = Anthropic(
        id="anthropic",
        connection=AnthropicConnection(),
        model="claude-sonnet-4-5-20250929",
        is_postponed_component_init=True,
    )

    agent = Agent(
        id="agent",
        llm=llm,
        tools=[cua_tool],
        role="Computer Use Agent",
        max_loops=50,
        is_postponed_component_init=True,
        inference_mode=InferenceMode.XML,
        file_store=FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True),
        summarization_config=SummarizationConfig(
            enabled=True,
            max_token_context_length=50000,
            context_usage_ratio=0.5,
            context_history_length=10,
        ),
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
    """Run the CuaDesktop tool example."""
    with get_connection_manager() as cm:
        wf = set_wf_with_agent(cm)

        # Example 1
        print("\nExample 1")
        result = wf.run(
            input_data={
                "input": (
                    "Use the CuaDesktop tool to open YouTube, search for 'Eurovision', "
                    "and extract the title of the 10 videos shown in the results. "
                    "Use keyboard shortcuts."
                )
            }
        )

        print("Agent result:")
        print(result.output.get("agent", {}).get("output", {}).get("content"))

        # Example 2
        print("\nExample 2")
        result = wf.run(
            input_data={
                "input": (
                    "Create a detailed report on the performance of the"
                    " blue chip stocks throughout the Q3 of 2025."
                    " Use CuaDesktopTool to extensively search the web and extract the data."
                    " Save the report to a file called 'blue_chip_stocks_report.txt'"
                    " and return the size of the file in bytes."
                )
            }
        )

        print("Agent result:")
        print(result.output.get("agent", {}).get("output", {}).get("content"))

        # Example 3
        print("\nExample 3")
        result = wf.run(
            input_data={
                "input": "Generate python script that prints 'hello world' using FileWriteTool"
                " Upload the file to the CuaDesktop tool and run it using bash."
                " List files in the folder to verify it is uploaded"
            }
        )
        print("Agent result:")
        print(result.output.get("agent", {}).get("output", {}).get("content"))


if __name__ == "__main__":
    main()
