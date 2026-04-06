from dynamiq import Workflow
from dynamiq.connections import E2B as E2BDesktopConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.storages.file import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore

from dynamiq.nodes.tools.e2b_desktop import E2BDesktopTool


def set_wf_with_agent(cm):
    """Set up workflow with ReAct agent using E2B tool."""
    e2b_tool = E2BDesktopTool(
        connection=E2BDesktopConnection(),
        is_postponed_component_init=True,
    )

    from dynamiq.nodes.types import InferenceMode

    llm = OpenAI(
        id="openai",
        connection=OpenAIConnection(),
        model="gpt-5-chat-latest",
        is_postponed_component_init=True,
    )

    agent = Agent(
        id="agent",
        llm=llm,
        tools=[e2b_tool],
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
    """Run the E2B tool example."""
    with get_connection_manager() as cm:
        wf = set_wf_with_agent(cm)

        # Example 1
        print("\nExample 1")
        result = wf.run(
            input_data={
                "input": (
                    "Use the E2B tool to open YouTube, search for 'Eurovision', "
                    "and extract the title of the 10 videos shown in the results. "
                    "Use keyboard shortcuts mostly."
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
                    "Find out how many episodes there were in 'Breaking Bad', "
                    "create a list with 10 highest scoring episodes of the show (based on IMDb ratings), "
                    "save the list to a file called 'breaking_bad_episodes.txt' "
                    "and return the size of the file in bytes."
                )
            }
        )

        print("Agent result:")
        print(result.output.get("agent", {}).get("output", {}).get("content"))

        # Example 3
        print("\nExample 3")
        result = wf.run(
            input_data={
                "input": "Generate python script that prints 'hello world' using file-write tool."
                " Upload the file to the E2B tool and run it using bash."
                " List files in the folder to verify it is uploaded"
            }
        )
        print("Agent result:")
        print(result.output.get("agent", {}).get("output", {}).get("content"))


if __name__ == "__main__":
    main()
