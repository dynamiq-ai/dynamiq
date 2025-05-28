import os

from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections.managers import get_connection_manager
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.stagehand_tool import StagehandConnection, StagehandTool


def set_wf_with_agent(cm):
    stagehand_tool = StagehandTool(
        connection=StagehandConnection(
            model_api_key=os.getenv("OPENAI_API_KEY"),
        ),
        model_name="gpt-4o",
        is_postponed_component_init=True,
    )

    llm = OpenAI(
        id="openai",
        connection=OpenAIConnection(),
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1000,
        is_postponed_component_init=True,
    )

    agent = ReActAgent(
        id="react-agent",
        llm=llm,
        tools=[stagehand_tool],
        role="assistant",
        max_loops=10,
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


if __name__ == "__main__":
    with get_connection_manager() as cm:
        wf = set_wf_with_agent(cm)

        result = wf.run(
            input_data={
                "input": (
                    "Use the Stagehand tool to open YouTube, search for 'Eurovision', "
                    "and extract the title of the first video shown in the results."
                )
            }
        )

    print("Agent result:")
    print(result.output.get("react-agent", {}).get("output", {}).get("content"))
