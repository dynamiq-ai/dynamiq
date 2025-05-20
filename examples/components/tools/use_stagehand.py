from dynamiq import Workflow
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.stagehand_tool import StagehandConnection, StagehandTool


def set_wf_with_agent():
    stagehand_tool = StagehandTool(
        connection=StagehandConnection(),
        model_name="gpt-4o",
    )

    llm = OpenAI(
        id="openai",
        connection=OpenAIConnection(),
        model="gpt-4o",
        temperature=0.3,
        max_tokens=1000,
    )

    agent = ReActAgent(
        id="react-agent",
        llm=llm,
        tools=[stagehand_tool],
        role="assistant",
        max_loops=10,
    )

    wf = Workflow()
    wf.flow.add_nodes(agent)
    return wf


if __name__ == "__main__":
    wf = set_wf_with_agent()

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
