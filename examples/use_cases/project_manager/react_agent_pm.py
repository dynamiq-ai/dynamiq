import os

from composio import Action
from composio_tool import ComposioTool

from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents.react import ReActAgent
from examples.llm_setup import setup_llm

# Create tool instance
tool_1 = ComposioTool(action=Action.LINEAR_LIST_LINEAR_PROJECTS, api_key=os.getenv("COMPOSIO_API_KEY"))
tool_2 = ComposioTool(action=Action.LINEAR_LIST_LINEAR_TEAMS, api_key=os.getenv("COMPOSIO_API_KEY"))
tool_3 = ComposioTool(action=Action.LINEAR_CREATE_LINEAR_ISSUE, api_key=os.getenv("COMPOSIO_API_KEY"))

llm = setup_llm()
memory = Memory(backend=InMemory())

agent = ReActAgent(
    name="AI Agent",
    llm=llm,
    tools=[tool_1, tool_2, tool_3],
    memory=memory,
)

result = agent.run(
    input_data={
        "input": (
            "Show me the project and team list. "
            "Create the one task with simple description in any available project."
        ),
    },
    config=None,
)
print(result.output)
