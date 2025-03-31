import os

from composio import Action
from composio_tool import ComposioTool

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents import Agent
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm


def setup_agent() -> Agent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """
    tool_1 = ComposioTool(action=Action.LINEAR_LIST_LINEAR_PROJECTS, api_key=os.getenv("COMPOSIO_API_KEY"))
    tool_2 = ComposioTool(action=Action.LINEAR_LIST_LINEAR_TEAMS, api_key=os.getenv("COMPOSIO_API_KEY"))
    tool_3 = ComposioTool(action=Action.LINEAR_CREATE_LINEAR_ISSUE, api_key=os.getenv("COMPOSIO_API_KEY"))

    llm = setup_llm()
    memory = Memory(backend=InMemory())
    streaming_config = StreamingConfig(enabled=True, mode=StreamingMode.FINAL, by_tokens=True)

    agent = Agent(
        name="PM Manager",
        llm=llm,
        tools=[tool_1, tool_2, tool_3],
        memory=memory,
        streaming=streaming_config,
    )

    return agent


def generate_agent_response(agent: Agent, user_input: str):
    """
    Processes the user input using the agent. Supports both streaming and non-streaming responses.
    """
    if agent.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()
        agent.run(
            input_data={"input": user_input, "user_id": "1", "session_id": "1"},
            config=RunnableConfig(callbacks=[streaming_handler]),
        )

        response_text = ""

        for chunk in streaming_handler:
            print(chunk)
            content = chunk.data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                response_text += " " + content
                yield " " + content

    else:
        result = agent.run({"input": user_input})
        response_text = result.output.get("content", "")
        yield response_text
