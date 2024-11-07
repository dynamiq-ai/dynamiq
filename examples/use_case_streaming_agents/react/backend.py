import os

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import ScaleSerp
from dynamiq.memory import Memory
from dynamiq.memory.backend.in_memory import InMemory
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm


def setup_agent(agent_role: str, streaming_enabled: bool) -> ReActAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """
    llm = setup_llm()
    memory = Memory(backend=InMemory())
    streaming_config = StreamingConfig(enabled=streaming_enabled)
    tool_search = ScaleSerpTool(connection=ScaleSerp())
    agent = ReActAgent(
        name="Agent",
        llm=llm,
        role=agent_role,
        id="agent",
        memory=memory,
        tools=[tool_search],
        streaming=streaming_config,
        streaming_mode=StreamingMode.ALL,
    )
    return agent


def generate_agent_response(agent: ReActAgent, user_input: str):
    """
    Processes the user input using the agent. Supports both streaming and non-streaming responses.
    """
    response_text = ""
    if agent.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()
        agent.run(
            input_data={"input": user_input}, config=RunnableConfig(callbacks=[streaming_handler], streaming=True)
        )

        for chunk in streaming_handler:
            if isinstance(chunk.data, dict):
                content = chunk.data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            elif isinstance(chunk.data, str):
                content = chunk.data
            if content:
                response_text += content
                yield content
    else:
        result = agent.run({"input": user_input})
        response_text = result.output.get("content", "")
        yield response_text
