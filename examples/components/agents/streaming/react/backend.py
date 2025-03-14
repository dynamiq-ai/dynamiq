from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B, ScaleSerp
from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm


def setup_agent(agent_role: str, streaming_enabled: bool, streaming_mode: str, streaming_tokens: bool) -> ReActAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """
    llm = setup_llm()
    memory = Memory(backend=InMemory())
    mode_mapping = {"Answer": StreamingMode.FINAL, "Steps": StreamingMode.ALL}
    mode = mode_mapping.get(streaming_mode, StreamingMode.FINAL)
    streaming_config = StreamingConfig(enabled=streaming_enabled, mode=mode, by_tokens=streaming_tokens)
    tool_search = ScaleSerpTool(connection=ScaleSerp())
    tool_code = E2BInterpreterTool(connection=E2B())
    agent = ReActAgent(
        name="Agent",
        llm=llm,
        role=agent_role,
        id="agent",
        memory=memory,
        tools=[tool_code, tool_search],
        streaming=streaming_config,
    )
    return agent


def generate_agent_response(agent: ReActAgent, user_input: str):
    """
    Processes the user input using the agent. Supports both streaming and non-streaming responses.
    """
    if agent.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()
        agent.run(input_data={"input": user_input}, config=RunnableConfig(callbacks=[streaming_handler]))

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
