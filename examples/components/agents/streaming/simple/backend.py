from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.memory import Memory
from dynamiq.memory.backends.in_memory import InMemory
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm


def setup_agent(agent_role: str, streaming_enabled: bool, streaming_mode: str) -> SimpleAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """

    llm = setup_llm()
    memory = Memory(backend=InMemory())

    mode_mapping = {"Answer": StreamingMode.FINAL, "Steps": StreamingMode.ALL}
    mode = mode_mapping.get(streaming_mode, StreamingMode.FINAL)
    streaming_config = StreamingConfig(enabled=streaming_enabled, mode=mode)

    agent = SimpleAgent(
        name="Agent",
        llm=llm,
        role=agent_role,
        id="agent",
        memory=memory,
        streaming=streaming_config,
    )
    return agent


def generate_agent_response(agent: SimpleAgent, user_input: str):
    """
    Processes the user input using the agent. Supports both streaming and non-streaming responses.
    Extracts and yields only the content within <output> tags.
    """
    response_text = ""
    if agent.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()
        agent.run(
            input_data={"input": user_input, "user_id": "1", "session_id": "1"},
            config=RunnableConfig(callbacks=[streaming_handler]),
        )

        for chunk in streaming_handler:
            content = chunk.data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                print(content)
                response_text += " " + content
                yield content
    else:
        result = agent.run({"input": user_input})
        response_text = result.output.get("content", "")
        yield response_text
