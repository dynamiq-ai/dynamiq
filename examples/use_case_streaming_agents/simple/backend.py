from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.memory import Memory
from dynamiq.memory.backend.in_memory import InMemory
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig
from examples.llm_setup import setup_llm


def setup_agent(agent_role: str, streaming_enabled: bool) -> SimpleAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """
    llm = setup_llm()
    memory = Memory(backend=InMemory())
    streaming_config = StreamingConfig(enabled=streaming_enabled)

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
    """
    response_text = ""
    if agent.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()
        agent.run(
            input_data={"input": user_input}, config=RunnableConfig(callbacks=[streaming_handler], streaming=True)
        )

        for chunk in streaming_handler:
            content = chunk.data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                response_text += " " + content
                yield " " + content
    else:
        result = agent.run({"input": user_input})
        response_text = result.output.get("content", "")
        yield response_text
