import re

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.memory import Memory
from dynamiq.memory.backend.in_memory import InMemory
from dynamiq.nodes.agents.reflection import ReflectionAgent
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm


def setup_agent(agent_role: str, streaming_enabled: bool, streaming_mode: str) -> ReflectionAgent:
    """
    Initializes an AI agent with a specified role and streaming configuration.
    """

    llm = setup_llm()
    memory = Memory(backend=InMemory())

    mode_mapping = {"Final": StreamingMode.FINAL, "All": StreamingMode.ALL}
    mode = mode_mapping.get(streaming_mode, StreamingMode.FINAL)
    streaming_config = StreamingConfig(enabled=streaming_enabled, mode=mode)

    agent = ReflectionAgent(
        name="Agent",
        llm=llm,
        role=agent_role,
        id="agent",
        memory=memory,
        streaming=streaming_config,
    )
    return agent


def generate_agent_response(agent: ReflectionAgent, user_input: str):
    """
    Processes the user input using the agent. Supports both streaming and non-streaming responses.
    Extracts and yields only the content within <output> tags.
    """
    response_text = ""
    if agent.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()
        agent.run(
            input_data={"input": user_input}, config=RunnableConfig(callbacks=[streaming_handler], streaming=True)
        )

        for chunk in streaming_handler:
            content = chunk.data.get("content", " ")
            if content:
                response_text += " " + content
                yield " " + content
    else:
        result = agent.run({"input": user_input})
        response_text = result.output.get("content", "")
        output_content = extract_output_content(response_text)
        if output_content:
            yield output_content


def extract_output_content(text):
    """
    Extracts content within <output> tags from the text.
    """
    match = re.search(r"<output>(.*?)</output>", text, re.DOTALL)
    return match.group(1).strip() if match else ""
