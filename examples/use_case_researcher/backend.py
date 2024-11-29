import io

from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B, Firecrawl, ScaleSerp
from dynamiq.memory import Memory
from dynamiq.memory.backend.in_memory import InMemory
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.nodes.tools.firecrawl import FirecrawlTool
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.types import Behavior, InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

# Agent Constants
AGENT_ROLE = (
    "Assistant that can access the internet and Python coding tools."
    "It performs preliminary searches, scrapes data, writes code, and executes it."
    "Tasks are broken into smaller subtasks and solved systematically."
    "The agent ensures results are verified and rechecked before final delivery."
)

PROMPT = (
    "From the input file, for each company, find the company site, scrape it, and find the LinkedIn page."
    "Deliver the final answer in a table with columns: company name, company site, LinkedIn page."
    "Cover this for all items."
)


def read_file_as_bytesio(file) -> io.BytesIO:
    """
    Convert a file to BytesIO object for processing.
    """
    if not file:
        return None

    file_content = file.read()
    file_io = io.BytesIO(file_content)
    file_io.name = file.name
    file_io.description = getattr(file, "description", "No description provided")

    return file_io


def setup_agent() -> ReActAgent:
    """
    Set up and configure the ReAct agent.
    """
    streaming_config = StreamingConfig(enabled=True, mode=StreamingMode.FINAL, by_tokens=True)

    # Set environment variables (use secure methods in production)

    tools = [
        E2BInterpreterTool(connection=E2B()),
        FirecrawlTool(connection=Firecrawl()),
        ScaleSerpTool(connection=ScaleSerp()),
    ]

    memory = Memory(backend=InMemory())

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o", temperature=0.001)

    return ReActAgent(
        name="Assistant",
        llm=llm,
        role=AGENT_ROLE,
        id="react-agent",
        tools=tools,
        streaming=streaming_config,
        max_loops=30,
        memory=memory,
        inference_mode=InferenceMode.XML,
        behaviour_on_max_loops=Behavior.RETURN,
    )


agent = setup_agent()


def generate_agent_response(agent: ReActAgent, user_input: str, files=None):
    """
    Generate a response using the agent, with optional file processing.
    """
    if agent.streaming.enabled:
        streaming_handler = StreamingIteratorCallbackHandler()

        input_data = {"input": user_input}
        if files:
            input_data["files"] = files

        agent.run(input_data=input_data, config=RunnableConfig(callbacks=[streaming_handler]))

        for chunk in streaming_handler:
            content = chunk.data.get("choices", [{}])[0].get("delta", {}).get("content", "")
            if content:
                yield content
    else:
        result = agent.run({"input": user_input})
        yield result.output.get("content", "")
