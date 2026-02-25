from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.runnables import RunnableConfig
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

AGENT_ROLE = """
You are helpful assistant that can read and write files to the filesystem
and has access to core memory tools.
You can store and retrieve information from memory to help with tasks.
"""

EXAMPLE_QUERY = """
Create a file called 'test.txt' with the content 'Hello, world!'
Read the content of the file and return it.
"""


def setup_agent() -> Agent:
    """
    Set up and return a ReAct agent with filesystem tools and core memory tools enabled.

    Returns:
        Agent: Configured ReAct agent with filesystem and memory capabilities.
    """

    llm = setup_llm(model_provider="claude", model_name="claude-3-5-sonnet-20241022", temperature=0.2)

    file_store = FileStoreConfig(enabled=True, backend=InMemoryFileStore(), agent_file_write_enabled=True)

    agent = Agent(
        name="AgentFileInteractionWithMemory",
        id="AgentFileInteractionWithMemory",
        llm=llm,
        file_store=file_store,
        role=AGENT_ROLE,
        max_loops=5,
    )

    return agent


def run_workflow(
    agent: Agent = None, input_prompt: str = EXAMPLE_QUERY, workflow_type: str = "filesystem"
) -> tuple[str, dict, list]:
    """
    Run the agent workflow for either filesystem operations or memory operations.

    Args:
        agent: The agent to use, if None will create a new one
        input_prompt: The input prompt for the agent
        workflow_type: Either "filesystem" or "memory" to determine workflow type

    Returns:
        tuple: (output_content, tracing_data, streaming_data)
    """
    if agent is None:
        agent = setup_agent()

    tracing = TracingCallbackHandler()
    streaming_handler = StreamingIteratorCallbackHandler()

    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": input_prompt},
            config=RunnableConfig(callbacks=[tracing, streaming_handler]),
        )

        print(f"Files in storage after {workflow_type} workflow:")
        file_store = agent.tools[0].file_store
        for file_info in file_store.list_files():
            print(f"File: {file_info.name}")
            print(f"Path: {file_info.path}")
            print(f"Size: {file_info.size} bytes")
            print(f"Content: {file_store.retrieve(file_info.path).decode('utf-8')}")
            print("-" * 30)

        return result.output[agent.id]["output"]["content"]

    except Exception as e:
        logger.exception(f"An error occurred during {workflow_type} workflow: {e}")
        logger.error(f"ERROR: {e}")
        return "", {}, []


if __name__ == "__main__":
    print("=== Testing Filesystem Interaction ===")
    result = run_workflow(input_prompt=EXAMPLE_QUERY, workflow_type="filesystem")
    print(f"Filesystem Result: {result}")
