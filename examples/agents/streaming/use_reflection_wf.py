from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.reflection import ReflectionAgent
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig
from examples.llm_setup import setup_llm

# Constants
AGENT_ROLE = "professional writer,goal is to produce a well-written and informative response"
INPUT_QUESTION = "What is the capital of France?"
STREAMING_EVENT = "data"
STREAMING_CONFIG = StreamingConfig(
    enabled=True,
    event=STREAMING_EVENT,
)


def run_simple_workflow() -> tuple[str, dict]:
    """
    Execute a workflow using the OpenAI agent to process a predefined question.

    Returns:
        tuple[str, dict]: The generated content by the agent and the trace logs.

    Raises:
        Exception: Captures and prints any errors during workflow execution.
    """
    llm = setup_llm()
    agent = ReflectionAgent(
        name=" Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        streaming=STREAMING_CONFIG,
    )
    streaming_handler = StreamingIteratorCallbackHandler()
    tracing = TracingCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": INPUT_QUESTION},
            config=RunnableConfig(callbacks=[tracing, streaming_handler], streaming=True),
        )
        for chunk in streaming_handler:
            chunk_data = chunk.data
            content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
            print(content)

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    output, traces = run_simple_workflow()
    print(output)
