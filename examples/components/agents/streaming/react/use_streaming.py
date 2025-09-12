from dynamiq import Workflow
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode


def run_workflow_with_streaming():
    """
    Execute a workflow with two different agents and different streaming configurations:
    1. A ReActAgent that streams all content (reasoning, tool usage, and answers)
    2. A SimpleAgent that only streams the final answer

    Shows how to use:
    - Different streaming event channels
    - Different streaming modes
    - Token-based vs chunk-based streaming
    - Event filtering in the streaming handler
    """
    # Set up LLM
    llm = OpenAI(
        model="gpt-4o",
        temperature=0.7,
    )

    # Set up ReActAgent with full streaming (reasoning, tools, answers)
    react_agent = ReActAgent(
        name="Research Assistant",
        id="research_agent",
        llm=llm,
        tools=[],
        role="Research assistant that provides detailed analysis with step-by-step reasoning.",
        streaming=StreamingConfig(
            enabled=True,
            event="research_stream",  # Custom event channel name
            mode=StreamingMode.FINAL,
        ),
        max_loops=5,
        inference_mode=InferenceMode.DEFAULT,
    )

    # Set up streaming callback handler
    streaming_handler = StreamingIteratorCallbackHandler()

    # Create workflow with both agents
    wf = Workflow(flow=Flow(nodes=[react_agent]))

    # Run workflow
    result = wf.run(
        input_data={
            "input": "Hey",
        },
        config=RunnableConfig(callbacks=[streaming_handler]),
    )

    print("\n=== STREAMING OUTPUT ===\n")

    for chunk in streaming_handler:
        print(chunk)

    print(result)


if __name__ == "__main__":
    run_workflow_with_streaming()
