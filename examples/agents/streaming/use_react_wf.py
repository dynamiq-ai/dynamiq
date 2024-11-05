from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

# Constants
AGENT_ROLE = (
    "teacher for children, goal is to craft a well-structured and simple final answer"
    "with a lot of emojis to empathize with the children."
)
QUERY = "Who won the Euro 2024?"
STREAMING_EVENT = "data"
STREAMING_CONFIG = StreamingConfig(
    enabled=True,
    event=STREAMING_EVENT,
)


def setup_react_agent() -> ReActAgent:
    """Set up and return a ReAct agent with specified LLM and tools."""
    llm = setup_llm()
    tool_search = ScaleSerpTool(connection=ScaleSerp())

    return ReActAgent(
        name="ReAct Agent - Children Teacher",
        id="react",
        llm=llm,
        tools=[tool_search],
        role=AGENT_ROLE,
        streaming=STREAMING_CONFIG,
        streaming_mode=StreamingMode.ALL,  # or StreamingMode.ALL for all steps
    )


def run_workflow(agent: ReActAgent = setup_react_agent(), input_prompt: str = QUERY) -> tuple[str, dict]:
    """Execute workflow with streaming support."""
    tracing = TracingCallbackHandler()
    streaming_handler = StreamingIteratorCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))

    try:
        result = wf.run(
            input_data={"input": input_prompt},
            config=RunnableConfig(callbacks=[tracing, streaming_handler], streaming=True),
        )

        # Process streaming chunks
        for chunk in streaming_handler:
            if chunk.data:
                print(chunk.data)
                content = chunk.data.get("content", {})
                if isinstance(content, dict):
                    chunk_type = content.get("type")
                    chunk_content = content.get("content")

                    if chunk_type == "final_answer":
                        print(f"Final Answer: {chunk_content}")
                    elif chunk_type == "intermediate":
                        print(f"Intermediate Step {content.get('loop')}: {chunk_content}")
                    elif chunk_type == "tool_execution":
                        print(f"Tool Execution: {chunk_content}")

        return result.output[agent.id]["output"]["content"], tracing.runs
    except Exception as e:
        print(f"An error occurred: {e}")
        return "", {}


if __name__ == "__main__":
    output, traces = run_workflow()
    print("Agent Output:", output)
