from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

# Constants
AGENT_ROLE = "professional writer,goal is to produce a well-written and informative response"
INPUT_QUESTION = "Sum the first 10 random numbers, and prove that the sum is correct."


def run_agent(event: str = "data") -> str:
    """
    Runs the Agent node with streaming enabled.
    """
    llm = setup_llm(model_provider="groq", model_name="groq/llama3-70b-8192")

    e2b_tool = E2BInterpreterTool(
        name="E2B Tool",
        connection=E2B(),
        id="e2b_tool",
    )
    agent = ReActAgent(
        name="React Agent",
        id="agent",
        llm=llm,
        tools=[e2b_tool],
        streaming=StreamingConfig(enabled=True, event=event, mode=StreamingMode.ALL),
    )

    streaming_handler = StreamingIteratorCallbackHandler()

    response = agent.run(
        input_data={"input": INPUT_QUESTION}, config=RunnableConfig(callbacks=[streaming_handler], streaming=True)
    )
    print("Response:", response)

    print("Streaming Output:")
    full_content = ""
    for chunk in streaming_handler:
        chunk_data = chunk.data
        content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
        if content:
            full_content += content
            print(content, end=" ")
    return full_content


output = run_agent(event="event")

print("Streamed Output:", output)
