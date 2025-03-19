from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

# Constants
AGENT_ROLE = "Helpful assistant with the goal of providing useful information and answering questions."
INPUT_QUESTION = "Add the first 10 numbers and determine if the result is a prime number."


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
        streaming=StreamingConfig(enabled=True, event=event, mode=StreamingMode.ALL, by_tokens=True),
    )

    streaming_handler = StreamingIteratorCallbackHandler()

    response = agent.run(input_data={"input": INPUT_QUESTION}, config=RunnableConfig(callbacks=[streaming_handler]))
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
