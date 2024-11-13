from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import E2B
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.tools.e2b_sandbox import E2BInterpreterTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

# Constants
AGENT_ROLE = "professional writer,goal is to produce a well-written and informative response"
INPUT_QUESTION = "Sum first 10 randmos numbers"


def run_agent(event: str = "data") -> str:
    """
    Runs the OpenAI LLM node with streaming enabled.
    """
    # Set up the OpenAI node with streaming enabled
    llm = setup_llm(model_provider="groq", model_name="groq/llama3-70b-8192")
    agent = SimpleAgent(
        name=" Agent",
        llm=llm,
        role=AGENT_ROLE,
        id="agent",
        streaming=StreamingConfig(enabled=True, event=event),
    )
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
        agents=[agent],
        streaming=StreamingConfig(enabled=True, event=event, mode=StreamingMode.ALL),
    )

    # Set up streaming callback handler to capture streamed output
    streaming_handler = StreamingIteratorCallbackHandler()

    # Run the LLM node with streaming support
    response = agent.run(
        input_data={"input": INPUT_QUESTION}, config=RunnableConfig(callbacks=[streaming_handler], streaming=True)
    )
    print("Response:", response)

    # Collect the streamed responses
    print("Streaming Output:")
    full_content = ""
    for chunk in streaming_handler:
        print("Chunk", chunk)
        chunk_data = chunk.data
        print("Chunk Data", chunk_data)
        content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
        if content:
            full_content += content
            print("Chunk Content", content)
    return full_content


# Run the OpenAI node in structured output mode with streaming
output = run_agent(event="event")

print("Streamed Output:", output)
