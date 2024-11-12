from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.nodes.llms import OpenAI
from dynamiq.prompts import Prompt
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig


def run_openai_node_with_streaming(prompt: Prompt):
    """
    Runs the OpenAI LLM node with streaming enabled.
    """
    # Set up the OpenAI node with streaming enabled
    openai_node = OpenAI(
        model="gpt-4o-mini",
        streaming=StreamingConfig(
            enabled=True,
            event="data",
        ),
    )

    # Set up streaming callback handler to capture streamed output
    streaming_handler = StreamingIteratorCallbackHandler()

    # Run the LLM node with streaming support
    response = openai_node.run(
        input_data={}, prompt=prompt, config=RunnableConfig(callbacks=[streaming_handler], streaming=True)
    )
    print("Response:", response)

    # Collect the streamed responses
    print("Streaming Output:")
    full_content = ""
    for chunk in streaming_handler:
        chunk_data = chunk.data
        content = chunk_data.get("choices", [{}])[0].get("delta", {}).get("content")
        if content:
            full_content += content
            print(content)
    return full_content


# Example prompt using OpenAI with structured output in streaming mode
prompt = Prompt(
    messages=[
        {"role": "system", "content": "Extract document information."},
        {
            "role": "user",
            "content": (
                "I recently read 'Harry Potter 7', which is about a young wizard's journey."
                " It can be categorized as fiction, fantasy, and young adult literature."
            ),
        },
    ]
)

# Run the OpenAI node in structured output mode with streaming
output = run_openai_node_with_streaming(
    prompt=prompt,
)

print("Streamed Output:", output)
