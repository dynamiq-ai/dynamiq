import asyncio
import os

from dynamiq import Workflow
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.nodes.llms import Anthropic
from dynamiq.prompts.prompts import Message, Prompt
from dynamiq.types.streaming import StreamingConfig
from examples.use_cases.chainlit.main_chainlit_agent import run_wf_async


async def start_chat():
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    anthropic_node = Anthropic(
        name="anthropic",
        model="claude-3-7-sonnet-20250219",
        connection=AnthropicConnection(api_key=ANTHROPIC_API_KEY),
        prompt=Prompt(
            messages=[
                Message(
                    role="user",
                    content="What is an LLM?",
                ),
            ],
        ),
        thinking_enabled=True,
        max_tokens=1100,  # Must be greater than 1024 (the default value for "thinking")
        temperature=1.0,  # Should always be set to 1.0 when "thinking" is enabled
        streaming=StreamingConfig(enabled=True),
    )
    wf = Workflow()
    wf.flow.add_nodes(anthropic_node)
    streaming = AsyncStreamingIteratorCallbackHandler()

    await asyncio.create_task(run_wf_async(wf, {}, streaming))
    await asyncio.sleep(0.01)
    thinking_block = []
    async for event in streaming:
        if event.entity_id != wf.id:
            # All streaming events that contains thinking blocks
            if event.data["choices"][0]["delta"].get("thinking_blocks"):
                if data := event.data["choices"][0]["delta"]["thinking_blocks"][0]["thinking"]:
                    thinking_block.append(data)
    print("".join(thinking_block))


if __name__ == "__main__":
    asyncio.run(start_chat())
