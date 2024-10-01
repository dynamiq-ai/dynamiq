import asyncio
import os
import uuid
from functools import partial

import chainlit as cl

from dynamiq import Workflow, connections, flows, prompts, runnables
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.nodes import llms
from dynamiq.types.streaming import StreamingConfig


async def run_wf_async(
    wf: Workflow, wf_data: dict, streaming: AsyncStreamingIteratorCallbackHandler
) -> None:
    wf_run = partial(
        wf.run,
        input_data=wf_data,
        config=runnables.RunnableConfig(callbacks=[streaming]),
    )
    asyncio.get_running_loop().run_in_executor(None, wf_run)


@cl.on_chat_start
def start_chat():
    openai_api_key = os.getenv("OPENAI_API_KEY")
    wf = Workflow(
        id=str(uuid.uuid4()),
        flow=flows.Flow(
            nodes=[
                llms.OpenAI(
                    name="OpenAI",
                    model="gpt-3.5-turbo",
                    connection=connections.OpenAI(
                        api_key=openai_api_key,
                    ),
                    prompt=prompts.Prompt(
                        messages=[
                            prompts.Message(
                                role="user",
                                content="{{prompt_content}}",
                            ),
                        ],
                    ),
                    streaming=StreamingConfig(enabled=True),
                ),
            ],
        ),
    )
    cl.user_session.set("dynamiq", wf)


@cl.on_message
async def main(message: cl.Message):
    wf = cl.user_session.get("dynamiq")

    msg = cl.Message(content="")
    await msg.send()

    streaming = AsyncStreamingIteratorCallbackHandler()

    # Run in async mode to avoid blocking the main thread
    await asyncio.create_task(
        run_wf_async(wf, {"prompt_content": message.content}, streaming)
    )
    await asyncio.sleep(0.01)

    async for event in streaming:
        if event.entity_id != wf.id:
            # All streaming events without final with full output
            if data := event.data["choices"][0]["delta"]["content"]:
                await msg.stream_token(data)

    await msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
