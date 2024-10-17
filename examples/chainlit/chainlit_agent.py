import asyncio
import logging
from functools import partial

import chainlit as cl

from dynamiq import Workflow, runnables
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import ReflectionAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.types.streaming import StreamingConfig
from dynamiq.utils.logger import logger

logger.setLevel(logging.INFO)


TEMPERATURE = 0.1
MAX_TOKENS = 1000
OPENAI_MODEL = "gpt-4o"
AGENT_ROLE = "professional writer, goal is to provide high-quality content based on the user input"  # noqa: E501
AGENT_STREAMING_EVENT = "writer-agent"


async def run_wf_async(
    wf: Workflow, wf_data: dict, streaming: AsyncStreamingIteratorCallbackHandler
) -> None:
    wf_run = partial(
        wf.run,
        input_data=wf_data,
        config=runnables.RunnableConfig(callbacks=[streaming]),
    )
    asyncio.get_running_loop().run_in_executor(None, wf_run)


def create_writer_reflexion_agent():
    connection_openai = OpenAIConnection()
    llm_openai = OpenAI(
        connection=connection_openai,
        model=OPENAI_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )
    writer_agent = ReflectionAgent(
        name="Professional Writer Agent",
        llm=llm_openai,
        role=AGENT_ROLE,
        streaming=StreamingConfig(enabled=True, event=AGENT_STREAMING_EVENT),
    )
    wf = Workflow(flow=Flow(nodes=[writer_agent]))
    return wf


@cl.on_chat_start
async def start_chat():
    agent_wf = create_writer_reflexion_agent()
    cl.user_session.set("writer_agent", agent_wf)


@cl.on_message
async def main(message: cl.Message):
    agent_wf = cl.user_session.get("writer_agent")

    msg = cl.Message(content="")
    await msg.send()

    streaming = AsyncStreamingIteratorCallbackHandler()

    input_msg = message.content

    # Run in async mode to avoid blocking the main thread
    async with cl.Step(name="Writer Agent processing") as step:
        step.input = input_msg

        await asyncio.create_task(
            run_wf_async(agent_wf, {"input": input_msg}, streaming)
        )
        await asyncio.sleep(0.01)

        idx = 1
        async for event in streaming:
            if event.event == AGENT_STREAMING_EVENT:
                content = event.data["content"]
                async with cl.Step(name=f"Writer Agent step {idx}") as child_step:
                    child_step.output = content
                idx += 1

        step.output = content

    msg.content = content
    await msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
