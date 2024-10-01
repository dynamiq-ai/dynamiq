import asyncio
import logging
from functools import partial

import chainlit as cl
from dotenv import load_dotenv

from dynamiq import Workflow, runnables
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import ScaleSerp, ZenRows
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.nodes.tools.zenrows import ZenRowsTool
from dynamiq.types.streaming import StreamingConfig
from dynamiq.utils.logger import logger

logger.setLevel(logging.INFO)

# Constants
SERP_API_URL = "https://api.scaleserp.com/search"
ZENROWS_API_URL = "https://api.zenrows.com/v1/"
GPT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0.1
MAX_TOKENS = 4000
AGENT_ROLE = "professional recruiter"
AGENT_GOAL = "is to engage the user and provide helpful job-related information"
AGENT_STREAMING_EVENT = "react-agent"


def load_environment_variables():
    load_dotenv()


async def run_wf_async(
    wf: Workflow, wf_data: dict, streaming: AsyncStreamingIteratorCallbackHandler
) -> None:
    wf_run = partial(
        wf.run,
        input_data=wf_data,
        config=runnables.RunnableConfig(callbacks=[streaming]),
    )
    asyncio.get_running_loop().run_in_executor(None, wf_run)


def create_react_agent():
    load_environment_variables()

    # Set up API connections
    serp_connection = ScaleSerp()
    scrape_connection = ZenRows()

    # Create tools
    tool_search = ScaleSerpTool(connection=serp_connection)
    tool_scrape = ZenRowsTool(connection=scrape_connection)

    # Configure OpenAI connection and language model
    connection_openai = OpenAIConnection()
    llm_openai = OpenAI(
        connection=connection_openai,
        model=GPT_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    # Create the agent with tools and configuration
    react_agent = ReActAgent(
        name="React Agent",
        llm=llm_openai,
        tools=[tool_search, tool_scrape],
        role=AGENT_ROLE,
        goal=AGENT_GOAL,
        streaming=StreamingConfig(enabled=True, event=AGENT_STREAMING_EVENT),
    )

    wf = Workflow(flow=Flow(nodes=[react_agent]))
    return wf


@cl.on_chat_start
async def start_chat():
    agent_wf = create_react_agent()
    cl.user_session.set("react_agent", agent_wf)


@cl.on_message
async def main(message: cl.Message):
    agent_wf = cl.user_session.get("react_agent")

    msg = cl.Message(content="")
    await msg.send()

    streaming = AsyncStreamingIteratorCallbackHandler()

    input_msg = message.content

    # Run in async mode to avoid blocking the main thread
    async with cl.Step(name="React Agent processing") as step:
        step.input = input_msg

        await asyncio.create_task(
            run_wf_async(agent_wf, {"input": input_msg}, streaming)
        )
        await asyncio.sleep(0.01)

        idx = 1
        async for event in streaming:
            if event.event == AGENT_STREAMING_EVENT:
                content = event.data.get("model_observation", {}).get("initial", "")
                if content:
                    async with cl.Step(name=f"React Agent step {idx}") as child_step:
                        child_step.output = content
                    idx += 1

                tool_using = event.data.get("model_observation", {}).get("tool_using")
                tool_input = event.data.get("model_observation", {}).get("tool_input")
                tool_output = event.data.get("model_observation", {}).get("tool_output")

                if tool_using and tool_input and tool_output:
                    async with cl.Step(name=f"Tool: {tool_using}") as tool_step:
                        tool_step.input = tool_input
                        tool_step.output = tool_output

                final_answer = event.data.get("final_answer")
                if final_answer:
                    async with cl.Step(name="Final Answer") as final_step:
                        final_step.output = final_answer
                    step.output = final_answer
                    msg.content = final_answer
                    await msg.update()
                    break

    if not msg.content:
        msg.content = "I apologize, but I couldn't generate a final answer. Please try asking your question again."
        await msg.update()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
