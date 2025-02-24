import asyncio

import streamlit as st

from dynamiq import Workflow
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools import TavilyTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

AGENT_ROLE = "A helpful Assistant with access to web tools."
INPUT_TASK = "Research on Google. Do at least 3 iteratiohns"


def streamlit_callback(messages):
    for index, message in enumerate(messages):
        st.markdown(f"**Step**: {message}")
        if index != len(messages) - 1:
            st.markdown("---")


def run_agent(request: str, send_handler: AsyncStreamingIteratorCallbackHandler) -> str:
    """
    Creates and runs agent

    Args:
    send_handler (AsyncStreamingIteratorCallbackHandler): Handler of output messages.

    Returns:
        str: Agent final output.
    """
    send_handler.queue.put_nowait({"Something is here"})
    connection_tavily = TavilyConnection()
    tool_search = TavilyTool(connection=connection_tavily)

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)
    research_agent = ReActAgent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.DEFAULT,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL, by_tokens=False),
    )

    flow = Workflow(
        flow=Flow(nodes=[research_agent]),
    )

    response = flow.run(input_data={"input": request}, config=RunnableConfig(callbacks=[send_handler]))
    return response.output[research_agent.id]["output"]["content"]


async def _send_stream_events_by_ws(send_handler):
    messages = []
    async for message in send_handler:
        try:
            step = message.data["choices"][-1]["delta"]["step"]
            if "reasoning_" in step:
                content = message.data["choices"][-1]["delta"]["content"]["thought"]
                messages.append(content)
                streamlit_callback(messages)
        except Exception:
            continue


async def run_agent_async(request: str) -> str:
    send_handler = AsyncStreamingIteratorCallbackHandler()
    current_loop = asyncio.get_running_loop()
    task = current_loop.create_task(_send_stream_events_by_ws(send_handler))
    await asyncio.sleep(0.01)
    response = await current_loop.run_in_executor(None, run_agent, request, send_handler)

    await task

    return response

if __name__ == "__main__":
    print(asyncio.run(run_agent_async("Write report about Google")))
