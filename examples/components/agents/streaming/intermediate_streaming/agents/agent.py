import asyncio

import streamlit as st

from dynamiq import Workflow
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools import TavilyTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

AGENT_ROLE = "A helpful Assistant with access to web tools."


def init_stream_ui(reset: bool = False):
    """Initialize or reset dynamic step boxes for streaming UI."""
    if reset or "step_placeholders" not in st.session_state:
        st.session_state.step_placeholders = {}
        st.session_state.step_contents = {}
        st.session_state.step_order = []
        st.session_state.current_loop_nums = {}


def streamlit_callback(step: str, content):
    if not content:
        return

    if step not in ["reasoning", "answer"]:
        return

    step_key = (step or "").lower().strip() or "unknown"
    display_name = step_key.capitalize()

    # Initialize loop tracking if not exists
    if "current_loop_nums" not in st.session_state:
        st.session_state.current_loop_nums = {}

    # Handle the reasoning step
    if step == "reasoning" and isinstance(content, dict):
        thought = content.get("thought", "")
        loop_num = content.get("loop_num", 0)

        current_loop = st.session_state.current_loop_nums.get(step_key, -1)
        if current_loop != loop_num and current_loop != -1:
            # Add separator when loop changes
            if step_key in st.session_state.step_contents:
                st.session_state.step_contents[step_key] += "\n\n"

        st.session_state.current_loop_nums[step_key] = loop_num
        content_to_display = thought
    else:
        # Handle the answer step
        content_to_display = str(content) if not isinstance(content, str) else content

    # Create a new box for a step on first occurrence
    if step_key not in st.session_state.step_placeholders:
        box = st.container(border=True)
        with box:
            st.markdown(f"### {display_name}")
            placeholder = st.empty()
        st.session_state.step_placeholders[step_key] = placeholder
        st.session_state.step_contents[step_key] = ""
        st.session_state.step_order.append(step_key)

    # Append the incoming content and update the box
    st.session_state.step_contents[step_key] += content_to_display
    st.session_state.step_placeholders[step_key].markdown(st.session_state.step_contents[step_key])


def run_agent(request: str, send_handler: AsyncStreamingIteratorCallbackHandler) -> str:
    """
    Creates and runs agent
    Args:
    send_handler (AsyncStreamingIteratorCallbackHandler): Handler of output messages.
    Returns:
        str: Agent final output.
    """
    connection_tavily = TavilyConnection()
    tool_search = TavilyTool(connection=connection_tavily)

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)
    research_agent = Agent(
        name="Agent",
        id="Agent",
        llm=llm,
        tools=[tool_search],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
    )

    flow = Workflow(
        flow=Flow(nodes=[research_agent]),
    )

    response = flow.run(input_data={"input": request}, config=RunnableConfig(callbacks=[send_handler]))
    return response.output[research_agent.id]["output"]["content"]


async def _send_stream_events_by_ws(send_handler):
    async for message in send_handler:
        data = message.data
        if not isinstance(data, dict):
            continue
        choices = data.get("choices") or []
        if not choices:
            continue
        delta = choices[-1].get("delta", {})
        step = delta.get("step", "")
        content = delta.get("content", "")
        if step and content:
            streamlit_callback(step, content)


async def run_agent_async(request: str) -> str:
    init_stream_ui(reset=True)
    send_handler = AsyncStreamingIteratorCallbackHandler()
    current_loop = asyncio.get_running_loop()
    task = current_loop.create_task(_send_stream_events_by_ws(send_handler))
    await asyncio.sleep(0.01)
    response = await current_loop.run_in_executor(None, run_agent, request, send_handler)

    await task

    return response


if __name__ == "__main__":
    print(asyncio.run(run_agent_async("Write a report about Google.")))
