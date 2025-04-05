import asyncio

import streamlit as st

from dynamiq import Workflow
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveAgentManager, AdaptiveOrchestrator, ActionCommand

from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools import TavilyTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from examples.llm_setup import setup_llm

AGENT_RESEARCHER_ROLE = "A helpful Assistant with access to web tools."
AGENT_WRITER_ROLE = "You are helpful assistant that accumulates key findings into report."

INPUT_TASK = "Research on Google. Do at least 3 iteratiohns"


def streamlit_callback(message):
    st.markdown(f"{message}")


def run_orchestrator(request: str, send_handler: AsyncStreamingIteratorCallbackHandler) -> str:
    """
    Creates and runs orchestrator
    Args:
    send_handler (AsyncStreamingIteratorCallbackHandler): Handler of output messages.
    Returns:
        str: Agent final output.
    """
    connection_tavily = TavilyConnection()
    tool_search = TavilyTool(connection=connection_tavily)

    llm = setup_llm(model_provider="gpt", model_name="gpt-4o-mini", temperature=0)

    research_agent = ReActAgent(
        name="Research Agent",
        id="Research Agent",
        llm=llm,
        tools=[tool_search],
        role=AGENT_RESEARCHER_ROLE,
        inference_mode=InferenceMode.STRUCTURED_OUTPUT,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL, by_tokens=False),
    )

    writer_agent = ReActAgent(
        name="Writer Agent",
        id="Writer Agent",
        llm=llm,
        role=AGENT_WRITER_ROLE,
        inference_mode=InferenceMode.STRUCTURED_OUTPUT,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL, by_tokens=False),
    )

    agent_manager = AdaptiveAgentManager(llm=llm)

    linear_orchestrator = AdaptiveOrchestrator(
        manager=agent_manager,
        agents=[research_agent, writer_agent],
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL, by_tokens=False),
    )

    flow = Workflow(
        flow=Flow(nodes=[linear_orchestrator]),
    )

    response = flow.run(input_data={"input": request}, config=RunnableConfig(callbacks=[send_handler]))
    return response.output[linear_orchestrator.id]["output"]["content"]


async def _send_stream_events_by_ws(send_handler):
    async for message in send_handler:
        if "choices" in message.data:
            step = message.data["choices"][-1]["delta"]["step"]
            if step == "manager_planning":
                next_action = message.data["choices"][-1]["delta"]["content"]["next_action"]
                if next_action["command"] == ActionCommand.DELEGATE:
                    task = next_action["task"]
                    agent = next_action["agent"]
                    content = f"Delegating task: '{task}' for agent: **{agent}**."
            elif step == "reasoning":
                content = message.data["choices"][-1]["delta"]["content"]["thought"]
            elif step == "final":
                content = message.data["choices"][-1]["delta"]["content"]
            else:
                print(f"unhandled {step}")
                content = message.data["choices"][-1]["delta"]["content"]
                continue
            entity = message.data["choices"][-1]["delta"]["source"]
            content = f"**{entity}:**  \n" + str(content)
            streamlit_callback(content)


async def run_orchestrator_async(request: str) -> str:
    send_handler = AsyncStreamingIteratorCallbackHandler()
    current_loop = asyncio.get_running_loop()
    task = current_loop.create_task(_send_stream_events_by_ws(send_handler))
    await asyncio.sleep(0.01)
    response = await current_loop.run_in_executor(None, run_orchestrator, request, send_handler)

    await task

    return response


if __name__ == "__main__":
    print(asyncio.run(run_orchestrator_async("Write report about Google")))
