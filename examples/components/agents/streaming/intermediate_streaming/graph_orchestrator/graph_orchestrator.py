import asyncio
from typing import Any

import streamlit as st

from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.orchestrators.adaptive import ActionCommand
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingMode

AGENT_RESEARCHER_ROLE = "A helpful Assistant with access to web tools."
AGENT_WRITER_ROLE = "You are helfull assistant that accumulates key findings into report."

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

    llm = OpenAI(
        connection=OpenAIConnection(),
        model="gpt-4o",
        temperature=0.1,
    )

    email_writer = ReActAgent(
        name="email-writer-agent",
        llm=llm,
        role="Write personalized emails taking into account feedback.",
    )

    def gather_feedback(context: dict[str, Any], **kwargs):
        """Gather feedback about email draft."""
        feedback = input(
            f"Email draft:\n"
            f"{context.get('history', [{}])[-1].get('content', 'No draft')}\n"
            f"Type in SEND to send email, CANCEL to exit, or provide feedback to refine email: \n"
        )

        reiterate = True

        result = f"Gathered feedback: {feedback}"

        feedback = feedback.strip().lower()
        if feedback == "send":
            print("####### Email was sent! #######")
            result = "Email was sent!"
            reiterate = False
        elif feedback == "cancel":
            print("####### Email was canceled! #######")
            result = "Email was canceled!"
            reiterate = False

        return {"result": result, "reiterate": reiterate}

    def router(context: dict[str, Any], **kwargs):
        """Determines next state based on provided feedback."""
        if context.get("reiterate", False):
            return "generate_sketch"

        return END

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL, by_tokens=False),
    )

    orchestrator.add_state_by_tasks("generate_sketch", [email_writer])
    orchestrator.add_state_by_tasks("gather_feedback", [gather_feedback])

    orchestrator.add_edge(START, "generate_sketch")
    orchestrator.add_edge("generate_sketch", "gather_feedback")

    orchestrator.add_conditional_edge("gather_feedback", ["generate_sketch", END], router)

    response = orchestrator.run(input_data={"input": request}, config=RunnableConfig(callbacks=[send_handler]))
    return response.output["content"]


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
            elif step == "answer":
                content = "Finished execution: '" + message.data["choices"][-1]["delta"]["content"] + "'"
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
