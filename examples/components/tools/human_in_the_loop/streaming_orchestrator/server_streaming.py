import asyncio
import logging
from queue import Queue
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from dynamiq import Workflow
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphAgentManager, GraphOrchestrator
from dynamiq.nodes.llms.anthropic import Anthropic
from dynamiq.nodes.node import InputTransformer, NodeDependency
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool, MessageSenderTool
from dynamiq.runnables import RunnableConfig, RunnableResult
from dynamiq.runnables.base import NodeRunnableConfig
from dynamiq.types.feedback import FeedbackMethod
from dynamiq.types.streaming import StreamingConfig, StreamingEventMessage
from dynamiq.utils.logger import logger

HOST = "127.0.0.1"
PORT = 6001

app = FastAPI()

logging.basicConfig(level=logging.INFO)


def create_orchestrator() -> GraphOrchestrator:
    """
    Creates orchestrator.

    Returns:
        GraphOrchestrator: The configured orchestrator.
    """
    llm = Anthropic(
        name="LLM",
        connection=AnthropicConnection(),
        model="claude-3-5-sonnet-20241022",
        temperature=0.1,
    )

    def process_feedback(context: dict[str, Any], **kwargs):
        feedback = context.get("history")[-1]["content"]
        if feedback == "SEND":
            return {"result": "Email was sent!", **context}
        elif feedback == "CANCEL":
            return {"result": "Email was canceled!", **context}
        return {"result": "Unknown command", **context}

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
    )

    agent = Agent(llm=llm)
    human_feedback_tool = HumanFeedbackTool(
        input_transformer=InputTransformer(
            selector={
                "sketch": "$.history[-1].content",
            },
        ),
        input_method=FeedbackMethod.STREAM,
        msg_template="Generated draft of email: {{sketch}}." " Approve by sending SEND, cancel by sending CANCEL.",
    )
    orchestrator.add_state_by_tasks("generate_sketch", [agent])
    orchestrator.add_state_by_tasks("gather_feedback", [human_feedback_tool])
    orchestrator.add_state_by_tasks("process_feedback", [process_feedback])

    orchestrator.add_edge(START, "generate_sketch")
    orchestrator.add_edge("generate_sketch", "gather_feedback")
    orchestrator.add_edge("gather_feedback", "process_feedback")
    orchestrator.add_edge("process_feedback", END)

    return orchestrator, human_feedback_tool.id


def run_workflow(nodes, input_data, config) -> RunnableResult:
    """Runs orchestrator"""
    wf = Workflow(flow=Flow(nodes=nodes))

    _ = wf.run(input_data=input_data, config=config)


async def _send_stream_events_by_ws(websocket: WebSocket, send_handler: Any):
    """Sends streaming events by websocket."""
    try:
        async for event in send_handler:
            await websocket.send_text(event.to_json())
        logger.info("All streaming events sent")
    except WebSocketDisconnect as e:
        logger.error(f"WebSocket disconnected. Error: {e}")
    except asyncio.CancelledError:
        logger.error("Task cancelled")
    except Exception as e:
        logger.error(f"Unexpected error. Error: {e}")
    finally:
        pass


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connected")

    message_queue = Queue()

    message_tool = MessageSenderTool(
        output_method=FeedbackMethod.STREAM,
        msg_template="Workflow started execution!",
        streaming=StreamingConfig(enabled=True),
    )

    orchestrator, feedback_tool_id = create_orchestrator()

    final_status_tool = MessageSenderTool(
        output_method=FeedbackMethod.STREAM,
        msg_template="Workflow finished with status: {{status}}",
        streaming=StreamingConfig(enabled=True),
        depends=[NodeDependency(orchestrator)],
        input_transformer=InputTransformer(
            selector={
                "status": f"$.{orchestrator.id}.output.content",
            },
        ),
    )

    send_handler = AsyncStreamingIteratorCallbackHandler()

    asyncio.create_task(_send_stream_events_by_ws(websocket, send_handler))
    await asyncio.sleep(0.01)

    config = RunnableConfig(
        callbacks=[send_handler],
        nodes_override={
            feedback_tool_id: NodeRunnableConfig(
                streaming=StreamingConfig(
                    enabled=True,
                    input_queue=message_queue,
                )
            )
        },
    )

    try:
        while True:
            ws_data = await websocket.receive_text()
            try:
                ws_event = StreamingEventMessage.model_validate_json(ws_data)
            except ValueError as e:
                logger.error(f"Error parsing WebSocket message: {e}")

            if ws_event.entity_id is None:
                asyncio.get_running_loop().run_in_executor(
                    None, run_workflow, [message_tool, orchestrator, final_status_tool], ws_event.data["input"], config
                )

            else:
                message_queue.put(ws_event.to_json())

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
