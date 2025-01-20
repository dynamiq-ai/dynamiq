import asyncio
import logging
from queue import Queue
from typing import Any, Literal

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig
from dynamiq.types.feedback import ApprovalConfig, FeedbackMethod
from dynamiq.types.streaming import StreamingConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

HOST = "127.0.0.1"
PORT = 6001

WF_ID = "dd643e12-fe89-4eef-b48c-050f01c74517"
app = FastAPI()

logging.basicConfig(level=logging.INFO)


class SocketMessage(BaseModel):
    type: Literal["run", "message"]
    content: Any

    def to_json(self, **kwargs) -> str:
        """Convert to JSON string.

        Returns:
            str: JSON string representation.
        """
        return self.model_dump_json(**kwargs)


PYTHON_TOOL_CODE = """
def run(inputs):
    return {"content": "Email sent"}
"""


def run_agent(request: str, input_queue: Queue, send_handler: AsyncStreamingIteratorCallbackHandler) -> dict:
    """
    Creates agent

    Returns:
        dict: Agent final output.
    """
    llm = setup_llm()

    email_sender_tool = Python(
        name="EmailSenderTool",
        description="Sends email. Put all email in string under 'email' key. ",
        code=PYTHON_TOOL_CODE,
        approval=ApprovalConfig(
            enabled=True,
            feedback_method=FeedbackMethod.STREAM,
            msg_template="Email sketch: {{input_data.email}}. "
            "Approve or cancel email sending. Send nothing for approval;"
            "provide feedback to cancel and regenerate.",
        ),
        streaming=StreamingConfig(enabled=True, input_queue=input_queue),
    )

    human_feedback_tool = HumanFeedbackTool(
        description="This tool can be used to request some clarifications from user.",
        input_method=FeedbackMethod.STREAM,
        streaming=StreamingConfig(enabled=True, input_queue=input_queue),
    )

    agent = ReActAgent(
        name="research_agent",
        role="You are a helpful assistant that has access to the internet using Tavily Tool. ",
        llm=llm,
        tools=[email_sender_tool, human_feedback_tool],
    )

    return agent.run(input_data={"input": request}, config=RunnableConfig(callbacks=[send_handler])).output["content"]


async def _send_stream_events_by_ws(websocket: WebSocket, send_handler: Any):
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

    try:
        while True:
            message_raw = await websocket.receive_text()
            message = SocketMessage.model_validate_json(message_raw)

            if message.type == "run":
                send_handler = AsyncStreamingIteratorCallbackHandler()

                asyncio.create_task(_send_stream_events_by_ws(websocket, send_handler))
                await asyncio.sleep(0.01)

                asyncio.get_running_loop().run_in_executor(
                    None, run_agent, message.content, message_queue, send_handler
                )

            elif message.type == "message":
                message_queue.put(message.content)

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
