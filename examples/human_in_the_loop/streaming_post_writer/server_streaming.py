import asyncio
import logging
from queue import Queue
from typing import Any, Literal
from uuid import uuid4

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphAgentManager, GraphOrchestrator
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.prompts import Message, Prompt
from dynamiq.runnables import RunnableConfig, RunnableResult
from dynamiq.types.streaming import StreamingConfig, StreamingEventMessage
from dynamiq.utils.logger import logger

app = FastAPI()

logging.basicConfig(level=logging.INFO)


class SocketMessage(BaseModel):
    type: Literal["run", "message"]
    content: str


def send_message(message: StreamingEventMessage, config: RunnableConfig):
    sender_callback_streaming = next(
        (item for item in config.callbacks if isinstance(item, AsyncStreamingIteratorCallbackHandler)), None
    )
    sender_callback_streaming.queue.put_nowait(message)


def create_orchestrator() -> GraphOrchestrator:
    """
    Creates orchestrator

    Returns:
        GraphOrchestrator: The configured orchestrator.
    """
    llm = OpenAI(
        name="OpenAI LLM",
        connection=OpenAIConnection(),
        model="gpt-4o-mini",
        temperature=0.1,
    )

    def generate_sketch(context: dict[str, Any], config: RunnableConfig = None, **kwargs):
        """Generate sketch"""
        input_queue = kwargs.get("input_queue")
        messages = context.get("messages")

        if feedback := context.get("feedback"):
            messages += [Message(role="user", content=f"Generate text again taking into account feedback {feedback}")]

        llm_openai = OpenAI(
            name="OpenAI LLM",
            connection=OpenAIConnection(),
            model="gpt-4o-mini",
            temperature=0.1,
            streaming=StreamingConfig(enabled=True),
        )

        response = llm_openai.run(
            input_data={},
            prompt=Prompt(
                messages=messages,
            ),
            config=config,
        ).output["content"]

        event_message = StreamingEventMessage(
            entity_id=str(uuid4()),
            data=(
                "This is draft of post. Type in: <br> • 'SEND' - to publish post,<br>"
                " • 'CANCEL' - to NOT publish post,<br> • Any feedback to refine post."
            ),
        )

        send_message(event_message, config)

        feedback = input_queue.get()

        context["messages"] += [
            Message(
                role="assistant",
                content=f"{response}",
            ),
            Message(
                role="user",
                content=f"{feedback}",
            ),
        ]

        return {"result": "Generated draft", **context}

    def accept_sketch(context: dict[str, Any], config: RunnableConfig = None, **kwargs):
        feedback = context.get("messages")[-1]["content"]

        if feedback == "SEND":
            event_message = StreamingEventMessage(
                entity_id=str(uuid4()),
                data="Message was sent!",
            )
            send_message(event_message, config)
            return END

        elif feedback == "CANCEL":
            print("Get here")
            event_message = StreamingEventMessage(
                entity_id=str(uuid4()),
                data="Message was NOT sent!",
            )
            send_message(event_message, config)
            return END

        return "generate_sketch"

    orchestrator = GraphOrchestrator(
        name="Graph orchestrator",
        manager=GraphAgentManager(llm=llm),
    )

    orchestrator.add_state_by_tasks("generate_sketch", [generate_sketch])

    orchestrator.add_edge(START, "generate_sketch")
    orchestrator.add_conditional_edge("generate_sketch", ["generate_sketch", END], accept_sketch)

    return orchestrator


def run_orchestrator(
    orchestrator, queue: Queue, handler, request="Write and publish small post about AI in Sales."
) -> RunnableResult:
    """Runs orchestrator"""
    _ = orchestrator.run(input_data={"input": request}, config=RunnableConfig(callbacks=[handler]), input_queue=queue)


async def _send_stream_events_by_ws(websocket: WebSocket, send_handler: Any):
    try:
        async for event in send_handler:
            print("message send")
            print(event)
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

    orchestrator = create_orchestrator()
    orchestrator.context = {
        "messages": [Message(role="assistant", content="Hello, how can I help you?")],
    }

    try:
        while True:
            message_raw = await websocket.receive_text()
            message = SocketMessage.model_validate_json(message_raw)

            if message.type == "run":
                send_handler = AsyncStreamingIteratorCallbackHandler()

                asyncio.create_task(_send_stream_events_by_ws(websocket, send_handler))
                await asyncio.sleep(0.01)

                orchestrator.context["messages"].append(Message(role="user", content=message.content))
                asyncio.get_running_loop().run_in_executor(
                    None, run_orchestrator, orchestrator, message_queue, send_handler
                )

            elif message.type == "message":
                message_queue.put(message.content)

    except WebSocketDisconnect:
        logging.info("WebSocket disconnected")
