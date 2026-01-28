import asyncio
import json
import logging
import time

import websockets
from websocket import WebSocket

from dynamiq.nodes.tools.human_feedback import HumanFeedbackAction
from dynamiq.types.feedback import APPROVAL_EVENT
from dynamiq.types.streaming import StreamingEventMessage
from examples.components.tools.human_in_the_loop.streaming_orchestrator.server_streaming import PORT

logger = logging.getLogger(__name__)
WS_URI = f"ws://localhost:{PORT}/ws"
WF_ID = "dag-workflow"


async def handle_event_data(websocket: WebSocket, event: StreamingEventMessage, restart_count: int) -> bool:
    """Handle the event data received from the websocket.

    Args:
        websocket: The websocket connection
        event: The event received from the server
        restart_count: The number of times the workflow has been restarted

    Returns:
        bool: Whether to break the receive loop
    """
    # Handle approval event
    if event.event == APPROVAL_EVENT:
        feedback = input(event.data["template"])

        feedback_message = StreamingEventMessage(
            entity_id=event.entity_id,
            wf_run_id=event.wf_run_id,
            data={"feedback": feedback},
            event=APPROVAL_EVENT,
        )

        await websocket.send(feedback_message.to_json())

    # Handle content from choices
    if "choices" in event.data:
        if content := event.data.get("choices")[0].get("delta").get("content"):
            logger.info(f"Client: {content}")

    # Handle HumanFeedbackTool events (both ask and send actions)
    if event.source.type == "dynamiq.nodes.tools.HumanFeedbackTool":
        prompt = event.data.get("prompt", "")
        action = event.data.get("action", HumanFeedbackAction.SEND)
        logger.info(f"Client: {prompt}")

        # For 'ask' action, we need to get user input and send it back
        if action == HumanFeedbackAction.ASK:
            feedback = input("Your response: ")
            wf_run_event = StreamingEventMessage(
                entity_id=WF_ID,
                data={"content": feedback},
            )
            await websocket.send(wf_run_event.to_json())
    return False


async def websocket_client(input_query: str | None = None) -> float:
    """Run a websocket client and return the execution time in seconds.

    Args:
        input_query: The query to send. If None, will prompt for input.

    Returns:
        float: The execution time in seconds.
    """
    start_time = time.time()
    restart_count = 0
    max_restarts = 0

    async with websockets.connect(WS_URI, open_timeout=60) as websocket:
        if input_query is None:
            input_query = input("Provide request: ")

        logger.info(f"Client: Starting with query '{input_query}'")

        wf_run_event = StreamingEventMessage(entity_id=None, data={"stream": True, "input": {"input": input_query}})
        await websocket.send(wf_run_event.to_json())

        try:
            while True:
                event_data = json.loads(await websocket.recv())
                try:
                    event = StreamingEventMessage(**event_data)
                    if event.source.type == "dynamiq.workflows.Workflow":
                        print(event)
                        break
                except Exception as e:
                    logger.error(f"Client: Error while parsing message. Error: {e}")
                    continue

                should_break = await handle_event_data(websocket, event, restart_count)

                if should_break:
                    if restart_count < max_restarts:
                        logger.info("Client: Restarting workflow in same WebSocket connection")
                        wf_run_event = StreamingEventMessage(
                            entity_id=WF_ID,
                            data={
                                "stream": True,
                                "input": {"input": input_query},
                            },
                        )
                        await websocket.send(wf_run_event.to_json())
                        restart_count += 1
                    else:
                        break

        except websockets.ConnectionClosed:
            logger.error("Client: WebSocket connection closed by the server")

        await websocket.close()

    execution_time = time.time() - start_time
    logger.info(f"Client: Completed in {execution_time:.2f} seconds")
    return execution_time


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    query = input("Describe details of the email: ")
    execution_time = asyncio.run(websocket_client(query))
