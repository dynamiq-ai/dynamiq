import asyncio
import logging
from collections import defaultdict

import websockets

from dynamiq.nodes.tools.human_feedback import HFStreamingInputEventMessage, HFStreamingInputEventMessageData
from dynamiq.types.streaming import StreamingEventMessage
from examples.components.core.websocket.ws_server_fastapi import HF_NODE_STREAMING_EVENT, HOST, PORT, WF_ID

logger = logging.getLogger(__name__)


WS_URI = f"ws://{HOST}:{PORT}/workflows/test"
INPUT = {"a": 1, "b": 2, "input": "How old are you?"}
WF_RUNS = 3


async def websocket_client():
    wf_runs = 0
    async with websockets.connect(WS_URI) as websocket:
        output_stream = defaultdict(str)
        # Run WF
        wf_run_event = StreamingEventMessage(entity_id=WF_ID, data=INPUT)
        await websocket.send(wf_run_event.to_json())
        wf_runs += 1

        # Keep receiving messages from the server
        try:
            while True:
                event_raw = await websocket.recv()
                # await websocket.close()
                try:
                    event = StreamingEventMessage.model_validate_json(event_raw)
                except Exception as e:
                    logger.error(f"Error while parsing message. Error: {e}")

                message = event.data
                if event.entity_id == WF_ID:
                    logger.info(f"Final output {event.entity_id}: {message}")
                    if wf_runs >= WF_RUNS:
                        break

                    # Run WF again
                    wf_run_event = StreamingEventMessage(entity_id=WF_ID, data=INPUT)
                    await websocket.send(wf_run_event.to_json())
                    wf_runs += 1

                elif event.event == HF_NODE_STREAMING_EVENT:
                    # Send a user input to the server back
                    hf_user_content = input(event.data["prompt"])
                    hf_response_event = HFStreamingInputEventMessage(
                        wf_run_id=event.wf_run_id,
                        entity_id=event.entity_id,
                        data=HFStreamingInputEventMessageData(content=hf_user_content),
                        event=event.event,
                    )
                    await websocket.send(hf_response_event.to_json())
                    logger.info(f"Streaming {event.wf_run_id} - {event.entity_id}: {hf_user_content}")
                else:
                    if content := message["choices"][0]["delta"]["content"]:
                        output_stream[event.entity_id] += content
                        logger.info(
                            f"Streaming {event.wf_run_id} - {event.entity_id}: {output_stream[event.entity_id]}"
                        )

        except websockets.ConnectionClosed:
            logger.error("WebSocket connection closed by the server")

        await websocket.close()


if __name__ == "__main__":
    asyncio.run(websocket_client())
