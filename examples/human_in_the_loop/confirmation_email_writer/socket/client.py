import asyncio
import json
import logging

import websockets

from dynamiq.types.feedback import APPROVAL_EVENT
from dynamiq.types.streaming import StreamingEventMessage
from examples.human_in_the_loop.confirmation_email_writer.socket.socket_agent import HOST, PORT, WF_ID, SocketMessage

logger = logging.getLogger(__name__)

WS_URI = f"ws://{HOST}:{PORT}/ws"


async def websocket_client():
    async with websockets.connect(WS_URI) as websocket:
        input_query = input("Provide email details: ")
        wf_run_event = SocketMessage(type="run", content=input_query)
        await websocket.send(wf_run_event.to_json())

        try:
            while True:
                event_raw = json.loads(await websocket.recv())

                if event_raw["event"] == "approval":
                    feedback = input(event_raw["data"])
                    feedback = StreamingEventMessage(entity_id=WF_ID, data=feedback, event=APPROVAL_EVENT).to_json()

                else:
                    feedback = input(event_raw["data"]["prompt"])
                    feedback = StreamingEventMessage(entity_id=WF_ID, data={"content": feedback}).to_json()
                await websocket.send(SocketMessage(type="message", content=feedback).to_json())

                if "finish" in event_raw["data"] and event_raw["data"]["finish"]:
                    break

        except websockets.ConnectionClosed:
            logger.error("WebSocket connection closed by the server")

        await websocket.close()


if __name__ == "__main__":
    asyncio.run(websocket_client())
