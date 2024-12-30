import asyncio
import json
import logging

import websockets

from examples.human_in_the_loop.post_writer.server import HOST, PORT, SocketMessage

logger = logging.getLogger(__name__)

WS_URI = f"ws://{HOST}:{PORT}/ws"


async def websocket_client():
    async with websockets.connect(WS_URI) as websocket:
        input_query = input("Provide topic for post: ")
        wf_run_event = SocketMessage(type="run", content=input_query)
        await websocket.send(wf_run_event.to_json())

        try:
            while True:
                event_raw = json.loads(await websocket.recv())

                if "require_feedback" in event_raw["data"] and event_raw["data"]["require_feedback"]:
                    feedback = input(event_raw["data"]["content"])
                    await websocket.send(SocketMessage(type="message", content=feedback).to_json())
                else:
                    logger.info(event_raw["data"]["content"])

                if "finish" in event_raw["data"] and event_raw["data"]["finish"]:
                    break

        except websockets.ConnectionClosed:
            logger.error("WebSocket connection closed by the server")

        await websocket.close()


if __name__ == "__main__":
    asyncio.run(websocket_client())
