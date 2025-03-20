import asyncio
import logging
from collections import defaultdict

import httpx
from httpx_sse import aconnect_sse

from dynamiq.types.streaming import StreamingEventMessage
from examples.components.core.websocket.sse.http_server_fastapi import HOST, PORT
from examples.components.core.websocket.ws_server_fastapi import (
    OPENAI_2_NODE_STREAMING_EVENT,
    OPENAI_NODE_STREAMING_EVENT,
)

logger = logging.getLogger(__name__)


WF_URL = f"http://{HOST}:{PORT}/workflow/run"
INPUT = {"a": 1, "b": 2}


async def http_client():
    output_stream = defaultdict(str)

    async with httpx.AsyncClient() as client:  # nosec
        async with aconnect_sse(
            client=client,
            method="POST",
            url=WF_URL,
            json=INPUT,
            timeout=60,
        ) as event_source:
            async for sse in event_source.aiter_sse():
                try:
                    event = StreamingEventMessage.model_validate_json(sse.data)
                except Exception as e:
                    logger.error(f"Error while parsing message. Error: {e}")

                if sse.event in (
                    OPENAI_NODE_STREAMING_EVENT,
                    OPENAI_2_NODE_STREAMING_EVENT,
                ):
                    message = event.data
                    if message_content := message["choices"][0]["delta"]["content"]:
                        output_stream[event.entity_id] += message_content
                        logger.info(f"Streaming {event.entity_id}: {output_stream[event.entity_id]}")
                else:
                    logger.info(f"Final output {event.entity_id}: {event.data}")


if __name__ == "__main__":
    asyncio.run(http_client())
