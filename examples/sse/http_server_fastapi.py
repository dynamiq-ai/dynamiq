import asyncio
import logging

from fastapi import FastAPI
from sse_starlette import EventSourceResponse, ServerSentEvent

from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from examples.ws.ws_server_fastapi import run_wf

app = FastAPI()


logger = logging.getLogger(__name__)


HOST = "127.0.0.1"
PORT = 6001


async def send_stream_events_by_sse(
    streaming_handler: AsyncStreamingIteratorCallbackHandler,
):
    try:
        async for event in streaming_handler:
            yield ServerSentEvent(data=event.to_json(), event=event.event)
        logger.info("All streaming events sent")
    except asyncio.CancelledError:
        logger.error("Task cancelled")


async def run_wf_async(
    wf_data: dict, streaming_handler: AsyncStreamingIteratorCallbackHandler
):
    await asyncio.get_running_loop().run_in_executor(
        None, run_wf, wf_data, streaming_handler
    )


@app.post("/workflow/run")
async def wf_run(wf_data: dict):
    logger.info("Workflow run request received")

    streaming_handler = AsyncStreamingIteratorCallbackHandler()

    await asyncio.create_task(run_wf_async(wf_data, streaming_handler))
    await asyncio.sleep(0.01)

    return EventSourceResponse(send_stream_events_by_sse(streaming_handler))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
