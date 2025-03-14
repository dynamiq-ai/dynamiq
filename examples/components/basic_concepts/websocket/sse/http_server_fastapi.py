import asyncio
import logging

from fastapi import FastAPI
from sse_starlette import EventSourceResponse, ServerSentEvent

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.flows import Flow
from dynamiq.runnables import RunnableConfig
from examples.components.basic_concepts.websocket.ws_server_fastapi import OPENAI_NODE, WF_ID

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
    wf: Workflow,
    wf_input: dict,
    streaming_handler: AsyncStreamingIteratorCallbackHandler,
    tracing_handler: TracingCallbackHandler,
):
    asyncio.get_running_loop().run_in_executor(
        None,
        wf.run,
        wf_input,
        RunnableConfig(callbacks=[streaming_handler, tracing_handler]),
    )


@app.post("/workflow/run")
async def wf_run(wf_data: dict):
    logger.info("Workflow run request received")

    streaming_handler = AsyncStreamingIteratorCallbackHandler()
    tracing_handler = TracingCallbackHandler()
    wf = Workflow(id=WF_ID, flow=Flow(nodes=[OPENAI_NODE]))
    await asyncio.create_task(run_wf_async(wf, wf_data, streaming_handler, tracing_handler))
    return EventSourceResponse(send_stream_events_by_sse(streaming_handler))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
