import asyncio
import logging
import os
import uuid
from queue import Queue
from threading import Event

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, ConfigDict

from dynamiq import Workflow, callbacks, connections, flows, prompts
from dynamiq.callbacks.streaming import AsyncStreamingIteratorCallbackHandler
from dynamiq.nodes import llms
from dynamiq.nodes.node import NodeDependency
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool, InputMethod
from dynamiq.runnables import RunnableConfig
from dynamiq.runnables.base import NodeRunnableConfig
from dynamiq.types.streaming import StreamingConfig, StreamingEventMessage

app = FastAPI()


logger = logging.getLogger(__name__)


HOST = "127.0.0.1"
PORT = 6000
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


OPENAI_CONNECTION = connections.OpenAI(
    id=str(uuid.uuid4()),
    api_key=OPENAI_API_KEY,
)
OPENAI_NODE_STREAMING_EVENT = "streaming-openai-1"
OPENAI_NODE = llms.OpenAI(
    name="OpenAI",
    model="gpt-3.5-turbo",
    connection=OPENAI_CONNECTION,
    prompt=prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is AI?",
            ),
        ],
    ),
    streaming=StreamingConfig(enabled=True, event=OPENAI_NODE_STREAMING_EVENT),
)
OPENAI_2_NODE_STREAMING_EVENT = "streaming-openai-2"
OPENAI_2_NODE = llms.OpenAI(
    name="OpenAI2",
    model="gpt-3.5-turbo",
    connection=OPENAI_CONNECTION,
    prompt=prompts.Prompt(
        messages=[
            prompts.Message(
                role="user",
                content="What is Data Science?",
            ),
        ],
    ),
    streaming=StreamingConfig(enabled=True, event=OPENAI_2_NODE_STREAMING_EVENT),
)
HF_NODE_STREAMING_EVENT = "streaming-hf"
HF_NODE = HumanFeedbackTool(
    input_method=InputMethod.stream,
    streaming=StreamingConfig(enabled=True, event=HF_NODE_STREAMING_EVENT, timeout=15),
    depends=[NodeDependency(node=OPENAI_NODE)],
)
WF_ID = "9cd3e052-6af8-4e89-9e88-5a654ec9c492"


class WorkflowRunConnector(BaseModel):
    wf_id: str | uuid.UUID
    queue: Queue | asyncio.Queue
    done_event: Event | asyncio.Event

    model_config = ConfigDict(arbitrary_types_allowed=True)


class WorkflowRunNodeConnector(WorkflowRunConnector):
    node_id: str | uuid.UUID


class WorkflowWSHandler:
    def __init__(self, workflow: Workflow, websocket: WebSocket):
        # Set up input streaming queues and done events
        self.wf = workflow
        self.ws = websocket
        self.wf_run_connectors = {}
        self.wf_run_node_connectors = {}

    def _create_wf_run_node_connector(self, ws_event: StreamingEventMessage) -> dict[str, WorkflowRunNodeConnector]:
        wf_run_node_connector = {}
        if ws_event.entity_id == self.wf.id:
            wf_run_node_connector = {
                node.id: WorkflowRunNodeConnector(wf_id=self.wf.id, node_id=node.id, queue=Queue(), done_event=Event())
                for node in self.wf.flow.nodes
                if node.streaming.enabled
            }
        return wf_run_node_connector

    def _create_wf_run_connector(self, ws_event: StreamingEventMessage) -> WorkflowRunConnector:
        if ws_event.entity_id == self.wf.id:
            return WorkflowRunConnector(
                wf_id=self.wf.id,
                queue=asyncio.Queue(),
                done_event=asyncio.Event(),
            )

    def _run_wf(self, wf_data: dict, config: RunnableConfig):
        try:
            logger.info("Start workflow run")
            return self.wf.run(input_data=wf_data, config=config).output
        except Exception as e:
            logger.error(f"Error in workflow run. Error: {e}")

    async def _send_stream_events_by_ws(
        self, websocket: WebSocket, streaming_handler: AsyncStreamingIteratorCallbackHandler
    ):
        try:
            async for event in streaming_handler:
                await websocket.send_text(event.to_json())
            logger.info("All streaming events sent")
        except WebSocketDisconnect as e:
            logger.error(f"WebSocket disconnected. Error: {e}")
        except asyncio.CancelledError:
            logger.error("Task cancelled")
        except Exception as e:
            logger.error(f"Unexpected error. Error: {e}")
        finally:
            streaming_handler.done_event.set()

    async def _parse_ws_event(self, ws_data: str) -> StreamingEventMessage | None:
        try:
            return StreamingEventMessage.model_validate_json(ws_data)
        except ValueError as e:
            logger.error(f"Error parsing WebSocket message: {e}")
            return

    async def _process_wf_run_event(self, ws_event: StreamingEventMessage):
        logger.info("Workflow run event received")

        wf_run_connector = self._create_wf_run_connector(ws_event)
        wf_run_node_connectors = self._create_wf_run_node_connector(ws_event)
        self.wf_run_connectors[ws_event.wf_run_id] = wf_run_connector
        self.wf_run_node_connectors[ws_event.wf_run_id] = wf_run_node_connectors

        streaming_handler = AsyncStreamingIteratorCallbackHandler(
            queue=wf_run_connector.queue, done_event=wf_run_connector.done_event
        )
        tracing_handler = callbacks.TracingCallbackHandler(trace_id=ws_event.wf_run_id)

        asyncio.create_task(self._send_stream_events_by_ws(self.ws, streaming_handler))
        await asyncio.sleep(0.01)

        # Create runnable config with streaming connector for nodes
        run_config = RunnableConfig(
            run_id=ws_event.wf_run_id,
            callbacks=[tracing_handler, streaming_handler],
            nodes_override={
                node_id: NodeRunnableConfig(
                    streaming=StreamingConfig(
                        enabled=True,
                        input_queue=connector.queue,
                        input_queue_done_event=connector.done_event,
                    )
                )
                for node_id, connector in wf_run_node_connectors.items()
            },
        )

        # Run Workflow in executor. Final workflow output also sent via WS
        asyncio.get_running_loop().run_in_executor(None, self._run_wf, ws_event.data, run_config)
        logger.info("Workflow run event processed successfully")

    async def _process_node_input_streaming_event(self, ws_event: StreamingEventMessage):
        logger.info("Node input streaming event received")
        wf_run_node_connector = self.wf_run_node_connectors[ws_event.wf_run_id][ws_event.entity_id]
        wf_run_node_connector.queue.put_nowait(ws_event.to_json())
        logger.info("Node input streaming event processed successfully")

    async def _process_ws_event(self, ws_event: StreamingEventMessage):
        if ws_event.entity_id == self.wf.id:
            await self._process_wf_run_event(ws_event)
        elif ws_event.entity_id in self.wf_run_node_connectors.get(ws_event.wf_run_id, {}):
            await self._process_node_input_streaming_event(ws_event)
        else:
            logger.warning(f"Unhandled event: {ws_event}")

    async def _cleanup(self):
        logger.info("Cleanup of resources started")
        for wf_run_connector in self.wf_run_connectors.values():
            wf_run_connector.done_event.set()
        for wf_run_node_connector in self.wf_run_node_connectors.values():
            for node_connector in wf_run_node_connector.values():
                node_connector.done_event.set()

        logger.info("Cleanup completed")

    async def handle(self):
        await self.ws.accept()
        try:
            while True:
                ws_data = await self.ws.receive_text()
                if (ws_event := await self._parse_ws_event(ws_data)) is None:
                    logger.warning(f"Unhandled event data: {ws_data}")
                    continue

                await self._process_ws_event(ws_event)
        except WebSocketDisconnect:
            logger.info("WebSocket disconnected")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
        finally:
            await self._cleanup()


@app.websocket("/workflows/test")
async def websocket_endpoint(websocket: WebSocket):
    wf = Workflow(id=WF_ID, flow=flows.Flow(nodes=[OPENAI_NODE, HF_NODE]))
    ws_handler = WorkflowWSHandler(workflow=wf, websocket=websocket)
    await ws_handler.handle()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
