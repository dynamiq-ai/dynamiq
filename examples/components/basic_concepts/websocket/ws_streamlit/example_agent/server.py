import os
import uuid

from fastapi import FastAPI, WebSocket

from dynamiq import Workflow, connections, flows
from dynamiq.connections import ScaleSerp
from dynamiq.nodes import llms
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.node import StreamingConfig
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from examples.components.basic_concepts.websocket.ws_server_fastapi import WorkflowWSHandler

app = FastAPI()

HOST = "127.0.0.1"
PORT = 6050
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WF_ID = "9cd3e052-6af8-4e89-9e88-5a654ec9c492"


OPENAI_CONNECTION = connections.OpenAI(
    id=str(uuid.uuid4()),
    api_key=OPENAI_API_KEY,
)
OPENAI_NODE_STREAMING_EVENT = "streaming-openai-1"
OPENAI_NODE = llms.OpenAI(
    name="OpenAI",
    model="gpt-4o-mini",
    connection=OPENAI_CONNECTION,
    streaming=StreamingConfig(enabled=True, event=OPENAI_NODE_STREAMING_EVENT),
)

tool_search = ScaleSerpTool(connection=ScaleSerp())

agent = ReActAgent(
    name="ReAct Agent - Children Teacher",
    id="react",
    llm=OPENAI_NODE,
    tools=[tool_search],
)


@app.websocket("/workflows/test")
async def websocket_endpoint(websocket: WebSocket):
    wf = Workflow(id=WF_ID, flow=flows.Flow(nodes=[agent]))
    ws_handler = WorkflowWSHandler(workflow=wf, websocket=websocket)
    await ws_handler.handle()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
