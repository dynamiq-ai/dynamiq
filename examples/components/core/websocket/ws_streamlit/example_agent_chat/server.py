import os
import uuid

from fastapi import FastAPI, WebSocket

from dynamiq import Workflow, connections, flows
from dynamiq.memory import Memory
from dynamiq.nodes import llms
from dynamiq.nodes.agents.simple import SimpleAgent
from dynamiq.nodes.node import StreamingConfig
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

memory_in_memory = Memory()
AGENT_ROLE = "helpful assistant, goal is to provide useful information and answer questions"
agent = SimpleAgent(
    name="Agent",
    llm=OPENAI_NODE,
    role=AGENT_ROLE,
    id="agent",
    memory=memory_in_memory,
)


@app.websocket("/workflows/test")
async def websocket_endpoint(websocket: WebSocket):
    wf = Workflow(id=WF_ID, flow=flows.Flow(nodes=[agent]))
    ws_handler = WorkflowWSHandler(workflow=wf, websocket=websocket)
    await ws_handler.handle()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
