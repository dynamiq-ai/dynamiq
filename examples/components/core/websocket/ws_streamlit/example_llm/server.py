import os
import uuid

from fastapi import FastAPI, WebSocket

from dynamiq import Workflow, connections, flows, prompts
from dynamiq.nodes import llms
from dynamiq.runnables import RunnableConfig
from dynamiq.types.streaming import StreamingConfig
from dynamiq.utils.logger import logger
from examples.components.core.websocket.ws_server_fastapi import WorkflowWSHandler

app = FastAPI()


HOST = "127.0.0.1"
PORT = 6098
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


class WorkflowWSHandlerPrompt(WorkflowWSHandler):
    def _run_wf(self, wf_data: dict, config: RunnableConfig):
        try:
            logger.info("Start workflow run")
            # Create the prompt dynamically
            prompt = prompts.Prompt(
                messages=[
                    prompts.Message(
                        role="system",
                        content="You are a chatbot. Please answer the following questions.",
                    ),
                    prompts.Message(role="user", content=f"What is {wf_data['input']}"),
                ],
            )
            return self.wf.run(input_data=wf_data, config=config, prompt=prompt).output
        except Exception as e:
            logger.error(f"Error in workflow run. Error: {e}")


@app.websocket("/workflows/test")
async def websocket_endpoint(websocket: WebSocket):
    wf = Workflow(id=WF_ID, flow=flows.Flow(nodes=[OPENAI_NODE]))
    ws_handler = WorkflowWSHandlerPrompt(workflow=wf, websocket=websocket)
    await ws_handler.handle()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)
