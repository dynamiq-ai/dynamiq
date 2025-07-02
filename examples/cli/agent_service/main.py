import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query

from dynamiq.connections import Exa
from dynamiq.connections import TogetherAI as TogetherAIConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.togetherai import TogetherAI
from dynamiq.nodes.tools.exa_search import ExaTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.utils.logger import logger

load_dotenv()

# Ensure required env vars are present
required_env = ["TOGETHER_API_KEY", "EXA_API_KEY"]
for key in required_env:
    if not os.getenv(key):
        raise OSError(f"Missing required environment variable: {key}")

connection_exa = Exa()
tool_search = ExaTool(connection=connection_exa)
llm = TogetherAI(
    connection=TogetherAIConnection(),
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0,
    max_tokens=4000,
)
agent = ReActAgent(
    name="Agent",
    id="Agent",
    llm=llm,
    tools=[tool_search],
    inference_mode=InferenceMode.XML,
)

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Dynamiq test service is running."}


@app.get("/search")
def search(query: str = Query(..., description="Your search prompt")):
    try:
        result = agent.run(input_data={"input": query})
        content = result.output.get("content")
        logger.info("RESULT")
        logger.info(content)
        return {"result": content}
    except Exception as e:
        logger.error(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
