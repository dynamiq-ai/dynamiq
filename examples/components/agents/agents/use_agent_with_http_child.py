import json

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Http as HttpConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

CHILD_ROLE = """
You are an API research assistant.
Use the available HTTP tools to gather interesting animal facts.
Always call a tool before replying.
"""

PARENT_ROLE = """
You are the manager. Delegate API lookups to the Research Agent tool.
Summarise the results for the user once the research concludes.
"""


def make_http_tools():
    cat_connection = HttpConnection(method="GET", url="https://catfact.ninja/fact")
    dog_connection = HttpConnection(method="GET", url="https://catfact.ninja/fact")

    cat_api = HttpApiCall(
        id="cat-facts-api-456",
        name="CatFactApi",
        description="Gets a random cat fact from CatFact API",
        connection=cat_connection,
        response_type=ResponseType.JSON,
        success_codes=[200, 201],
        timeout=30,
        params={"limit": 10},
    )

    dog_api = HttpApiCall(
        id="dog-facts-api-789",
        name="DogFactApi",
        description="Gets a random dog fact (proxying cat API for demo purposes)",
        connection=dog_connection,
        response_type=ResponseType.JSON,
        success_codes=[200, 201],
        timeout=30,
        params={"limit": 10},
    )

    return cat_api, dog_api


def make_child_agent(llm):
    cat_api, dog_api = make_http_tools()

    return Agent(
        name="Research Agent",
        description="Calls HTTP APIs to gather animal facts.",
        role=CHILD_ROLE,
        llm=llm,
        tools=[cat_api, dog_api],
        max_loops=3,
        inference_mode=InferenceMode.XML,
    )


def make_parent_agent(llm, child_agent):
    return Agent(
        name="Manager Agent",
        description="Delegates API lookups to the Research Agent and summarises results.",
        role=PARENT_ROLE,
        llm=llm,
        tools=[child_agent],
        max_loops=3,
        inference_mode=InferenceMode.XML,
    )


def run_workflow():
    llm = setup_llm()
    child = make_child_agent(llm)
    parent = make_parent_agent(llm, child)

    tracing = TracingCallbackHandler()
    workflow = Workflow(flow=Flow(nodes=[parent]))

    input_data = {
        "input": "Fetch one cat fact and one dog fact, then summarise them for me.",
        "user_id": "http-demo-user",
        "tool_params": {
            "global": {"metadata": {"request_id": "cats-vs-dogs"}},
            "by_name": {
                "Research Agent": {
                    "tool_params": {
                        "global": {"metadata": {"caller": "manager"}},
                        "by_name": {
                            "CatFactApi": {"headers": {"Authorization": "Bearer cat_api_token"}},  # nosec B105
                        },
                        "by_id": {
                            "dog-facts-api-789": {"headers": {"X-API-Key": "dog_api_key"}},  # nosec B105
                        },
                    }
                },
            },
        },
    }

    logger.info("Manager input tool_params: %s", input_data["tool_params"])

    result = workflow.run(
        input_data=input_data,
        config=RunnableConfig(callbacks=[tracing]),
    )

    json.dumps({"runs": [run.to_dict() for run in tracing.runs.values()]}, cls=JsonWorkflowEncoder)

    content = result.output[parent.id]["output"]["content"]
    return content, tracing.runs


if __name__ == "__main__":
    output, _ = run_workflow()
    logger.info("=== AGENT OUTPUT ===")
    logger.info(output)
