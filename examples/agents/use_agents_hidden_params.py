from dynamiq.connections import Http as HttpConnection
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm


def setup_react_agent_with_apis() -> ReActAgent:
    """
    Set up and return a ReAct agent with two API tools.

    Returns:
        ReActAgent: Configured ReAct agent.
    """
    llm = setup_llm()

    cat_connection = HttpConnection(
        method="GET",
        url="https://catfact.ninja/fact",
    )

    cat_api = HttpApiCall(
        id="cat-facts-api-456",
        connection=cat_connection,
        success_codes=[200, 201],
        timeout=60,
        response_type=ResponseType.JSON,
        params={"limit": 10},
        name="CatFactApi",
        description="Gets a random cat fact from the CatFact API",
    )

    dog_connection = HttpConnection(
        method="GET",
        url="https://catfact.ninja/fact",
    )

    dog_api = HttpApiCall(
        id="dog-facts-api-789",
        connection=dog_connection,
        success_codes=[200, 201],
        timeout=60,
        response_type=ResponseType.JSON,
        params={"limit": 10},
        name="DogFactApi",
        description="Gets a random dog fact (using cat API for demo purposes)",
    )

    agent = ReActAgent(
        name="AI Agent",
        llm=llm,
        tools=[cat_api, dog_api],
        role="is to help users retrieve interesting animal facts",
        inference_mode=InferenceMode.DEFAULT,
    )

    return agent


def run_agent_with_tokens():
    agent = setup_react_agent_with_apis()

    cat_fact_token = "cat_api_token_12345"  # nosec B105
    dog_fact_token = "dog_api_key_67890"  # nosec B105

    result = agent.run(
        input_data={
            "input": "Get me a cat fact using the CatFactApi tool, "
            "then find and return a dog fact using the DogFactApi tool.",
            "tool_params": {
                "global": {"timeout": 30},
                "CatFactApi": {"headers": {"Authorization": f"Bearer {cat_fact_token}"}},
                "dog-facts-api-789": {"headers": {"Authorization": f"Bearer {dog_fact_token}"}},
            },
        }
    )

    print("\nAgent Final Result:")
    print(result.output.get("content"))


if __name__ == "__main__":
    run_agent_with_tokens()
