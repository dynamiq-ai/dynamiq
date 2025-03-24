from dynamiq.connections import Http as HttpConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.tools.http_api_call import HttpApiCall, ResponseType
from dynamiq.nodes.types import InferenceMode
from examples.llm_setup import setup_llm


def setup_react_agent_with_apis() -> Agent:
    """
    Set up and return an AI agent configured with two API tools for retrieving animal facts.

    Returns:
        Agent: The configured agent with cat and dog fact API tools.
    """
    llm = setup_llm()

    # Configure the Cat Fact API tool
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

    # Configure the Dog Fact API tool (using cat API endpoint for demo purposes)
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

    # Create the AI agent with both API tools
    agent = Agent(
        name="AI Agent",
        llm=llm,
        tools=[cat_api, dog_api],
        role="is to help users retrieve interesting animal facts",
        inference_mode=InferenceMode.DEFAULT,
    )
    return agent


def run_agent_with_tokens():
    """
    Run the agent with API tokens provided in the tool parameters.

    This function sets up the agent, provides authentication tokens for the tools,
    executes a request for both a cat fact and a dog fact, and prints the final output.
    """
    agent = setup_react_agent_with_apis()

    cat_fact_token = "cat_api_token_12345"  # nosec B105
    dog_fact_token = "dog_api_key_67890"  # nosec B105

    input_data = {
        "input": "Get me a cat fact and a dog fact",
        "tool_params": {
            "global": {"timeout": 30},
            "by_name": {"CatFactApi": {"headers": {"Authorization": f"Bearer {cat_fact_token}"}}},
            "by_id": {"dog-facts-api-789": {"headers": {"Authorization": f"Bearer {dog_fact_token}"}}},
        },
    }

    result = agent.run(input_data=input_data)

    print("\nAgent Final Result:")
    print(result.output.get("content"))


if __name__ == "__main__":
    run_agent_with_tokens()
