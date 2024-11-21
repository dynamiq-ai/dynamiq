import textwrap

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

# Please use your own file path
OUTPUT_FILE_PATH = "trip.md"


def _validate_input_data(input_data):
    """
    Validate the input data dictionary for required keys.

    Args:
        input_data (dict): The input data dictionary to validate

    Raises:
        KeyError: If any required key is missing from the input_data dictionary
    """
    required_keys = ["dates", "location", "cities", "interests"]
    for key in required_keys:
        if key not in input_data:
            raise KeyError(f"Missing required key: {key}")


def _format_prompt(template, input_data):
    """
    Format the prompt template with input data.

    Args:
        template (str): The prompt template string
        input_data (dict): The input data dictionary

    Returns:
        str: The formatted prompt string
    """
    return textwrap.dedent(template).format(**input_data)


def generate_customer_prompt(input_data):
    """
    Generate a detailed customer prompt for comprehensive trip planning.

    This function creates a prompt that instructs to analyze and select the best city,
    compile an in-depth city guide, and create a 7-day travel itinerary.

    Args:
        input_data (dict): A dictionary containing trip information with the following keys:
            - dates (str): The dates of the trip
            - location (str): The traveler's starting location
            - cities (str): A list of potential city options
            - interests (str): The traveler's interests

    Returns:
        str: A formatted string containing the detailed customer prompt

    Raises:
        KeyError: If any required key is missing from the input_data dictionary
    """
    _validate_input_data(input_data)

    template = """
    Analyze and select the best city for the trip based on weather patterns, seasonal events, and travel costs.
    Compare multiple cities considering current weather, upcoming events, and travel expenses. Provide a detailed
    report on the chosen city, including flight costs, weather forecast, and attractions.

    Next, compile an in-depth city guide with key attractions, local customs, events, and daily activity recommendations.
    Include hidden gems, cultural hotspots, must-visit landmarks, weather forecasts, and costs. The guide should be rich
    in cultural insights and practical tips to enhance the travel experience.

    Finally, expand the guide into a 7-day travel itinerary with detailed daily plans, including weather forecasts,
    places to eat, packing suggestions, and a budget breakdown. Suggest actual places to visit, hotels, and restaurants.
    The itinerary should cover all aspects of the trip, from arrival to departure, with a daily schedule, recommended
    clothing, items to pack, and a detailed budget.

    Trip Date: {dates}
    Traveling from: {location}
    City Options: {cities}
    Traveler Interests: {interests}
    """  # noqa: E501

    return _format_prompt(template, input_data)


def run_workflow(input_data: dict) -> dict:
    """Runs workflow"""
    llm_agent = setup_llm()
    http_connection_serp = ScaleSerp()
    tool_search = ScaleSerpTool(connection=http_connection_serp)

    # Create agents
    agent_selection_city = ReActAgent(
        name="City Selection Expert",
        role="An expert in analyzing travel data to pick ideal destinations",
        goal=(
            "help select the best city for a trip based on specific criteria such as weather patterns, seasonal events, and travel costs."  # noqa: E501
        ),
        llm=llm_agent,
        tools=[tool_search],
        max_loops=15,
    )

    agent_city_guide = ReActAgent(
        name="City Guide Expert",
        role="An expert in gathering information about a city",
        goal=(
            "compile an in-depth guide for someone traveling to a city, including key attractions, local customs, special events, and daily activity recommendations."  # noqa: E501
        ),
        llm=llm_agent,
        tools=[tool_search],
        max_loops=15,
    )

    agent_writer = Agent(
        name="City Guide Writer",
        role="An expert in creating detailed travel guides",
        goal="write a detailed travel guide for a city, including key attractions, local customs, special events, and daily activity recommendations.",  # noqa: E501
        llm=llm_agent,
    )

    agent_manager = GraphAgentManager(
        llm=llm_agent,
    )

    # Create a linear orchestrator
    orchestrator = GraphOrchestrator(
        manager=agent_manager,
        final_summarizer=True,
    )

    orchestrator.add_node("select_city", [agent_selection_city])
    orchestrator.add_node("gather_city_information", [agent_city_guide])
    orchestrator.add_node("document_infromation", [agent_writer])

    orchestrator.add_edge(START, "select_city")
    orchestrator.add_edge("select_city", "gather_city_information")
    orchestrator.add_edge("gather_city_information", "document_infromation")
    orchestrator.add_edge("document_infromation", END)

    # Create a workflow
    workflow = Workflow(flow=Flow(nodes=[orchestrator]))

    user_prompt = generate_customer_prompt(input_data)

    tracing = TracingCallbackHandler()

    result = workflow.run(
        input_data={
            "input": user_prompt,
        },
        config=RunnableConfig(callbacks=[tracing]),
    )
    logger.info("Workflow completed")
    return result.output[orchestrator.id]["output"]["content"], tracing.runs


if __name__ == "__main__":
    user_location = input("Enter your location: ")
    user_cities = input("Enter cities you want to visit: ")
    user_dates = input("Enter dates: ")
    user_interests = input("Enter your interests: ")
    input_data = {
        "location": user_location,
        "cities": user_cities,
        "dates": user_dates,
        "interests": user_interests,
    }
    content, _ = run_workflow(input_data)
    print(content)
    with open(OUTPUT_FILE_PATH, "w") as f:
        f.write(content)
