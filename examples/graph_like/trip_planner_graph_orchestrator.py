from dynamiq import Workflow
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import ScaleSerp  # ZenRows
from dynamiq.flows import Flow
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.graph import END, START, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.anthropic import Anthropic
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.utils.logger import logger
from examples.trip_planner.prompts import generate_customer_prompt

# Please use your own file path
OUTPUT_FILE_PATH = "trip.md"


def choose_provider(model_type, model_name):
    if model_type == "gpt":
        _connection = OpenAIConnection()
        _llm = OpenAI(
            connection=_connection,
            model=model_name,
            temperature=0.1,
            max_tokens=4000,
        )
    elif model_type == "claude":
        _connection = AnthropicConnection()
        _llm = Anthropic(
            connection=_connection,
            model=model_name,
            temperature=0.1,
            max_tokens=4000,
        )
    else:
        raise ValueError("Invalid model provider specified.")
    return _llm


def inference(input_data: dict, model_type="gpt", model_name="gpt-4o-mini") -> dict:
    llm_agent = choose_provider(model_type, model_name)
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
        max_loops=10,
    )

    agent_city_guide = ReActAgent(
        name="City Guide Expert",
        role="An expert in gathering information about a city",
        goal=(
            "compile an in-depth guide for someone traveling to a city, including key attractions, local customs, special events, and daily activity recommendations."  # noqa: E501
        ),
        llm=llm_agent,
        tools=[tool_search],
        max_loops=10,
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

    print(user_prompt)
    result = workflow.run(
        input_data={
            "input": user_prompt,
        }
    )
    logger.info("Workflow completed")
    content = result.output[orchestrator.id]
    return content


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
    content = inference(input_data)["output"]["content"]
    print(content)
    with open(OUTPUT_FILE_PATH, "w") as f:
        f.write(content)
