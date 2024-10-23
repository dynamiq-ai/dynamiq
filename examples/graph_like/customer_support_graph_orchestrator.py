from typing import Literal

from dotenv import load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators.graph import BaseContext, GraphOrchestrator, END
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm
from dynamiq.nodes.tools.function_tool import function_tool

# Load environment variables
load_dotenv()


class CustomerSupportContext(BaseContext):
    dialog_state: str = ""


def create_workflow() -> Workflow:
    """
    Create the workflow with all necessary agents and tools.

    Returns:
        Workflow: The configured workflow.
    """

    llm = setup_llm()

    llm = llm
    agent_manager = GraphAgentManager(llm=llm)

    graph_orchestrator = GraphOrchestrator(
        manager=agent_manager, final_summarizer=True, context=CustomerSupportContext(), initial_state="fetch_user_info"
    )

    def fetch_user_info():
        return "Use has ticket from Canada to NewYork at 12.9.2024 19:28"

    def route_to_workflow(
        ctx: CustomerSupportContext,
    ) -> Literal[
        "primary_assistant",
        "update_flight",
        "book_car_rental",
        "book_hotel",
        "book_excursion",
    ]:
        """If we are in a delegated state, route directly to the appropriate assistant."""
        dialog_state = ctx.dialog_state
        if not dialog_state:
            return "primary_assistant"
        return dialog_state

    @function_tool
    def udpate_flight_tool(*args, **kwargs):
        return ""
        
    primary_assistant = ReActAgent(
            name="Primary assistant",
            llm=llm,
            role=(
                "You are a helpful customer support assistant for Swiss Airlines. "
                "Your primary role is to search for flight information and company policies to answer customer queries. "
                "If a customer requests to update or cancel a flight, book a car rental, book a hotel, or get trip recommendations, "
                "delegate the task to the appropriate specialized assistant by invoking the corresponding tool. You are not able to make these types of changes yourself."
                " Only the specialized assistants are given permission to do this for the user."
                "The user is not aware of the different specialized assistants, so do not mention them; just quietly delegate through function calls. "
                "Provide detailed information to the customer, and always double-check the database before concluding that information is unavailable. "
                " When searching, be persistent. Expand your query bounds if the first search returns no results. "
                " If a search comes up empty, expand your search before giving up."
            )
            tools=
        )
    
    def primary_assistant_func(state: CustomerSupportContext):
        primary_assistant.run(
                input_data={
                    "input": 'Get stock price for nvidia.'
            },
        )
        tracing = primary_assistant.tracing_final
        return 
        
    primary_assistant.tools = []
    update_flight_assistant = ReActAgent(
        name="Primary assistant",
        llm=llm,
        role=(
            "You are a specialized assistant for handling flight updates. "
            "The primary assistant delegates work to you whenever the user needs help updating their bookings. "
            "Confirm the updated flight details with the customer and inform them of any additional fees. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            "If you need more information or the customer changes their mind, escalate the task back to the main assistant."
            "Remember that a booking isn't completed until after the relevant tool has successfully been used."
        ),
    )

    def create_entry_node(assistant_name: str):
        def entry_node(state: CustomerSupportContext) -> dict:
            return {
                "messages": (
                    f"The assistant is now the {assistant_name}. Reflect on the above conversation between the host assistant and the user."
                    f" The user's intent is unsatisfied. Use the provided tools to assist the user. Remember, you are {assistant_name},"
                    " and the booking, update, other other action is not complete until after you have successfully invoked the appropriate tool."
                    " If the user changes their mind or needs help for other tasks, call the CompleteOrEscalate function to let the primary host assistant take control."
                    " Do not mention who you are - just act as the proxy for the assistant."
                )
            }

        return entry_node

    def route_primary_assistant(state: CustomerSupportContext):
        match state.dialog_state:
            case "update_flight":
                return "enter_update_flight"
        raise ValueError("Invalid route")

    graph_orchestrator.add_node("fetch_user_info", fetch_user_info)
    graph_orchestrator.add_conditional_edge("fetch_user_info", route_to_workflow)

    graph_orchestrator.add_node("primary_assistant", primary_assistant_func)

    graph_orchestrator.add_node("entry_update_flight", create_entry_node("Update flight assistant"))
    graph_orchestrator.add_node("update_flight", update_flight_assistant)

    graph_orchestrator.add_conditional_edge("primary_assistant", route_primary_assistant)

    graph_orchestrator.add_edge("update_flight", END)
    return Workflow(
        flow=Flow(nodes=[graph_orchestrator]),
    )


def run_planner() -> tuple[str, dict]:
    # Create workflow
    workflow = create_workflow()

    user_prompt = """
    Hello
    """  # noqa: E501

    # Run workflow
    tracing = TracingCallbackHandler()
    try:
        result = workflow.run(
            input_data={"input": user_prompt},
            config=RunnableConfig(callbacks=[tracing]),
        )

        logger.info("Workflow completed successfully")

        # Print and save result
        output = result.output[workflow.flow.nodes[0].id]["output"]["content"]

        return output, tracing.runs

    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {str(e)}")
        return "", {}


if __name__ == "__main__":
    output, tracing = run_planner()
    print(output)
