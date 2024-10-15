from typing import Literal

from dotenv import load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators.graph import BaseContext, GraphOrchestrator
from dynamiq.nodes.agents.orchestrators.graph_manager import GraphAgentManager
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

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

    graph_orchestrator.add_node("fetch_user_info", fetch_user_info)
    graph_orchestrator.add_conditional_edge("fetch_user_info", route_to_workflow)

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
