from dotenv import load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Tavily as TavilyConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators import LinearOrchestrator
from dynamiq.nodes.agents.orchestrators.linear_manager import LinearAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.tools import TavilyTool
from dynamiq.runnables import RunnableConfig
from dynamiq.types.feedback import PlanApprovalConfig
from dynamiq.utils.logger import logger
from examples.llm_setup import setup_llm

# Load environment variables
load_dotenv()

AGENT_RESEARCHER_ROLE = (
    "An expert in gathering information about a job. "
    "The goal is to analyze the company website and the provided description "
    "to extract insights on culture, values, and specific needs."
)

AGENT_WRITER_ROLE = (
    "An expert in creating job descriptions. "
    "The goal is to craft a detailed, engaging, and enticing job posting "
    "that resonates with the company's values and attracts the right candidates."
)

AGENT_REVIEWER_ROLE = (
    "An expert in reviewing and editing content. "
    "The goal is to ensure the job description is accurate, engaging, "
    "and aligned with the company's values and needs."
)


def create_workflow() -> Workflow:
    """
    Create the workflow with all necessary agents and tools.

    Returns:
        Workflow: The configured workflow.
    """
    llm = setup_llm()

    search_connection = TavilyConnection(api_key="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")
    tool_search = TavilyTool(connection=search_connection)

    agent_researcher = ReActAgent(name="Researcher Analyst", role=AGENT_RESEARCHER_ROLE, llm=llm, tools=[tool_search])
    agent_writer = ReActAgent(
        name="Job Description Writer",
        role=AGENT_WRITER_ROLE,
        llm=llm,
    )
    agent_reviewer = ReActAgent(
        name="Job Description Reviewer and Editor",
        role=AGENT_REVIEWER_ROLE,
        llm=llm,
    )
    agent_manager = LinearAgentManager(llm=llm)

    linear_orchestrator = LinearOrchestrator(
        manager=agent_manager,
        agents=[agent_researcher, agent_writer, agent_reviewer],
        final_summarizer=True,
        plan_approval=PlanApprovalConfig(enabled=True),
    )

    return Workflow(
        flow=Flow(nodes=[linear_orchestrator]),
    )


def run_planner() -> tuple[str, dict]:
    workflow = create_workflow()

    user_prompt = "Analyze the Google's company culture, values, and mission."  # noqa: E501

    tracing = TracingCallbackHandler()
    try:
        result = workflow.run(
            input_data={"input": user_prompt},
            config=RunnableConfig(callbacks=[tracing]),
        )

        logger.info("Workflow completed successfully")

        output = result.output[workflow.flow.nodes[0].id]["output"]["content"]
        print(output)

        return output, tracing.runs

    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {str(e)}")
        return "", {}


if __name__ == "__main__":
    run_planner()
