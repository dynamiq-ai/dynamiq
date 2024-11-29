from dotenv import load_dotenv
from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Anthropic, ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators import LinearOrchestrator
from dynamiq.nodes.agents.orchestrators.linear_manager import LinearAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms import Anthropic as AnthropicLLM
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.utils.logger import logger
from examples.tools.file_reader import FileReaderTool

# Load environment variables
load_dotenv()
AGENT_RESEARCHER_ROLE = (
    "An expert in gathering information about a job."
    "Goal is to analyze the company website and provided description"
    "to extract insights on culture, values, and specific needs."
)
AGENT_WRITER_ROLE = (
    "An expert in creating job descriptions."
    "Goal is to craft a detailed, engaging, "
    "and enticing job posting that resonates "
    "with the company's values and attracts the right candidates."
)
AGENT_REVIEWER_ROLE = (
    "An expert in reviewing and editing content."
    "Goal is to ensure the job description is accurate, "
    "engaging, and aligned with the company's values and needs."
)


def create_workflow() -> Workflow:
    """
    Create the workflow with all necessary agents and tools.

    Returns:
        Workflow: The configured workflow.
    """
    # Initialize connections
    anthropic_connection = Anthropic()
    search_connection = ScaleSerp()

    # Initialize LLM
    llm = AnthropicLLM(
        connection=anthropic_connection,
        model="claude-3-5-sonnet-20240620",
        temperature=0.5,
        max_tokens=4000,
    )

    # Initialize tools
    tool_search = ScaleSerpTool(connection=search_connection)
    tool_file_read = FileReaderTool(file_path="job_example.md")

    # Create agents
    agent_researcher = ReActAgent(
        name="Researcher Analyst",
        role=AGENT_RESEARCHER_ROLE,
        llm=llm,
        tools=[tool_search],
    )
    agent_writer = ReActAgent(
        name="Job Description Writer",
        role=AGENT_WRITER_ROLE,
        llm=llm,
        tools=[tool_file_read, tool_search],
    )
    agent_reviewer = ReActAgent(
        name="Job Description Reviewer and Editor",
        role=AGENT_REVIEWER_ROLE,
        llm=llm,
        tools=[tool_search, tool_file_read],
    )
    agent_manager = LinearAgentManager(llm=llm)

    # Create agent orchestrator
    linear_orchestrator = LinearOrchestrator(
        manager=agent_manager,
        agents=[agent_researcher, agent_writer, agent_reviewer],
        final_summarizer=True,
    )

    return Workflow(
        flow=Flow(nodes=[linear_orchestrator]),
    )


def run_planner() -> tuple[str, dict]:
    # Create workflow
    workflow = create_workflow()

    # Prepare input data
    company_link = "getdynamiq.ai"
    company_domain = "ai, gen ai, llms, it"
    hiring_needs = "llm engineer, with solid experience"
    specific_benefits = "holidays, stocks, insurance"

    user_prompt = f"""
    Analyze the company's culture, values, and mission from its website and description.
    Understand the hiring needs for the specific role and identify the key skills, experiences, and qualities the ideal candidate should possess.
    Draft a job posting that includes an engaging introduction, detailed role description, responsibilities, required skills, and qualifications.
    Ensure the tone aligns with the company's culture and values. Highlight any unique benefits or opportunities offered by the company to attract the right candidates.
    Review and refine the draft for clarity, engagement, and grammatical accuracy, providing feedback for any necessary revisions.
    Format the final job posting in markdown.
    Here is company information:
    Company Link: {company_link}
    Company Domain: {company_domain}
    Hiring Needs: {hiring_needs}
    Specific Benefits: {specific_benefits}
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
        output = result.output[workflow.flow.nodes[0].id]['output']['content']['result']
        print(output)
        with open("job_generated.md", "w") as f:
            f.write(output)
        return output, tracing.runs

    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {str(e)}")
        return "", {}


if __name__ == "__main__":
    print(run_planner()[0])
