"""
This script sets up and runs a multi-agent system for generating a literature overview
on LLM-based Multi-Agent Systems and Frameworks using various LLM providers and tools.
"""

import json

from dotenv import load_dotenv

from dynamiq import Workflow
from dynamiq.callbacks import TracingCallbackHandler
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.connections import ScaleSerp
from dynamiq.flows import Flow
from dynamiq.nodes.agents.base import Agent
from dynamiq.nodes.agents.orchestrators.adaptive import AdaptiveOrchestrator
from dynamiq.nodes.agents.orchestrators.adaptive_manager import AdaptiveAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.anthropic import Anthropic
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.scale_serp import ScaleSerpTool
from dynamiq.runnables import RunnableConfig
from dynamiq.utils import JsonWorkflowEncoder
from dynamiq.utils.logger import logger

# Constants
AGENT_RESEARCHER_ROLE = (
    "The Senior Research Analyst, "
    "specializing in finding the latest and most accurate information. "
    "The goal is to locate the most relevant information on the requested topic and provide it to the user."
)

AGENT_WRITER_ROLE = (
    "The Senior Writer and Editor, "
    "specializing in creating high-quality content. "
    "The goal is to produce high-quality content based on the information provided by the Research Analyst."
)

REACT_AGENT_TEMPERATURE = 0.1
REGULAR_AGENT_TEMPERATURE = 0.1
MAX_TOKENS = 4000

PROMPT = """I need to write a literature overview on the topic
of `LLM-based Multi-Agent Systems and Frameworks` for my research paper.
Use the latest and most relevant information from the internet and scientific articles. Please follow a simple format such as:
- Introduction
- Main Concepts
- Applications
- Conclusion
Also, include the sources at the end of the document. Double-check that the information is up-to-date and relevant.
The final result must be provided in markdown format.
"""  # noqa: E501

# Please use your own file path
OUTPUT_FILE_PATH = "article_gpt.md"


def initialize_llm_models(model_type: str, model_name: str) -> tuple[OpenAI | Anthropic, OpenAI | Anthropic]:
    """
    Initialize and return LLM models based on the specified provider.

    Args:
        model (str): The model provider to use. Either "gpt" or "claude". Defaults to "gpt".

    Returns:
        Tuple[Union[OpenAI, Anthropic], Union[OpenAI, Anthropic]]: A tuple containing two LLM instances:
            - The first for the react agent (with higher temperature)
            - The second for the regular agent (with lower temperature)

    Raises:
        ValueError: If an invalid model provider is specified.
    """
    if model_type == "gpt":
        connection = OpenAIConnection()
        llm_react_agent = OpenAI(
            connection=connection,
            model=model_name,
            temperature=REACT_AGENT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stop=["Observation:", "\nObservation:", "\n\tObservation:"],
        )
        llm = OpenAI(
            connection=connection,
            model=model_name,
            temperature=REGULAR_AGENT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    elif model_type == "claude":
        connection = AnthropicConnection()
        llm_react_agent = Anthropic(
            connection=connection,
            model=model_name,
            temperature=REACT_AGENT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stop=["Observation:", "\nObservation:"],
        )
        llm = Anthropic(
            connection=connection,
            model=model_name,
            temperature=REGULAR_AGENT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    else:
        raise ValueError(f"Invalid model provider: {model_type}")

    return llm_react_agent, llm


def run_workflow(
    user_prompt: str = PROMPT, model_type: str = "gpt", model_name: str = "gpt-4o-mini"
) -> tuple[str, dict]:
    """
    Set up and run a multi-agent workflow to generate a literature overview.

    This function initializes LLM models, creates necessary tools and agents,
    sets up a workflow, and executes it based on a user-provided prompt.
    The resulting content is printed to the console and saved to a file.
    """
    # Load environment variables
    load_dotenv()

    # Initialize LLM models
    llm_react_agent, llm_agent = initialize_llm_models(model_type, model_name)

    # Set up API connections
    search_connection = ScaleSerp()

    # Initialize tools
    tool_search = ScaleSerpTool(connection=search_connection)

    # Create agents
    agent_researcher = ReActAgent(
        name="Research Analyst",
        llm=llm_react_agent,
        tools=[tool_search],
        role=AGENT_RESEARCHER_ROLE,
        max_loops=8,
        function_calling=True,
    )

    agent_writer = Agent(
        name="Writer and Editor",
        llm=llm_agent,
        role=AGENT_WRITER_ROLE,
    )

    # Create workflow
    agent_manager = AdaptiveAgentManager(llm=llm_agent)
    # Create orchestrator
    adaptive_orchestrator = AdaptiveOrchestrator(
        manager=agent_manager,
        agents=[agent_researcher, agent_writer],
        final_summarizer=True,
        saving_file=True,
    )
    workflow = Workflow(
        flow=Flow(nodes=[adaptive_orchestrator]),
    )

    # Define user prompt

    # Run the workflow
    tracing = TracingCallbackHandler()
    try:
        result = workflow.run(
            input_data={
                "input": user_prompt,
            },
            config=RunnableConfig(callbacks=[tracing]),
        )

        # Dump traces
        _ = json.dumps(
            {"runs": [run.to_dict() for run in tracing.runs.values()]},
            cls=JsonWorkflowEncoder,
        )

        logger.info("Workflow completed")

        content = result.output[adaptive_orchestrator.id]["output"]["content"]
        return content, tracing.runs
    except Exception as e:
        logger.error(f"An error occurred during workflow execution: {str(e)}")
        return "", {}


if __name__ == "__main__":
    content, _ = run_workflow()
    print(content)
    with open(OUTPUT_FILE_PATH, "w") as f:
        f.write(content)
