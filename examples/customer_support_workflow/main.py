from dynamiq import Workflow
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes.agents.orchestrators import LinearOrchestrator
from dynamiq.nodes.agents.orchestrators.linear_manager import LinearAgentManager
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms.anthropic import Anthropic
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.human_feedback import HumanFeedbackTool
from dynamiq.utils.logger import logger
from examples.customer_support_workflow.bank_api_sim import BankApiSim
from examples.customer_support_workflow.bank_rag_tool import BankRAGTool

# Constants
GPT_MODEL = "gpt-4o"
CLAUDE_MODEL = "claude-3-5-sonnet-20240620"
MAX_TOKENS = 4000
REACT_AGENT_TEMPERATURE = 0.1
MANAGER_AGENT_TEMPERATURE = 0.1


def create_llm_instances(
    model: str = "gpt",
) -> tuple[OpenAI | Anthropic, OpenAI | Anthropic]:
    """
    Create and return LLM instances based on the specified model.

    Args:
        model (str): The model to use, either "gpt" or "claude". Defaults to "gpt".

    Returns:
        Tuple[Union[OpenAI, Anthropic], Union[OpenAI, Anthropic]]: A tuple containing two LLM instances,
        the first for the ReActAgent and the second for the ManagerAgent.

    Raises:
        ValueError: If an invalid model is specified.
    """
    if model == "gpt":
        connection = OpenAIConnection()
        llm_react_agent = OpenAI(
            connection=connection,
            model=GPT_MODEL,
            temperature=REACT_AGENT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stop=["Observation:", "\nObservation:", "\n\tObservation:"],
        )
        llm = OpenAI(
            connection=connection,
            model=GPT_MODEL,
            temperature=MANAGER_AGENT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    elif model == "claude":
        connection = AnthropicConnection()
        llm_react_agent = Anthropic(
            connection=connection,
            model=CLAUDE_MODEL,
            temperature=REACT_AGENT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stop=["Observation:", "\nObservation:"],
        )
        llm = Anthropic(
            connection=connection,
            model=CLAUDE_MODEL,
            temperature=MANAGER_AGENT_TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    else:
        raise ValueError(f"Invalid model specified: {model}")

    return llm_react_agent, llm


def main():
    """
    Set up and run the customer support workflow.

    This function creates the necessary agents and tools, sets up the workflow,
    and executes it with a sample input.
    """
    # Create LLM instances
    llm_react_agent, llm_agent = create_llm_instances(model="gpt")

    # Create a ReActAgent for handling internal bank API queries
    agent_bank_support = ReActAgent(
        name="Bank Support: Internal API",
        role="customer support assistant for Internal Bank",
        goal="help with provided customer requests",
        llm=llm_react_agent,
        tools=[BankApiSim(), HumanFeedbackTool()],
    )

    # Create a ReActAgent for handling bank documentation queries
    agent_bank_documentation = ReActAgent(
        name="Bank Support: Documentation",
        role="customer support assistant for Internal Bank Documentation",
        goal="help with provided customer requests regarding Internal Bank Documentation",
        llm=llm_react_agent,
        tools=[BankRAGTool(), HumanFeedbackTool()],
    )

    # Create a ManagerAgent to oversee the workflow
    agent_manager = LinearAgentManager(llm=llm_agent)

    # Create a LinearOrchestrator to manage the workflow of multiple agents
    linear_orchestrator = LinearOrchestrator(
        manager=agent_manager,
        agents=[agent_bank_support, agent_bank_documentation],
    )
    workflow = Workflow(flow=Flow(nodes=[linear_orchestrator]))
    logger.info("Workflow created successfully")

    # Run the workflow with a sample input
    result = workflow.run(input_data={"input": "fast block my card"})
    print(result.output[linear_orchestrator.id]["output"]["content"]["result"])


if __name__ == "__main__":
    main()
