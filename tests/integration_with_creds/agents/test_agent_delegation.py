import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


@pytest.fixture(scope="module")
def llm_instance():
    """Setup LLM instance for testing."""
    logger.info("\n--- Setting up LLM (Fixture) ---")
    connection = OpenAIConnection()
    llm = OpenAI(
        connection=connection,
        model="gpt-5-mini",
        max_tokens=2000,
        temperature=0,
    )
    return llm


@pytest.fixture(scope="module")
def agent_role():
    """Basic agent role for testing."""
    return "You are a helpful assistant that provides accurate information."


@pytest.fixture(scope="module")
def run_config():
    """Standard runnable configuration."""
    return RunnableConfig(request_timeout=120)


@pytest.mark.integration
@pytest.mark.parametrize(
    "inference_mode",
    [
        InferenceMode.DEFAULT,
        InferenceMode.XML,
        InferenceMode.FUNCTION_CALLING,
        InferenceMode.STRUCTURED_OUTPUT,
    ],
    ids=["default", "xml", "function_calling", "structured_output"],
)
def test_delegation_flag_verified_in_traces(llm_instance, run_config, inference_mode):
    """Test that verifies delegation actually occurred and is properly traced across inference modes."""
    calculator_llm = llm_instance.model_copy(update={"name": "CalculatorLLM"})
    delegator_llm = llm_instance.model_copy(update={"name": "DelegatorLLM"})

    calculator_agent = Agent(
        name="CalculatorAgentTool",
        id="calculator_agent_tool",
        llm=calculator_llm,
        description="You are a mathematical calculator assistant. Use the calculator tool"
        "to perform calculations accurately.",
        role="You are a mathematical calculator assistant."
        "Use the calculator tool to perform calculations accurately.",
        inference_mode=inference_mode,
        tools=[],
        verbose=True,
    )

    main_agent = Agent(
        name="DelegatorAgent",
        id="delegator_agent",
        llm=delegator_llm,
        role="You are a helpful assistant. When users ask mathematical questions, delegate to"
        "the CalculatorAgentTool and use delegate_final.",
        inference_mode=inference_mode,
        tools=[calculator_agent],
        delegation_allowed=True,  # Enable delegation
        verbose=True,
    )

    from dynamiq.callbacks.tracing import TracingCallbackHandler

    tracing = TracingCallbackHandler()
    run_config.callbacks = [tracing]

    input_data = {"input": "Calculate 12 * 9. Delegate to the CalculatorAgentTool"}
    result = main_agent.run(input_data=input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS, f"Agent failed for {inference_mode.value}: {result.status}"
    content = result.output["content"]
    assert isinstance(content, str), f"Content should be string for {inference_mode.value}, got {type(content)}"
    assert "108" in content, f"Expected '108' in output for {inference_mode.value}, got: {content}"

    runs = list(tracing.runs.values())
    assert runs, f"[{inference_mode.value}] No tracing runs captured"

    main_agent_run = 0
    calculator_agent_run = 0

    for run in runs:
        node_metadata = run.metadata.get("node", {}) if run.metadata else {}
        node_group = node_metadata.get("group")
        node_name = run.name
        node_id = node_metadata.get("id")

        logger.info(
            f"[{inference_mode.value}] Run: name={node_name}, "
            f"id={node_id}, group={node_group}, parent_run_id={run.parent_run_id}"
        )
        logger.info("Node name")
        logger.info(node_name)
        if node_name == "DelegatorLLM":
            main_agent_run += 1

        if node_name == "CalculatorLLM":
            calculator_agent_run += 1

    assert main_agent_run == 1, f"[{inference_mode.value}] Expected 1 main agent run, got {main_agent_run}"
    assert (
        calculator_agent_run >= 1
    ), f"[{inference_mode.value}] Expected at least 1 calculator agent run, got {calculator_agent_run}"
