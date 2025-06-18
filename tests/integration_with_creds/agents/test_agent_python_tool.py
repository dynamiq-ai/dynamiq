import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


@pytest.fixture(scope="module")
def python_tool_code():
    return """
def run(input_data):
    text = input_data.get('text', '')
    return "The length of the string is " + str(len(text))
"""


@pytest.fixture(scope="module")
def test_input_string():
    return "dsfdfgdsfgfsghfghsddfg"


@pytest.fixture(scope="module")
def expected_length(test_input_string):
    return len(test_input_string)


@pytest.fixture(scope="module")
def agent_role():
    return "is to help user with various tasks, goal is to provide best of possible answers to user queries"


@pytest.fixture(scope="module")
def agent_input(test_input_string):
    return {"input": f"What is the length of the string '{test_input_string}'?"}


@pytest.fixture(scope="module")
def run_config():
    return RunnableConfig(request_timeout=120)


@pytest.fixture(scope="module")
def llm_instance():
    logger.info("\n--- Setting up REAL LLM (Fixture) ---")
    connection = OpenAIConnection()
    llm = OpenAI(
        connection=connection,
        model="gpt-4o-mini",
        max_tokens=1000,
        temperature=0,
    )
    return llm


@pytest.fixture(scope="module")
def string_length_tool_instance(python_tool_code):
    logger.info("--- Creating Python tool (StringLengthTool) (Fixture) ---")
    tool = Python(
        name="StringLengthTool",
        description="""Calculates the length of a given string. Input parameter: 'text' (string).""",
        code=python_tool_code,
    )
    logger.info("Tool created.")
    return tool


def run_and_assert_agent(agent: ReActAgent, agent_input, expected_length, run_config):
    """Helper function to run agent and perform common assertions."""
    logger.info(f"\n--- Running Agent: {agent.name} (Mode: {agent.inference_mode.value}) ---")
    agent_output = None
    try:
        result = agent.run(input_data=agent_input, config=run_config)
        logger.info(f"Agent raw result object: {result}")

        if result.status != RunnableStatus.SUCCESS:
            intermediate_steps = (
                result.output.get("intermediate_steps", "N/A") if isinstance(result.output, dict) else "N/A"
            )
            logger.info(f"Intermediate Steps on Failure: {intermediate_steps}")
            pytest.fail(f"Agent run failed with status '{result.status}'. Output: {result.output}.")

        if isinstance(result.output, dict) and "content" in result.output:
            agent_output = result.output["content"]
        else:
            agent_output = result.output
            logger.info(f"Warning: Agent output structure unexpected: {type(result.output)}")

        logger.info(f"Agent final output content: {agent_output}")

    except Exception as e:
        pytest.fail(f"Agent run failed with exception: {e}")

    logger.info("Asserting results...")
    assert agent_output is not None, "Agent output content should not be None"
    assert isinstance(agent_output, str), f"Agent output content should be a string, got {type(agent_output)}"

    expected_length_str = str(expected_length)
    assert (
        expected_length_str in agent_output
    ), f"Expected length '{expected_length_str}' not found in agent output: '{agent_output}'"

    logger.info(f"--- Test Passed for Mode: {agent.inference_mode.value} ---")


@pytest.mark.integration
def test_react_agent_default_mode(
    llm_instance, string_length_tool_instance, agent_role, agent_input, expected_length, run_config
):
    agent = ReActAgent(
        name="Test Agent DEFAULT",
        llm=llm_instance,
        tools=[string_length_tool_instance],
        role=agent_role,
        inference_mode=InferenceMode.DEFAULT,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_length, run_config)


@pytest.mark.integration
def test_react_agent_xml_mode(
    llm_instance, string_length_tool_instance, agent_role, agent_input, expected_length, run_config
):
    agent = ReActAgent(
        name="Test Agent XML",
        llm=llm_instance,
        tools=[string_length_tool_instance],
        role=agent_role,
        inference_mode=InferenceMode.XML,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_length, run_config)


@pytest.mark.skip(reason="Skipping test for JSON")
def test_react_agent_structured_output_mode(
    llm_instance, string_length_tool_instance, agent_role, agent_input, expected_length, run_config
):
    agent = ReActAgent(
        name="Test Agent STRUCTURED_OUTPUT",
        llm=llm_instance,
        tools=[string_length_tool_instance],
        role=agent_role,
        inference_mode=InferenceMode.STRUCTURED_OUTPUT,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_length, run_config)


@pytest.mark.skip(reason="Skipping test for FC")
def test_react_agent_function_calling_mode(
    llm_instance, string_length_tool_instance, agent_role, agent_input, expected_length, run_config
):
    agent = ReActAgent(
        name="Test Agent FUNCTION_CALLING",
        llm=llm_instance,
        tools=[string_length_tool_instance],
        role=agent_role,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_length, run_config)
