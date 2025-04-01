import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig, RunnableStatus

python_tool_code = """
def run(input_data):
    text = input_data.get('text', '')
    return "The length of the string is " + str(len(text))
"""

TEST_INPUT_STRING = "dsfdfgdsfgfsghfghsddfg"
EXPECTED_LENGTH = len(TEST_INPUT_STRING)
AGENT_ROLE = "is to help user with various tasks, goal is to provide best of possible answers to user queries"
AGENT_INPUT = {"input": f"What is the length of the string '{TEST_INPUT_STRING}'?"}
RUN_CONFIG = RunnableConfig(request_timeout=120)


@pytest.fixture(scope="module")
def llm_instance():
    print("\n--- Setting up REAL LLM (Fixture) ---")
    connection = OpenAIConnection()
    llm = OpenAI(
        connection=connection,
        model="gpt-4o-mini",
        max_tokens=1000,
        temperature=0,
    )
    return llm


@pytest.fixture(scope="module")
def string_length_tool_instance():
    print("--- Creating Python tool (StringLengthTool) (Fixture) ---")
    tool = Python(
        name="StringLengthTool",
        description="""Calculates the length of a given string. Input parameter: 'text' (string).""",
        code=python_tool_code,
    )
    print("Tool created.")
    return tool


def _run_and_assert_agent(agent: ReActAgent):
    """Helper function to run agent and perform common assertions."""
    print(f"\n--- Running Agent: {agent.name} (Mode: {agent.inference_mode.value}) ---")
    agent_output = None
    try:
        result = agent.run(input_data=AGENT_INPUT, config=RUN_CONFIG)
        print(f"Agent raw result object: {result}")

        if result.status != RunnableStatus.SUCCESS:
            intermediate_steps = (
                result.output.get("intermediate_steps", "N/A") if isinstance(result.output, dict) else "N/A"
            )
            print(f"Intermediate Steps on Failure: {intermediate_steps}")
            pytest.fail(f"Agent run failed with status '{result.status}'. Output: {result.output}.")

        if isinstance(result.output, dict) and "content" in result.output:
            agent_output = result.output["content"]
        else:
            agent_output = result.output
            print(f"Warning: Agent output structure unexpected: {type(result.output)}")

        print(f"Agent final output content: {agent_output}")

    except Exception as e:
        pytest.fail(f"Agent run failed with exception: {e}")

    print("Asserting results...")
    assert agent_output is not None, "Agent output content should not be None"
    assert isinstance(agent_output, str), f"Agent output content should be a string, got {type(agent_output)}"

    expected_length_str = str(EXPECTED_LENGTH)
    assert (
        expected_length_str in agent_output
    ), f"Expected length '{expected_length_str}' not found in agent output: '{agent_output}'"

    print(f"--- Test Passed for Mode: {agent.inference_mode.value} ---")


@pytest.mark.integration
def test_react_agent_default_mode(llm_instance, string_length_tool_instance):
    agent = ReActAgent(
        name="Test Agent DEFAULT",
        llm=llm_instance,
        tools=[string_length_tool_instance],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.DEFAULT,
        verbose=True,
    )
    _run_and_assert_agent(agent)


@pytest.mark.integration
def test_react_agent_xml_mode(llm_instance, string_length_tool_instance):
    agent = ReActAgent(
        name="Test Agent XML",
        llm=llm_instance,
        tools=[string_length_tool_instance],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.XML,
        verbose=True,
    )
    _run_and_assert_agent(agent)


@pytest.mark.integration
def test_react_agent_structured_output_mode(llm_instance, string_length_tool_instance):
    agent = ReActAgent(
        name="Test Agent STRUCTURED_OUTPUT",
        llm=llm_instance,
        tools=[string_length_tool_instance],
        role=AGENT_ROLE,
        inference_mode=InferenceMode.STRUCTURED_OUTPUT,
        verbose=True,
    )
    _run_and_assert_agent(agent)
