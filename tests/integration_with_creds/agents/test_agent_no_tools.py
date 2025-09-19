import re

import pytest

from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents.react import InferenceMode, ReActAgent
from dynamiq.nodes.agents.utils import extract_thought_from_intermediate_steps
from dynamiq.nodes.llms import Anthropic, OpenAI
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


def create_openai_llm():
    connection = OpenAIConnection()
    return OpenAI(
        connection=connection,
        model="o3-mini",
        max_tokens=1000,
        temperature=0,
    )


def create_claude_llm():
    connection = AnthropicConnection()
    return Anthropic(
        connection=connection,
        model="claude-3-5-haiku-20241022",
        max_tokens=1000,
        temperature=0,
    )


LLM_PARAMS = [
    ("openai", create_openai_llm),
    ("claude", create_claude_llm),
]

MODE_PARAMS = [
    (InferenceMode.DEFAULT, "emoji_agent_role"),
    (InferenceMode.XML, "emoji_agent_role"),
    (InferenceMode.FUNCTION_CALLING, "emoji_agent_role"),
    (InferenceMode.STRUCTURED_OUTPUT, "base_agent_role"),
]


@pytest.fixture
def base_agent_role():
    return "is to help user with various tasks, goal is to provide best of possible answers to user queries"


@pytest.fixture
def emoji_agent_role(base_agent_role):
    return base_agent_role + ", always include emojis in your responses"


@pytest.fixture
def agent_input():
    return {"input": "What is the capital of the UK?"}


@pytest.fixture
def expected_answer():
    return "London"


@pytest.fixture
def run_config():
    return RunnableConfig(request_timeout=120)


def check_for_emoji(text):
    """Check if text contains emoji using multiple methods."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "\U0001FA70-\U0001FAFF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )

    if emoji_pattern.search(text):
        return True

    if "🇬🇧" in text:
        return True

    common_emoji = ["✨", "🌟", "⭐", "❤️", "✅"]
    return any(emoji in text for emoji in common_emoji)


# --- Test Function for Running Agent ---
def run_and_assert_agent(agent: ReActAgent, agent_input, expected_answer, run_config):
    """Helper function to run agent and perform common assertions."""
    llm_type = agent.llm.__class__.__name__
    logger.info(f"\n--- Running Agent: {agent.name} (Mode: {agent.inference_mode.value}, LLM: {llm_type}) ---")

    agent_output = None
    intermediate_steps = None

    try:
        result = agent.run(input_data=agent_input, config=run_config)
        logger.info(f"Agent run completed with status: {result.status}")

        if result.status != RunnableStatus.SUCCESS:
            intermediate_steps = (
                result.output.get("intermediate_steps", "N/A") if isinstance(result.output, dict) else "N/A"
            )
            logger.info(f"Intermediate Steps on Failure: {intermediate_steps}")
            pytest.fail(f"Agent run failed with status '{result.status}'. Output: {result.output}.")

        if isinstance(result.output, dict):
            if "content" in result.output:
                agent_output = result.output["content"]
            if "intermediate_steps" in result.output:
                intermediate_steps = result.output["intermediate_steps"]
        else:
            agent_output = result.output
            logger.info(f"Warning: Agent output structure unexpected: {type(result.output)}")

        logger.info(f"Agent final output content: {agent_output}")

        thought = None
        if intermediate_steps:
            thought = extract_thought_from_intermediate_steps(intermediate_steps)
            logger.info(f"Extracted thought: {thought}")

    except Exception as e:
        pytest.fail(f"Agent run failed with exception: {e}")

    logger.info("Asserting results...")
    assert agent_output is not None, "Agent output content should not be None"
    assert isinstance(agent_output, str), f"Agent output content should be a string, got {type(agent_output)}"

    assert (
        expected_answer in agent_output
    ), f"Expected answer '{expected_answer}' not found in agent output: '{agent_output}'"

    if "emojis" in agent.role:
        has_emoji = check_for_emoji(agent_output)
        assert has_emoji, f"Expected emoji in agent output, but none found: '{agent_output}'"
    else:
        has_emoji = check_for_emoji(agent_output)
        if has_emoji:
            logger.info(f"Note: Found emoji in output even though not required: '{agent_output}'")
        else:
            logger.info(f"Note: No emoji in output (as expected for this role configuration): '{agent_output}'")

    if agent.inference_mode == InferenceMode.DEFAULT:
        thought_found = thought is not None or "Thought:" in str(intermediate_steps)
        assert thought_found, "Expected thought process to be present in DEFAULT mode"

    elif agent.inference_mode == InferenceMode.XML:
        thought_found = thought is not None or "<thought>" in str(intermediate_steps)
        assert thought_found, "Expected <thought> tags to be present in XML mode"

    elif agent.inference_mode == InferenceMode.FUNCTION_CALLING:
        thought_found = thought is not None or '"thought"' in str(intermediate_steps)
        assert thought_found, "Expected thought field to be present in FUNCTION_CALLING mode"

    elif agent.inference_mode == InferenceMode.STRUCTURED_OUTPUT:
        thought_found = thought is not None or '"thought"' in str(intermediate_steps)
        assert thought_found, "Expected thought field to be present in STRUCTURED_OUTPUT mode"

    logger.info(f"--- Test Passed for Mode: {agent.inference_mode.value}, LLM: {llm_type} ---")


@pytest.mark.integration
@pytest.mark.parametrize("llm_name, llm_creator", LLM_PARAMS)
def test_react_agent_default_mode(llm_name, llm_creator, emoji_agent_role, agent_input, expected_answer, run_config):
    llm_instance = llm_creator()
    agent = ReActAgent(
        name=f"Test Agent DEFAULT ({llm_name.upper()})",
        llm=llm_instance,
        tools=[],
        role=emoji_agent_role,
        inference_mode=InferenceMode.DEFAULT,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_answer, run_config)


@pytest.mark.integration
@pytest.mark.parametrize("llm_name, llm_creator", LLM_PARAMS)
def test_react_agent_xml_mode(llm_name, llm_creator, emoji_agent_role, agent_input, expected_answer, run_config):
    llm_instance = llm_creator()
    agent = ReActAgent(
        name=f"Test Agent XML ({llm_name.upper()})",
        llm=llm_instance,
        tools=[],
        role=emoji_agent_role,
        inference_mode=InferenceMode.XML,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_answer, run_config)


@pytest.mark.integration
@pytest.mark.parametrize("llm_name, llm_creator", LLM_PARAMS)
def test_react_agent_function_calling_mode(
    llm_name, llm_creator, base_agent_role, agent_input, expected_answer, run_config
):
    llm_instance = llm_creator()
    agent = ReActAgent(
        name=f"Test Agent FC ({llm_name.upper()})",
        llm=llm_instance,
        tools=[],
        role=base_agent_role,
        inference_mode=InferenceMode.FUNCTION_CALLING,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_answer, run_config)


@pytest.mark.integration
@pytest.mark.parametrize("llm_name, llm_creator", LLM_PARAMS)
def test_react_agent_structured_output_mode(
    llm_name, llm_creator, base_agent_role, agent_input, expected_answer, run_config
):
    llm_instance = llm_creator()
    agent = ReActAgent(
        name=f"Test Agent SO ({llm_name.upper()})",
        llm=llm_instance,
        tools=[],
        role=base_agent_role,
        inference_mode=InferenceMode.STRUCTURED_OUTPUT,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_answer, run_config)
