import re

import pytest

from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import Gemini as GeminiConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic, Gemini, OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus

OPENAI_MODELS = [
    "o4-mini",
    "o3",
    "o3-mini",
    "o1",
    "gpt-4.1-2025-04-14",
    "gpt-4o-2024-11-20",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-mini-2024-07-18",
]

ANTHROPIC_MODELS = [
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-latest",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-latest",
]

GOOGLE_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

INFERENCE_MODES = [InferenceMode.DEFAULT, InferenceMode.XML]


def create_openai_llm(model):
    connection = OpenAIConnection()
    return OpenAI(
        connection=connection,
        model=model,
        max_tokens=1000,
        temperature=0,
    )


def create_claude_llm(model):
    connection = AnthropicConnection()
    return Anthropic(
        connection=connection,
        model=model,
        max_tokens=1000,
        temperature=0,
    )


def create_gemini_llm(model):
    connection = GeminiConnection()
    return Gemini(
        connection=connection,
        model=model,
        max_tokens=1000,
        temperature=0,
    )


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


def run_and_assert_agent(agent: Agent, agent_input, expected_answer, run_config):
    """Helper function to run agent and perform common assertions."""
    agent_output = None

    try:
        result = agent.run(input_data=agent_input, config=run_config)

        if result.status != RunnableStatus.SUCCESS:
            pytest.fail(f"Agent run failed with status '{result.status}'. Output: {result.output}.")

        if isinstance(result.output, dict):
            if "content" in result.output:
                agent_output = result.output["content"]
        else:
            agent_output = result.output

    except Exception as e:
        pytest.fail(f"Agent run failed with exception: {e}")

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


@pytest.mark.skip(reason="Model access limited by current API key")
@pytest.mark.integration
@pytest.mark.parametrize("model", OPENAI_MODELS)
@pytest.mark.parametrize("inference_mode", INFERENCE_MODES)
def test_react_agent_openai_models(model, inference_mode, emoji_agent_role, agent_input, expected_answer, run_config):
    """Test OpenAI models with different inference modes."""
    llm_instance = create_openai_llm(model)
    agent = Agent(
        name=f"Test Agent {inference_mode.value} (OPENAI-{model})",
        llm=llm_instance,
        tools=[],
        role=emoji_agent_role,
        inference_mode=inference_mode,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_answer, run_config)


@pytest.mark.skip(reason="Model access limited by current API key")
@pytest.mark.integration
@pytest.mark.parametrize("model", ANTHROPIC_MODELS)
@pytest.mark.parametrize("inference_mode", INFERENCE_MODES)
def test_react_agent_anthropic_models(
    model, inference_mode, emoji_agent_role, agent_input, expected_answer, run_config
):
    """Test Anthropic models with different inference modes."""
    llm_instance = create_claude_llm(model)
    agent = Agent(
        name=f"Test Agent {inference_mode.value} (ANTHROPIC-{model})",
        llm=llm_instance,
        tools=[],
        role=emoji_agent_role,
        inference_mode=inference_mode,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_answer, run_config)


@pytest.mark.skip(reason="Model access limited by current API key")
@pytest.mark.integration
@pytest.mark.parametrize("model", GOOGLE_MODELS)
@pytest.mark.parametrize("inference_mode", INFERENCE_MODES)
def test_react_agent_google_models(model, inference_mode, emoji_agent_role, agent_input, expected_answer, run_config):
    """Test Google models with different inference modes."""
    llm_instance = create_gemini_llm(model)
    agent = Agent(
        name=f"Test Agent {inference_mode.value} (GOOGLE-{model})",
        llm=llm_instance,
        tools=[],
        role=emoji_agent_role,
        inference_mode=inference_mode,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_answer, run_config)
