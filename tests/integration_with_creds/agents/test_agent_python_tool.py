from enum import Enum
from typing import Any, ClassVar, Literal

import pytest
from pydantic import BaseModel, Field

from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic, OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


class OutputFormat(str, Enum):
    SUMMARY = "summary"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"


class TextAnalysisInputSchema(BaseModel):
    text: str = Field(..., description="Text to analyze.")
    language: str = Field(default="en", description="ISO-639 language code.")
    format: OutputFormat = Field(default=OutputFormat.SUMMARY, description="Output format.")
    max_length: int = Field(default=500, description="Maximum output length in characters.")
    keywords: list[str] = Field(default_factory=list, description="Keywords to focus on.")
    include_stats: bool = Field(default=False, description="Include word/char statistics.")


class TextAnalysisTool(Node):
    """Lightweight tool with a complex typed input schema for schema-generation tests."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Text Analysis Tool"
    description: str = "Analyzes text and returns a summary, detailed breakdown, or bullet points."
    input_schema: ClassVar[type[TextAnalysisInputSchema]] = TextAnalysisInputSchema

    def execute(self, input_data: TextAnalysisInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        text = input_data.text
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        keyword_hits = [kw for kw in input_data.keywords if kw.lower() in text.lower()]

        if input_data.format == OutputFormat.BULLET_POINTS:
            body = "\n".join(f"- {w}" for w in words[:10])
        elif input_data.format == OutputFormat.DETAILED:
            body = f"Words: {word_count}, Chars: {char_count}, Keywords found: {keyword_hits}"
        else:
            body = f"Text has {word_count} words."

        if input_data.include_stats:
            body += f" (stats: {word_count} words, {char_count} chars)"

        return {"content": body[: input_data.max_length]}


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


@pytest.fixture(scope="module")
def anthropic_llm():
    connection = AnthropicConnection()
    return Anthropic(
        connection=connection,
        model="claude-3-5-haiku-latest",
        max_tokens=1000,
        temperature=0,
    )


@pytest.fixture(scope="module")
def text_analysis_tool():
    return TextAnalysisTool()


def run_and_assert_agent(agent: Agent, agent_input, expected_length, run_config):
    """Helper function to run agent and perform common assertions."""
    logger.info(f"\n--- Running Agent: {agent.name} (Mode: {agent.inference_mode.value}) ---")
    agent_output = None
    try:
        result = agent.run(input_data=agent_input, config=run_config)
        logger.info(f"Agent raw result object: {result}")

        if result.status != RunnableStatus.SUCCESS:
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
@pytest.mark.parametrize(
    "inference_mode",
    [
        InferenceMode.DEFAULT,
        InferenceMode.XML,
        InferenceMode.STRUCTURED_OUTPUT,
        InferenceMode.FUNCTION_CALLING,
    ],
    ids=["default", "xml", "structured_output", "function_calling"],
)
def test_react_agent_inference_modes(
    llm_instance, string_length_tool_instance, agent_role, agent_input, expected_length, run_config, inference_mode
):
    """Test agent with Python tool across different inference modes."""
    agent = Agent(
        name=f"Test Agent {inference_mode.value}",
        llm=llm_instance,
        tools=[string_length_tool_instance],
        role=agent_role,
        inference_mode=inference_mode,
        verbose=True,
    )
    run_and_assert_agent(agent, agent_input, expected_length, run_config)


def _run_complex_schema_test(llm, text_analysis_tool, run_config, inference_mode, label):
    """Helper: run an agent with TextAnalysisTool and assert success."""
    agent = Agent(
        name=f"Complex Schema Test ({label})",
        llm=llm,
        tools=[text_analysis_tool],
        role="You are a helpful text analysis assistant. Use the Text Analysis Tool to analyze text.",
        inference_mode=inference_mode,
        max_loops=3,
        verbose=True,
    )
    result = agent.run(
        input_data={"input": "Analyze the text 'Hello world from Python testing' in summary format"},
        config=run_config,
    )
    assert result.status == RunnableStatus.SUCCESS, f"Agent run failed for {label}: {result.output}"
    content = result.output.get("content", "")
    assert isinstance(content, str) and len(content) > 0, f"Expected non-empty string for {label}, got: {content!r}"


@pytest.mark.integration
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "inference_mode",
    [InferenceMode.DEFAULT, InferenceMode.XML],
    ids=["default", "xml"],
)
def test_complex_schema_tool_openai(llm_instance, text_analysis_tool, run_config, inference_mode):
    """Complex typed schema with OpenAI in DEFAULT and XML modes."""
    _run_complex_schema_test(
        llm_instance, text_analysis_tool, run_config, inference_mode, f"openai-{inference_mode.value}"
    )


@pytest.mark.integration
@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "inference_mode",
    [InferenceMode.STRUCTURED_OUTPUT, InferenceMode.FUNCTION_CALLING],
    ids=["structured_output", "function_calling"],
)
@pytest.mark.parametrize(
    "llm_fixture",
    ["llm_instance", "anthropic_llm"],
    ids=["openai", "anthropic"],
)
def test_complex_schema_tool_schema_modes(llm_fixture, text_analysis_tool, run_config, inference_mode, request):
    """Complex typed schema with both OpenAI and Anthropic in STRUCTURED_OUTPUT and FUNCTION_CALLING modes."""
    llm = request.getfixturevalue(llm_fixture)
    provider = "openai" if "llm_instance" in llm_fixture else "anthropic"
    _run_complex_schema_test(llm, text_analysis_tool, run_config, inference_mode, f"{provider}-{inference_mode.value}")
