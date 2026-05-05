from enum import Enum
from typing import Any, ClassVar, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field

from dynamiq import Workflow
from dynamiq.callbacks.streaming import StreamingIteratorCallbackHandler
from dynamiq.connections import Anthropic as AnthropicConnection
from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.flows import Flow
from dynamiq.nodes import Node, NodeGroup
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import Anthropic, OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.types.streaming import StreamingConfig, StreamingMode
from dynamiq.utils.logger import logger

from .streaming_assertions import assert_streaming_events, collect_streaming_events


class OutputFormat(str, Enum):
    SUMMARY = "summary"
    DETAILED = "detailed"
    BULLET_POINTS = "bullet_points"


class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FilterOptions(BaseModel):
    """Nested model -- tests Model | None union in the parent schema."""

    min_score: float = Field(default=0.0, description="Minimum score threshold.")
    tags: list[str] = Field(default_factory=list, description="Tags to filter by.")


class ActionType(str, Enum):
    """Required enum (no default) -- mirrors Stagehand's action_type."""

    ANALYZE = "analyze"
    SUMMARIZE = "summarize"
    EXTRACT = "extract"


class ComprehensiveInputSchema(BaseModel):
    """Single schema covering all type patterns found in real tool schemas.

    Non-nullable:
        str (required), str (default), Enum (default), Enum (required, no default),
        int (default), list[str] (default_factory), bool (default),
        str (required + min_length)
    Nullable:
        int|None (with ge/le), int|None (bare, no Field), str|None, bool|None,
        Enum|None, list[str]|None, Model|None
    Special:
        is_accessible_to_agent=False, ConfigDict(extra='allow')
    """

    model_config = ConfigDict(extra="allow")

    text: str = Field(..., description="Text to analyze.")
    action_type: ActionType = Field(..., description="Type of analysis to perform.")
    output_name: str = Field(..., min_length=1, description="Output identifier for the result.")
    language: str = Field(default="en", description="ISO-639 language code.")
    format: OutputFormat = Field(default=OutputFormat.SUMMARY, description="Output format.")
    max_length: int = Field(default=500, description="Maximum output length in characters.")
    keywords: list[str] = Field(default_factory=list, description="Keywords to focus on.")
    include_stats: bool = Field(default=False, description="Include word/char statistics.")
    count: int | None = None
    limit: int | None = Field(default=None, ge=1, le=100, description="Max results to return.")
    label: str | None = Field(default=None, description="Optional label.")
    verbose: bool | None = Field(default=None, description="Enable verbose output.")
    priority: Priority | None = Field(default=None, description="Optional priority level.")
    domains: list[str] | None = Field(default=None, description="Whitelist of domains.")
    filters: FilterOptions | None = Field(default=None, description="Optional filter configuration.")
    internal_trace_id: str | None = Field(
        default=None,
        description="Internal tracing ID.",
        json_schema_extra={"is_accessible_to_agent": False},
    )


class ComprehensiveTool(Node):
    """Tool whose schema exercises all major type patterns from real tools."""

    group: Literal[NodeGroup.TOOLS] = NodeGroup.TOOLS
    name: str = "Comprehensive Tool"
    description: str = (
        "Analyzes text with a specified action type. "
        "Required: 'text', 'action_type' (analyze/summarize/extract), 'output_name'. "
        "All other parameters are optional."
    )
    input_schema: ClassVar[type[ComprehensiveInputSchema]] = ComprehensiveInputSchema

    def execute(self, input_data: ComprehensiveInputSchema, config: RunnableConfig = None, **kwargs) -> dict[str, Any]:
        text = input_data.text
        words = text.split()
        word_count = len(words)
        char_count = len(text)

        keyword_hits = [kw for kw in input_data.keywords if kw.lower() in text.lower()]

        if input_data.format == OutputFormat.BULLET_POINTS:
            body = "Bullet-point analysis:\n" + "\n".join(f"- {w}" for w in words[:10])
        elif input_data.format == OutputFormat.DETAILED:
            body = (
                f"Detailed analysis: {word_count} words, {char_count} chars. "
                f"Keywords found: {keyword_hits or 'none'}."
            )
        else:
            body = f"Summary: {word_count} words, {char_count} chars in {input_data.language}."

        body = f"[{input_data.action_type.value}][{input_data.output_name}] {body}"

        if input_data.include_stats:
            body += f" Stats: {word_count}w, {char_count}c."

        extras = []
        if input_data.count is not None:
            extras.append(f"count={input_data.count}")
        if input_data.limit is not None:
            extras.append(f"limit={input_data.limit}")
        if input_data.label is not None:
            extras.append(f"label={input_data.label!r}")
        if input_data.verbose is not None:
            extras.append(f"verbose={input_data.verbose}")
        if input_data.priority is not None:
            extras.append(f"priority={input_data.priority.value}")
        if input_data.domains is not None:
            extras.append(f"domains={input_data.domains}")
        if input_data.filters is not None:
            extras.append(f"filters(min_score={input_data.filters.min_score}, tags={input_data.filters.tags})")
        if extras:
            body += " Options: " + ", ".join(extras) + "."

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
        model="gpt-5.4-mini",
        max_tokens=5000,
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
        model="claude-haiku-4-5",
        max_tokens=5000,
        temperature=0,
    )


@pytest.fixture(scope="module")
def comprehensive_tool():
    return ComprehensiveTool()


def run_and_assert_agent(agent: Agent, agent_input, expected_length, run_config):
    """Helper function to run agent and perform common assertions including streaming validation."""
    logger.info(f"\n--- Running Agent: {agent.name} (Mode: {agent.inference_mode.value}) ---")

    streaming = StreamingIteratorCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))
    result = wf.run(
        input_data=agent_input,
        config=RunnableConfig(callbacks=[streaming], request_timeout=120),
    )

    assert (
        result.status == RunnableStatus.SUCCESS
    ), f"Agent run failed with status '{result.status}'. Output: {result.output}."

    agent_result = result.output.get(agent.id, {}).get("output", {})
    if isinstance(agent_result, dict) and "content" in agent_result:
        agent_output = agent_result["content"]
    else:
        agent_output = agent_result
        logger.info(f"Warning: Agent output structure unexpected: {type(agent_result)}")

    logger.info(f"Agent final output content: {agent_output}")

    assert agent_output is not None, "Agent output content should not be None"
    assert isinstance(agent_output, str), f"Agent output content should be a string, got {type(agent_output)}"

    expected_length_str = str(expected_length)
    assert (
        expected_length_str in agent_output
    ), f"Expected length '{expected_length_str}' not found in agent output: '{agent_output}'"

    ordered_events = collect_streaming_events(streaming, agent.id)
    assert_streaming_events(ordered_events, agent.inference_mode, agent.streaming.mode)

    logger.info(f"--- Test Passed for Mode: {agent.inference_mode.value} ---")


@pytest.mark.integration
@pytest.mark.flaky(reruns=3)
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
        streaming=StreamingConfig(
            enabled=True,
            mode=StreamingMode.ALL,
        ),
    )
    run_and_assert_agent(agent, agent_input, expected_length, run_config)


def _run_comprehensive_schema_test(llm, comprehensive_tool, run_config, inference_mode, label):
    """Helper: run agent with ComprehensiveTool and assert success with streaming validation."""
    agent = Agent(
        name=f"Comprehensive Schema Test ({label})",
        llm=llm,
        tools=[comprehensive_tool],
        role=(
            "You are a helpful assistant. Use the Comprehensive Tool to analyze text. "
            "Pass only the parameters that are relevant; leave nullable ones as null."
        ),
        inference_mode=inference_mode,
        max_loops=5,
        verbose=True,
        streaming=StreamingConfig(
            enabled=True,
            mode=StreamingMode.ALL,
        ),
    )

    streaming = StreamingIteratorCallbackHandler()
    wf = Workflow(flow=Flow(nodes=[agent]))
    result = wf.run(
        input_data={
            "input": "Analyze the text 'Hello world from Python testing' in summary format with limit "
            "5 and high priority"
        },
        config=RunnableConfig(callbacks=[streaming], request_timeout=120),
    )

    assert result.status == RunnableStatus.SUCCESS, f"Agent run failed for {label}: {result.output}"
    agent_result = result.output.get(agent.id, {}).get("output", {})
    content = agent_result.get("content", "") if isinstance(agent_result, dict) else agent_result
    assert isinstance(content, str) and len(content) > 0, f"Expected non-empty string for {label}, got: {content!r}"

    ordered_events = collect_streaming_events(streaming, agent.id)
    assert_streaming_events(ordered_events, inference_mode, agent.streaming.mode)


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
def test_comprehensive_schema_tool_modes(llm_fixture, comprehensive_tool, run_config, inference_mode, request):
    """Comprehensive typed schema (non-nullable + nullable + hidden fields) with OpenAI and Anthropic."""
    llm = request.getfixturevalue(llm_fixture)
    provider = "openai" if "llm_instance" in llm_fixture else "anthropic"
    _run_comprehensive_schema_test(
        llm, comprehensive_tool, run_config, inference_mode, f"{provider}-{inference_mode.value}"
    )
