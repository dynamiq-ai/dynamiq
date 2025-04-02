import uuid
from unittest.mock import MagicMock

import pytest

from dynamiq import connections, prompts
from dynamiq.nodes.agents.exceptions import ActionParsingException
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableStatus


@pytest.fixture
def openai_connection():
    return connections.OpenAI(
        id=str(uuid.uuid4()),
        api_key="api-key",
    )


@pytest.fixture
def openai_node(openai_connection):
    return OpenAI(
        name="OpenAI",
        model="gpt-4o-mini",
        connection=openai_connection,
        prompt=prompts.Prompt(
            messages=[
                prompts.Message(
                    role="user",
                    content="{{input}}",
                ),
            ],
        ),
    )


@pytest.fixture
def default_react_agent(openai_node, mock_llm_executor):
    """ReActAgent with DEFAULT inference mode."""
    return ReActAgent(name="Test Agent", llm=openai_node, tools=[], inference_mode=InferenceMode.DEFAULT)


@pytest.fixture
def xml_react_agent(openai_node, mock_llm_executor):
    """ReActAgent with XML inference mode."""
    return ReActAgent(name="Test XML Agent", llm=openai_node, tools=[], inference_mode=InferenceMode.XML)


@pytest.fixture
def mock_tool():
    tool = MagicMock()
    tool.name = "TestTool"
    tool.id = "test-tool-123"
    tool.is_files_allowed = False

    result = MagicMock()
    result.status = RunnableStatus.SUCCESS
    result.output = {"content": "Tool execution result"}
    tool.run.return_value = result

    return tool


def test_parse_thought(default_react_agent):
    """Test extracting thought from agent output."""
    output = """
    Thought: I need to search for information about the weather.
    Action: search
    Action Input: {"query": "weather in San Francisco"}
    """
    thought = default_react_agent._parse_thought(output)
    assert thought == "I need to search for information about the weather."


def test_parse_action_malformed_json(default_react_agent):
    """Test parsing with malformed JSON raises an exception."""
    output = """
    Thought: I need to search for information about the weather.
    Action: search
    Action Input: {"query": "weather in San Francisco}
    """
    with pytest.raises(ActionParsingException) as exc_info:
        default_react_agent._parse_action(output)
    assert "invalid JSON" in str(exc_info.value).lower() or "Unable to parse" in str(exc_info.value)


def test_parse_action_missing_action_input(default_react_agent):
    """Test parsing with missing action input raises an exception."""
    output = """
    Thought: I need to search for information about the weather.
    Action: search
    """
    with pytest.raises(ActionParsingException):
        default_react_agent._parse_action(output)


def test_extract_final_answer(default_react_agent):
    """Test extracting the final answer from the output."""
    output = """
    Thought: I found all the information needed.
    Answer: The weather in San Francisco is foggy with a high of 65°F.
    """
    answer = default_react_agent._extract_final_answer(output)
    assert answer == "The weather in San Francisco is foggy with a high of 65°F."


def test_parse_xml_content(xml_react_agent):
    """Test extracting content from XML tags."""
    text = "<output><thought>I need to search for information</thought></output>"
    content = xml_react_agent.parse_xml_content(text, "thought")
    assert content == "I need to search for information"


def test_parse_xml_and_extract_info_valid(xml_react_agent):
    """Test extracting thought, action, and action_input from valid XML."""
    text = """
    <output>
        <thought>I need to search for the weather</thought>
        <action>search</action>
        <action_input>{"query": "weather in San Francisco"}</action_input>
    </output>
    """
    thought, action, action_input = xml_react_agent.parse_xml_and_extract_info(text)
    assert thought == "I need to search for the weather"
    assert action == "search"
    assert action_input == {"query": "weather in San Francisco"}


def test_parse_xml_missing_tags(xml_react_agent):
    """Test extracting from XML with missing tags raises an exception."""
    text = """
    <output>
        <thought>I need to search for the weather</thought>
        <action>search</action>
        <!-- Missing action_input tag -->
    </output>
    """
    with pytest.raises(ActionParsingException) as exc_info:
        xml_react_agent.parse_xml_and_extract_info(text)
    assert "Missing required XML tags" in str(exc_info.value) or "missing" in str(exc_info.value).lower()


def test_parse_xml_malformed_json(xml_react_agent):
    """Test extracting from XML with malformed JSON raises an exception."""
    text = """
    <output>
        <thought>I need to search for the weather</thought>
        <action>search</action>
        <action_input>{"query": "weather in San Francisco}</action_input>
    </output>
    """
    with pytest.raises(ActionParsingException) as exc_info:
        xml_react_agent.parse_xml_and_extract_info(text)
    assert "invalid JSON" in str(exc_info.value).lower() or "Unable to parse" in str(exc_info.value)


@pytest.mark.parametrize(
    "input_name,expected_output",
    [
        ("My Cool Tool!", "My-Cool-Tool"),
        ("search-api", "search-api"),
        ("data analysis (2023)", "data-analysis-2023"),
    ],
)
def test_sanitize_tool_name(default_react_agent, input_name, expected_output):
    """Test that tool names are sanitized correctly."""
    assert default_react_agent.sanitize_tool_name(input_name) == expected_output
