import uuid
from unittest.mock import MagicMock

import pytest

from dynamiq import connections, prompts
from dynamiq.nodes.agents.exceptions import (
    ActionParsingException,
    JSONParsingError,
    ParsingError,
    TagNotFoundError,
    XMLParsingError,
)
from dynamiq.nodes.agents.react import ReActAgent
from dynamiq.nodes.agents.utils import XMLParser
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
    assert answer[0] == "I found all the information needed."
    assert answer[1] == "The weather in San Francisco is foggy with a high of 65°F."


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


def test_generate_prompt_xml_mode(openai_node, mock_llm_executor):
    """Test prompt generation in XML inference mode."""
    agent = ReActAgent(name="XMLPromptAgent", llm=openai_node, tools=[], inference_mode=InferenceMode.XML)

    prompt = agent.generate_prompt()

    assert "<output>" in prompt
    assert "<thought>" in prompt
    assert "<answer>" in prompt
    assert "Always use this exact XML format" in prompt


def test_set_prompt_block(openai_node, mock_llm_executor):
    """Test modifying prompt blocks."""
    agent = ReActAgent(name="PromptBlockTestAgent", llm=openai_node, tools=[], inference_mode=InferenceMode.DEFAULT)

    custom_instructions = "Your goal is to analyze the given text and identify key points."
    agent.set_block("instructions", custom_instructions)

    prompt = agent.generate_prompt()

    assert custom_instructions in prompt

    custom_context = "You are a helpful scientific research assistant."
    agent.set_block("context", custom_context)

    prompt = agent.generate_prompt()

    assert custom_instructions in prompt
    assert custom_context in prompt


def test_set_prompt_variable(openai_node, mock_llm_executor):
    """Test setting prompt variables."""
    agent = ReActAgent(name="PromptVarTestAgent", llm=openai_node, tools=[], inference_mode=InferenceMode.DEFAULT)

    agent.set_prompt_variable("custom_date", "April 1, 2025")

    agent.set_block("date", "Today's date is {custom_date}")

    prompt = agent.generate_prompt()

    assert "Today's date is April 1, 2025" in prompt


def test_xmlparser_parse_valid_simple():
    text = "<output><thought>OK</thought><action>do</action></output>"
    result = XMLParser.parse(text, required_tags=["thought", "action"])
    assert result == {"thought": "OK", "action": "do"}


def test_xmlparser_parse_valid_with_optional():
    text = "<output><thought>OK</thought><action>do</action><optional>extra</optional></output>"
    result = XMLParser.parse(text, required_tags=["thought", "action"], optional_tags=["optional"])
    assert result == {"thought": "OK", "action": "do", "optional": "extra"}


def test_xmlparser_parse_valid_with_json():
    text = '<output><thought>OK</thought><action>do</action><action_input>{"p": 1}</action_input></output>'
    result = XMLParser.parse(text, required_tags=["thought", "action", "action_input"], json_fields=["action_input"])
    assert result == {"thought": "OK", "action": "do", "action_input": {"p": 1}}


def test_xmlparser_parse_missing_required_tag():
    text = "<output><thought>OK</thought></output>"
    with pytest.raises(TagNotFoundError, match="Required tag <action> not found"):
        XMLParser.parse(text, required_tags=["thought", "action"])


def test_xmlparser_parse_required_tag_empty():
    text = "<output><thought></thought><action>do</action></output>"
    with pytest.raises(TagNotFoundError, match="Required tag <thought> found but contains no text"):
        XMLParser.parse(text, required_tags=["thought", "action"])


def test_xmlparser_parse_malformed_json():
    text = '<output><thought>OK</thought><action_input>{"p": 1</action_input></output>'
    with pytest.raises(JSONParsingError, match="Failed to parse JSON content for field 'action_input'"):
        XMLParser.parse(text, required_tags=["thought", "action_input"], json_fields=["action_input"])


def test_xmlparser_parse_malformed_xml():
    text = "<output><thought>OK</action>"
    with pytest.raises((TagNotFoundError, XMLParsingError)):
        XMLParser.parse(text, required_tags=["thought", "action"])


def test_xmlparser_parse_with_markdown_fence():
    text = "```xml\n<output><thought>OK</thought><action>do</action></output>\n```"
    result = XMLParser.parse(text, required_tags=["thought", "action"])
    assert result == {"thought": "OK", "action": "do"}


def test_xmlparser_parse_with_extra_text():
    text = "Here is the plan:\n<output><thought>OK</thought><action>do</action></output>\nLet me know."
    result = XMLParser.parse(text, required_tags=["thought", "action"])
    assert result == {"thought": "OK", "action": "do"}


def test_xmlparser_parse_empty_input():
    with pytest.raises(ParsingError, match="Input text is empty"):
        XMLParser.parse("", required_tags=["thought"])
    result = XMLParser.parse("", required_tags=[])
    assert result == {}


def test_xmlparser_extract_lxml_found():
    text = "<root><other>ignore</other><final_answer>The Answer</final_answer></root>"
    result = XMLParser.extract_first_tag_lxml(text, ["output", "final_answer"])
    assert result == "The Answer"


def test_xmlparser_extract_lxml_first_preference():
    text = "<root><output>First</output><final_answer>Second</final_answer></root>"
    result = XMLParser.extract_first_tag_lxml(text, ["output", "final_answer"])
    assert result == "First"


def test_xmlparser_extract_lxml_not_found():
    text = "<root><other>ignore</other></root>"
    result = XMLParser.extract_first_tag_lxml(text, ["output", "final_answer"])
    assert result is None


def test_xmlparser_extract_lxml_empty_tag():
    text = "<root><final_answer></final_answer></root>"
    result = XMLParser.extract_first_tag_lxml(text, ["output", "final_answer"])
    assert result is None


def test_xmlparser_extract_regex_found():
    text = "Blah <final_answer> Regex Answer </final_answer> blah"
    result = XMLParser.extract_first_tag_regex(text, ["output", "final_answer"])
    assert result == "Regex Answer"


def test_xmlparser_extract_regex_not_found():
    text = "Blah blah"
    result = XMLParser.extract_first_tag_regex(text, ["output", "final_answer"])
    assert result is None


def test_xmlparser_parse_with_chart_in_answer():
    """Test that XML parser preserves markdown code blocks for charts in answer tags."""
    text = """<output>
  <thought>The user wants to create a chart.</thought>
  <answer>
    # Total Approved Expenses
    ```chart
    {
      "title": "Total Approved Expenses Per Month (USD)",
      "width": 500,
      "height": 300,
      "data": {
        "values": [
          {"month": "January 2025", "amount": 745982.33}
        ]
      }
    }
    ```
    The chart shows January expenses.
  </answer>
</output>"""

    result = XMLParser.parse(text, required_tags=["thought", "answer"])
    assert "```chart" in result["answer"]


def test_xmlparser_parse_with_markdown_in_answer():
    """Test that XML parser preserves markdown formatting in answer tags."""
    text = """<output>
  <thought>Let me provide a detailed answer.</thought>
  <answer>
    # Heading 1
    ## Heading 2

    This is **bold text** and *italic text*.

    * Bullet point 1
    * Bullet point 2

    1. Numbered item
    2. Another numbered item

    Here's a [link](https://example.com).

    And a code example:
    ```python
    def hello_world():
        print("Hello, world!")
    ```
  </answer>
</output>"""

    result = XMLParser.parse(text, required_tags=["thought", "answer"])

    # Check markdown elements are preserved
    assert "# Heading 1" in result["answer"]
    assert "**bold text**" in result["answer"]
    assert "```python" in result["answer"]
    assert "* Bullet point" in result["answer"]


def test_xmlparser_parse_with_special_characters_in_answer():
    """Test that XML parser preserves special characters like & in answer tags."""
    text = """<output>
  <thought>Let me provide information about R&D practices.</thought>
  <answer>
    # Research & Development (R&D)

    R&D departments are crucial for innovation. Companies like AT&T, Johnson & Johnson,
    and Procter & Gamble invest heavily in R&D.

    Common R&D focus areas:
    * AI & Machine Learning
    * Blockchain & Distributed Systems
    * AR/VR & Immersive Technologies

    The R&D tax credit can be 14% & 20% depending on jurisdiction.
  </answer>
</output>"""

    result = XMLParser.parse(text, required_tags=["thought", "answer"])

    assert "Research & Development (R&D)" in result["answer"]
    assert "R&D departments" in result["answer"]
    assert "AI & Machine Learning" in result["answer"]
    assert "14% & 20%" in result["answer"]


def test_xmlparser_parse_with_unclosed_answer_tag():
    """Test that XML parser correctly handles unclosed answer tags."""
    text = """<output>
  <thought>Let me provide an answer to your question.</thought>
  <answer>
    This is the answer to your question about climate change.

    Climate change is driven by several factors:
    1. Greenhouse gas emissions
    2. Deforestation
    3. Industrial processes

    Recent studies suggest that we need immediate action.
</output>"""

    result = XMLParser.parse(text, required_tags=["thought", "answer"])

    assert "climate change" in result["answer"]
    assert "Greenhouse gas emissions" in result["answer"]
    assert "immediate action" in result["answer"]


def test_xmlparser_parse_with_only_opening_answer_tag():
    """Test that XML parser handles cases where only the opening answer tag exists."""
    text = """<output>
  <thought>Here's what I found.</thought>
  <answer>
    The GDP of France in 2024 was approximately 3.2 trillion USD,
    representing a 2.1% increase from the previous year."""

    result = XMLParser.parse(text, required_tags=["thought", "answer"])

    assert "GDP of France" in result["answer"]
    assert "3.2 trillion USD" in result["answer"]
