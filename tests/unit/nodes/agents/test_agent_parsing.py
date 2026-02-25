import pytest

from dynamiq.nodes.agents.components import parser
from dynamiq.nodes.agents.exceptions import ActionParsingException


def test_parse_default_action_extra_newlines():
    """Test parsing of output with extra newlines and nested JSON."""
    output = r"""Thought: I need to calculate something.

Action: Calculator Tool

Action Input: {
    "code": "print('Hello World')\nprint({'nested': 'dict'})"
}"""

    thought, action, action_input = parser.parse_default_action(output)

    assert thought == "I need to calculate something."
    assert action == "Calculator Tool"
    assert action_input == {"code": "print('Hello World')\nprint({'nested': 'dict'})"}


def test_parse_default_action_standard_format():
    """Test parsing of standard strict format."""
    output = 'Thought: thinking\nAction: Tool\nAction Input: {"key": "value"}'

    thought, action, action_input = parser.parse_default_action(output)

    assert thought == "thinking"
    assert action == "Tool"
    assert action_input == {"key": "value"}


def test_parse_default_action_multiple_newlines():
    """Test parsing with multiple newlines between Action and Input."""
    output = """Thought: thinking

Action: Tool Name


Action Input: {"key": "value"}"""

    thought, action, action_input = parser.parse_default_action(output)

    assert thought == "thinking"
    assert action == "Tool Name"
    assert action_input == {"key": "value"}


def test_parse_default_action_with_json_markdown():
    """Test parsing when JSON is wrapped in markdown blocks."""
    output = """Thought: thinking
Action: Tool
Action Input: ```json
{"key": "value"}
```"""
    thought, action, action_input = parser.parse_default_action(output)

    assert thought == "thinking"
    assert action == "Tool"
    assert action_input == {"key": "value"}


def test_parse_default_action_nested_json():
    """Test parsing with deeply nested JSON structures."""
    output = """Thought: Complex data structure
Action: API Call
Action Input: {"params": {"nested": {"deeply": {"key": "value"}}, "list": [1, 2, 3]}}"""

    thought, action, action_input = parser.parse_default_action(output)

    assert thought == "Complex data structure"
    assert action == "API Call"
    assert action_input == {"params": {"nested": {"deeply": {"key": "value"}}, "list": [1, 2, 3]}}


def test_parse_default_action_json_with_list():
    """Test parsing with a list."""
    output = """Thought: Complex data structure
Action: API Call
Action Input: {"params": {"nested": {"deeply": {"key": "value"}}, "list": [1, 2, 3]}}"""

    thought, action, action_input = parser.parse_default_action(output)

    assert thought == "Complex data structure"
    assert action == "API Call"
    assert action_input == {"params": {"nested": {"deeply": {"key": "value"}}, "list": [1, 2, 3]}}


def test_parse_default_action_json_with_escaped_quotes():
    """Test parsing JSON containing escaped quotes."""
    output = """Thought: thinking
Action: Tool
Action Input: {"message": "He said \\"hello\\" to me"}"""

    thought, action, action_input = parser.parse_default_action(output)

    assert thought == "thinking"
    assert action == "Tool"
    assert action_input == {"message": 'He said "hello" to me'}


def test_parse_default_action_invalid_json():
    """Test that invalid JSON raises ActionParsingException."""
    output = """Thought: thinking
Action: Tool
Action Input: {invalid json}"""

    with pytest.raises(ActionParsingException) as excinfo:
        parser.parse_default_action(output)

    assert excinfo.value.recoverable


def test_parse_default_action_missing_action():
    """Test that missing Action raises ActionParsingException."""
    output = """Thought: thinking
Action Input: {"key": "value"}"""

    with pytest.raises(ActionParsingException) as excinfo:
        parser.parse_default_action(output)

    assert excinfo.value.recoverable


def test_parse_default_thought():
    """Test extraction of thought from output."""
    output = """Thought: I need to think about this carefully
Action: Tool
Action Input: {"key": "value"}"""

    thought = parser.parse_default_thought(output)
    assert thought == "I need to think about this carefully"


def test_parse_default_thought_multiline():
    """Test extraction of multiline thought."""
    output = """Thought: I need to think about this carefully
and consider multiple factors
before deciding
Action: Tool
Action Input: {"key": "value"}"""

    thought = parser.parse_default_thought(output)
    assert "I need to think about this carefully" in thought
    assert "and consider multiple factors" in thought
    assert "before deciding" in thought


def test_extract_default_final_answer():
    """Test extraction of final answer."""
    output = """Thought: I have all the information I need
Answer: The final answer is 42"""

    thought, answer, output_files = parser.extract_default_final_answer(output)
    assert thought == "I have all the information I need"
    assert answer == "The final answer is 42"
    assert output_files == ""


def test_extract_default_final_answer_multiline():
    """Test extraction of multiline final answer."""
    output = """Thought: I have all the information I need
Answer: The final answer is:
1. First point
2. Second point
3. Third point"""

    thought, answer, output_files = parser.extract_default_final_answer(output)
    assert thought == "I have all the information I need"
    assert "The final answer is:" in answer
    assert "1. First point" in answer
    assert "3. Third point" in answer
    assert output_files == ""


def test_extract_default_final_answer_with_output_files():
    """Test extraction of final answer with Output Files field."""
    output = """Thought: I created the requested files.
Output Files: /home/user/result.txt, /home/user/data.csv
Answer: Here are your files."""

    thought, answer, output_files = parser.extract_default_final_answer(output)
    assert thought == "I created the requested files."
    assert answer == "Here are your files."
    assert output_files == "/home/user/result.txt, /home/user/data.csv"


def test_extract_default_final_answer_without_output_files():
    """Test that Output Files is empty when not present."""
    output = """Thought: No files needed.
Answer: The answer is 42."""

    thought, answer, output_files = parser.extract_default_final_answer(output)
    assert thought == "No files needed."
    assert answer == "The answer is 42."
    assert output_files == ""
