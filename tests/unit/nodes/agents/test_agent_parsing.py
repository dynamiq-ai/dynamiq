import pytest

from dynamiq.connections import BaseConnection
from dynamiq.nodes.agents.agent import Agent
from dynamiq.nodes.llms.base import BaseLLM


class MockConnection(BaseConnection):
    def connect(self):
        pass


class MockLLM(BaseLLM):
    def __init__(self):
        super().__init__(model="mock-model", connection=MockConnection(), name="MockLLM")


class TestAgentParsing:
    @pytest.fixture
    def agent(self):
        llm = MockLLM()
        # Agent requires tools usually, but for parsing logic it might be optional or empty list is fine
        agent = Agent(llm=llm, name="TestAgent", tools=[])
        return agent

    def test_parse_action_deepseek_format(self, agent):
        """Test parsing of DeepSeek-style output with extra newlines and nested JSON."""
        # Use raw string to prevent python from interpreting \n as newline char
        output = r"""Thought: I need to calculate something.

Action: Calculator Tool

Action Input: {
    "code": "print('Hello World')\nprint({'nested': 'dict'})"
}"""

        thought, action, action_input = agent._parse_action(output)

        assert thought == "I need to calculate something."
        assert action == "Calculator Tool"
        assert action_input == {"code": "print('Hello World')\nprint({'nested': 'dict'})"}

    def test_parse_action_standard_format(self, agent):
        """Test parsing of standard strict format."""
        output = 'Thought: thinking\nAction: Tool\nAction Input: {"key": "value"}'

        thought, action, action_input = agent._parse_action(output)

        assert thought == "thinking"
        assert action == "Tool"
        assert action_input == {"key": "value"}

    def test_parse_action_multiple_newlines(self, agent):
        """Test parsing with multiple newlines between Action and Input."""
        output = """Thought: thinking

Action: Tool Name


Action Input: {"key": "value"}"""

        thought, action, action_input = agent._parse_action(output)

        assert action == "Tool Name"
        assert action_input == {"key": "value"}

    def test_parse_action_with_json_markdown(self, agent):
        """Test parsing when JSON is wrapped in markdown blocks."""
        output = """Thought: thinking
Action: Tool
Action Input: ```json
{"key": "value"}
```"""
        thought, action, action_input = agent._parse_action(output)
        assert action == "Tool"
        assert action_input == {"key": "value"}
