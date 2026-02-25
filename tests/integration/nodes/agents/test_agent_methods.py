import uuid

import pytest

from dynamiq import connections, prompts
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.components.parser import (
    extract_default_final_answer,
    parse_default_action,
    parse_default_thought,
)
from dynamiq.nodes.agents.exceptions import (
    ActionParsingException,
    JSONParsingError,
    ParsingError,
    TagNotFoundError,
    XMLParsingError,
)
from dynamiq.nodes.agents.utils import XMLParser
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.tools.todo_tools import TodoWriteTool
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableResult, RunnableStatus
from dynamiq.storages.file.base import FileStoreConfig
from dynamiq.storages.file.in_memory import InMemoryFileStore


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
    """Agent with DEFAULT inference mode."""
    return Agent(name="Test Agent", llm=openai_node, tools=[], inference_mode=InferenceMode.DEFAULT)


@pytest.fixture
def xml_react_agent(openai_node, mock_llm_executor):
    """Agent with XML inference mode."""
    return Agent(name="Test XML Agent", llm=openai_node, tools=[], inference_mode=InferenceMode.XML)


@pytest.fixture
def mock_tool():
    """A real Node-based tool for testing."""
    from dynamiq.nodes.tools.python import Python

    return Python(
        name="Calculator",
        description="A simple calculator tool for performing arithmetic operations",
        code="""
def run(input_data):
    result = eval(input_data.get('expression', '0'))
    return {"result": result}
""",
    )


def test_parse_default_thought(default_react_agent):
    """Test extracting thought from agent output."""
    output = """
    Thought: I need to search for information about the weather.
    Action: search
    Action Input: {"query": "weather in San Francisco"}
    """
    thought = parse_default_thought(output)
    assert thought == "I need to search for information about the weather."


def test_parse_default_action_missing_action_input(default_react_agent):
    """Test parsing with missing action input raises an exception."""
    output = """
    Thought: I need to search for information about the weather.
    Action: search
    """
    with pytest.raises(ActionParsingException):
        parse_default_action(output)


def test_extract_default_final_answer(default_react_agent):
    """Test extracting the final answer from the output."""
    output = """
    Thought: I found all the information needed.
    Answer: The weather in San Francisco is foggy with a high of 65°F.
    """
    answer = extract_default_final_answer(output)
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
    agent = Agent(name="XMLPromptAgent", llm=openai_node, tools=[], inference_mode=InferenceMode.XML)

    prompt = agent.generate_prompt()

    assert "<output>" in prompt
    assert "<thought>" in prompt
    assert "<answer>" in prompt
    assert "Always use this exact XML format" in prompt


def test_set_prompt_block(openai_node, mock_llm_executor):
    """Test modifying prompt blocks."""
    agent = Agent(name="PromptBlockTestAgent", llm=openai_node, tools=[], inference_mode=InferenceMode.DEFAULT)

    custom_instructions = "Your goal is to analyze the given text and identify key points."
    agent.set_block("instructions", custom_instructions)

    prompt = agent.generate_prompt()

    assert custom_instructions in prompt

    custom_context = "You are a helpful scientific research assistant."
    agent.set_block("context", custom_context)

    prompt = agent.generate_prompt()

    assert custom_instructions in prompt
    assert custom_context in prompt


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


def test_xmlparser_parse_json_with_literal_newlines_repaired():
    """Literal newlines inside JSON string values are repaired so parsing succeeds."""
    text = '<output><thought>Run code</thought><action>code-executor</action><action_input>{"python": "import pandas as pd\nprint(1)"}</action_input></output>'  # noqa E501
    result = XMLParser.parse(text, required_tags=["thought", "action", "action_input"], json_fields=["action_input"])
    assert result["action_input"] == {"python": "import pandas as pd\nprint(1)"}


def test_xmlparser_parse_action_input_with_less_than_and_ampersand():
    """Content with < (e.g. Python comparison) or & (e.g. shell &&) inside <action_input> must not break the parser."""
    text = '<output><thought>Compute loss</thought><action>code-executor</action><action_input>{"python": "loss_count = (moic_col < 1.0).sum()"}</action_input></output>'  # noqa E501
    result = XMLParser.parse(text, required_tags=["thought", "action", "action_input"], json_fields=["action_input"])
    assert result["action_input"] == {"python": "loss_count = (moic_col < 1.0).sum()"}
    # Shell with &&
    text2 = '<output><thought>List dirs</thought><action>code-executor</action><action_input>{"shell_command": "ls -la /home/user/input && ls -la /home/user/output"}</action_input></output>'  # noqa E501
    result2 = XMLParser.parse(text2, required_tags=["thought", "action", "action_input"], json_fields=["action_input"])
    assert result2["action_input"] == {"shell_command": "ls -la /home/user/input && ls -la /home/user/output"}


def test_xmlparser_parse_action_input_escaped_entities_unescaped():
    """When model sends &lt; &gt; &amp; in JSON strings, parser unescapes so tool receives real < > &."""
    text = '<output><thought>Run</thought><action>code-executor</action><action_input>{"python": "x = (a &lt; 1).sum()"}</action_input></output>'  # noqa E501
    result = XMLParser.parse(text, required_tags=["thought", "action", "action_input"], json_fields=["action_input"])
    assert result["action_input"] == {"python": "x = (a < 1).sum()"}


def test_xmlparser_parse_json_with_literal_crlf_newlines_repaired():
    """CRLF line breaks inside JSON string values are repaired and parsed."""
    text = '<output><thought>Run code</thought><action>code-executor</action><action_input>{"python": "line1\r\nline2"}</action_input></output>'  # noqa E501
    result = XMLParser.parse(text, required_tags=["thought", "action", "action_input"], json_fields=["action_input"])
    assert result["action_input"] == {"python": "line1\nline2"}


def test_xmlparser_parse_action_input_entities_unescaped_recursively():
    """HTML entities are unescaped recursively inside nested JSON dict/list values."""
    text = (
        "<output><thought>Run</thought><action>code-executor</action><action_input>"
        '{"commands": ["echo start &amp;&amp; echo done"], "meta": {"expr": "(x &gt; 1) &amp;&amp; (y &lt; 3)"}}'
        "</action_input></output>"
    )
    result = XMLParser.parse(text, required_tags=["thought", "action", "action_input"], json_fields=["action_input"])
    assert result["action_input"] == {
        "commands": ["echo start && echo done"],
        "meta": {"expr": "(x > 1) && (y < 3)"},
    }


def test_xmlparser_parse_json_invalid_after_repair_raises():
    """Invalid JSON syntax should still raise JSONParsingError after newline/entity repair attempts."""
    text = "<output><thought>Run</thought><action_input>{'python': 'print(1)'}</action_input></output>"
    with pytest.raises(JSONParsingError, match="Failed to parse JSON content for field 'action_input'"):
        XMLParser.parse(text, required_tags=["thought", "action_input"], json_fields=["action_input"])


def test_agent_xml_mode_invalid_action_input_json_is_recoverable(xml_react_agent):
    """Malformed action_input JSON should surface as recoverable ActionParsingException in XML mode."""
    llm_generated_output = (
        "<output><thought>Run code</thought><action>code-executor</action>"
        '<action_input>{"python": "print(1)"</action_input></output>'
    )

    with pytest.raises(ActionParsingException, match="must be valid JSON") as excinfo:
        xml_react_agent._handle_xml_mode(llm_generated_output=llm_generated_output, loop_num=1, config=RunnableConfig())

    assert excinfo.value.recoverable is True


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

    assert "# Research & Development (R&D)" in result["answer"]
    assert "R&D departments" in result["answer"]
    assert "AI & Machine Learning" in result["answer"]
    assert "14% & 20%" in result["answer"]


def test_xmlparser_parse_escapes_reserved_tag_mentions():
    """Parser should escape reserved tag mentions inside text nodes."""
    text = """<output>
  <thought>
    The previous attempt failed because the model mentioned the <answer> tag without closing it properly.
  </thought>
  <answer>
    Proper final answer content.
  </answer>
</output>"""

    result = XMLParser.parse(text, required_tags=["thought", "answer"])

    assert "&lt;answer&gt;" in result["thought"]
    assert "Proper final answer content." in result["answer"]


def test_xmlparser_parse_escapes_stray_closing_tag_mentions():
    """Stray closing tags in prose should be escaped and not break parsing."""
    text = """<output>
  <thought>
    Reminder: never output </answer> before you are done.
  </thought>
  <answer>
    Another final answer.
  </answer>
</output>"""

    result = XMLParser.parse(text, required_tags=["thought", "answer"])

    assert "&lt;/answer&gt;" in result["thought"]
    assert "Another final answer." in result["answer"]


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


def test_generate_structured_output_schemas(openai_node, mock_tool):
    """Test structured output schema generation."""
    from dynamiq.nodes.agents.components.schema_generator import generate_structured_output_schemas

    agent = Agent(name="Test Agent", llm=openai_node, tools=[mock_tool], inference_mode=InferenceMode.DEFAULT)

    schema = generate_structured_output_schemas(
        tools=[mock_tool], sanitize_tool_name=agent.sanitize_tool_name, delegation_allowed=False
    )

    # Verify schema structure
    assert "type" in schema
    assert schema["type"] == "json_schema"
    assert "json_schema" in schema

    json_schema = schema["json_schema"]
    assert json_schema["name"] == "plan_next_action"
    assert json_schema["strict"] is True

    # Verify required fields
    assert set(json_schema["schema"]["required"]) == {"thought", "action", "action_input"}

    # Verify properties
    properties = json_schema["schema"]["properties"]
    assert "thought" in properties
    assert "action" in properties
    assert "action_input" in properties


def test_generate_function_calling_schemas(openai_node, mock_tool):
    """Test function calling schema generation."""
    from dynamiq.nodes.agents.components.schema_generator import generate_function_calling_schemas

    agent = Agent(name="Test Agent", llm=openai_node, tools=[mock_tool], inference_mode=InferenceMode.FUNCTION_CALLING)

    schemas = generate_function_calling_schemas(
        tools=[mock_tool], delegation_allowed=False, sanitize_tool_name=agent.sanitize_tool_name, llm=openai_node
    )

    # Should have at least final answer function + mock tool
    assert len(schemas) >= 2

    # First schema should be the final answer function
    final_answer_schema = schemas[0]
    assert final_answer_schema["type"] == "function"
    assert final_answer_schema["function"]["name"] == "provide_final_answer"
    assert "thought" in final_answer_schema["function"]["parameters"]["properties"]
    assert "answer" in final_answer_schema["function"]["parameters"]["properties"]

    # Verify all schemas have required structure
    for schema in schemas:
        assert "type" in schema
        assert schema["type"] == "function"
        assert "function" in schema
        assert "name" in schema["function"]
        assert "parameters" in schema["function"]
        assert "properties" in schema["function"]["parameters"]


def test_agent_injects_file_store_into_python_code_executor(openai_node, mock_llm_executor):
    """Regression test: Agent should inject file_store into PythonCodeExecutor tools at runtime."""
    from dynamiq.nodes.tools.python_code_executor import PythonCodeExecutor
    from dynamiq.storages.file.base import FileStoreConfig
    from dynamiq.storages.file.in_memory import InMemoryFileStore

    executor = PythonCodeExecutor(name="TestExecutor")
    assert executor.file_store is None, "PythonCodeExecutor should allow None file_store"

    file_store_backend = InMemoryFileStore()
    agent = Agent(
        name="TestAgent",
        llm=openai_node,
        tools=[executor],
        file_store=FileStoreConfig(enabled=True, backend=file_store_backend),
    )

    assert agent.file_store_backend is file_store_backend

    tool = next((t for t in agent.tools if isinstance(t, PythonCodeExecutor)), None)
    assert tool is not None, "PythonCodeExecutor should be in agent's tools"

    if not tool.file_store:
        tool.file_store = agent.file_store_backend

    assert tool.file_store is file_store_backend, "Agent should inject file_store into PythonCodeExecutor"


def test_todo_tools_added_when_enabled(openai_node, mock_llm_executor):
    """Test that todo tools are added and instructions included when file_store.todo_enabled is True."""

    file_store_config = FileStoreConfig(enabled=True, backend=InMemoryFileStore(), todo_enabled=True)
    agent = Agent(
        name="Todo Agent",
        llm=openai_node,
        tools=[],
        file_store=file_store_config,
        inference_mode=InferenceMode.DEFAULT,
    )

    # Check that TodoWriteTool was automatically added
    todo_tools = [t for t in agent.tools if isinstance(t, TodoWriteTool)]
    assert len(todo_tools) == 1, "TodoWriteTool should be automatically added"

    # Check that todo instructions are in the prompt
    prompt = agent.generate_prompt()
    assert "TODO MANAGEMENT" in prompt, "TODO instructions should be in system prompt"
    assert "todo-write" in prompt, "todo-write tool should be mentioned in prompt"


def test_agent_state_updates_with_todos(openai_node):
    """Test that AgentState correctly updates and serializes todo state."""
    from dynamiq.nodes.agents.agent import AgentState
    from dynamiq.nodes.tools.todo_tools import TodoItem, TodoStatus

    state = AgentState()

    assert state.to_prompt_string() == ""

    state.max_loops = 15
    state.update_loop(3)
    prompt_str = state.to_prompt_string()
    assert "Progress: Loop 3/15" in prompt_str

    # Update with todos (accepts both dicts and TodoItem objects)
    state.update_todos(
        [
            TodoItem(id="1", content="First task", status=TodoStatus.COMPLETED),
            TodoItem(id="2", content="Second task", status=TodoStatus.IN_PROGRESS),
            TodoItem(id="3", content="Third task", status=TodoStatus.PENDING),
        ]
    )
    prompt_str = state.to_prompt_string()
    assert "[+] 1: First task" in prompt_str, "Completed todo should show [+]"
    assert "[~] 2: Second task" in prompt_str, "In-progress todo should show [~]"
    assert "[ ] 3: Third task" in prompt_str, "Pending todo should show [ ]"

    # Verify todos are TodoItem instances
    assert all(isinstance(t, TodoItem) for t in state.todos)

    # Reset should clear state
    state.reset(max_loops=10)
    assert state.current_loop == 0
    assert state.max_loops == 10
    assert state.todos == []


class TestParallelToolCloning:
    """Test parallel tool cloning."""

    @pytest.fixture
    def calculator_tool(self):
        return Python(
            name="Calculator",
            description="A simple calculator tool for performing arithmetic operations",
            code="",
            is_parallel_execution_allowed=True,
        )

    @pytest.fixture
    def formatter_tool(self):
        return Python(
            name="Formatter",
            description="A text formatting tool",
            code="",
            is_parallel_execution_allowed=True,
        )

    @pytest.fixture
    def parallel_agent(self, openai_node, mock_llm_executor, calculator_tool, formatter_tool):
        """Agent with parallel_tool_calls_enabled=True and two tools."""
        return Agent(
            name="Parallel Clone Agent",
            llm=openai_node,
            tools=[calculator_tool, formatter_tool],
            inference_mode=InferenceMode.XML,
            parallel_tool_calls_enabled=True,
        )

    # -- Low-level: _run_tool -------------------------------------------------

    def test_run_tool_does_not_clone_for_sequential_execution(self, parallel_agent, calculator_tool, mocker):
        """_run_tool must NOT clone when is_parallel is False (the default),
        even when the agent has parallel_tool_calls_enabled=True."""
        config = RunnableConfig()
        clone_spy = mocker.patch.object(
            parallel_agent,
            "_clone_tool_for_execution",
            return_value=(calculator_tool, config),
        )
        mock_result = RunnableResult(status=RunnableStatus.SUCCESS, output={"content": "42"})
        mocker.patch.object(Python, "run", return_value=mock_result)

        parallel_agent._run_tool(calculator_tool, {"expression": "1+1"}, config, is_parallel=False)

        assert clone_spy.call_count == 0, "_clone_tool_for_execution must not be called for sequential execution"

    def test_run_tool_clones_for_parallel_execution(self, parallel_agent, calculator_tool, mocker):
        """_run_tool must clone the tool when is_parallel=True."""
        config = RunnableConfig()
        clone_spy = mocker.patch.object(
            parallel_agent,
            "_clone_tool_for_execution",
            return_value=(calculator_tool, config),
        )
        mock_result = RunnableResult(status=RunnableStatus.SUCCESS, output={"content": "42"})
        mocker.patch.object(Python, "run", return_value=mock_result)

        parallel_agent._run_tool(calculator_tool, {"expression": "1+1"}, config, is_parallel=True)

        assert clone_spy.call_count == 1, "_clone_tool_for_execution must be called for parallel execution"

    def test_execute_tools_single_tool_does_not_pass_is_parallel(self, parallel_agent, mocker):
        """When _execute_tools dispatches a single tool it must NOT set
        is_parallel=True on the inner _execute_single_tool call."""
        mock_single = mocker.patch.object(
            parallel_agent,
            "_execute_single_tool",
            return_value=("result", [], False, True, {"node": "dep"}),
        )

        tools_data = [{"name": "Calculator", "input": {"expression": "1+1"}}]
        parallel_agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        mock_single.assert_called_once()
        call_kw = mock_single.call_args.kwargs
        assert call_kw.get("is_parallel", False) is False, "Single-tool dispatch must not set is_parallel=True"

    def test_execute_tools_multiple_tools_passes_is_parallel(self, parallel_agent, mocker):
        """When _execute_tools dispatches multiple tools concurrently it must
        set is_parallel=True on every _execute_single_tool call."""
        mock_single = mocker.patch.object(
            parallel_agent,
            "_execute_single_tool",
            return_value=("result", [], False, True, {"node": "dep"}),
        )

        tools_data = [
            {"name": "Calculator", "input": {"expression": "1+1"}},
            {"name": "Formatter", "input": {"text": "hello"}},
        ]
        parallel_agent._execute_tools(tools_data, "thinking", 1, RunnableConfig())

        assert mock_single.call_count == 2
        for call in mock_single.call_args_list:
            assert (
                call.kwargs.get("is_parallel") is True
            ), "Concurrent dispatch must set is_parallel=True for every tool"
