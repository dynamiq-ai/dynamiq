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


def test_extract_default_final_answer_output_files_phrase_in_thought():
    """'Output Files:' embedded in thought text must not be treated as the output-files field."""
    output = (
        "Thought: I checked the Output Files: section and found nothing relevant.\n"
        "Answer: No files found."
    )

    thought, answer, output_files = parser.extract_default_final_answer(output)
    assert thought == "I checked the Output Files: section and found nothing relevant."
    assert answer == "No files found."
    assert output_files == ""


def test_extract_default_final_answer_output_files_in_thought_and_real_field():
    """When 'Output Files:' appears both in the thought and as a real field, the real field wins."""
    output = (
        "Thought: I noted the Output Files: field is important.\n"
        "Output Files: /tmp/result.csv\n"
        "Answer: Done."
    )

    thought, answer, output_files = parser.extract_default_final_answer(output)
    assert thought == "I noted the Output Files: field is important."
    assert answer == "Done."
    assert output_files == "/tmp/result.csv"


def test_action_parsing_failure_emits_tool_input_error_event(mocker):
    """When the LLM returns unparseable output, the agent should naturally
    emit a tool_input_error streaming event via _emit_tool_input_error."""
    import uuid

    from dynamiq import connections, prompts
    from dynamiq.nodes.agents import Agent
    from dynamiq.nodes.agents.agent import Behavior
    from dynamiq.nodes.llms import OpenAI
    from dynamiq.nodes.tools.python import Python
    from dynamiq.nodes.types import InferenceMode
    from dynamiq.runnables import RunnableConfig
    from dynamiq.types.streaming import AgentToolInputErrorEventMessageData, StreamingConfig, StreamingMode

    conn = connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key")
    llm = OpenAI(
        name="TestLLM",
        model="gpt-4o-mini",
        connection=conn,
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )
    tool = Python(name="dummy", description="dummy tool", code="def run(inputs): return {}")

    agent = Agent(
        name="test-agent",
        llm=llm,
        tools=[tool],
        inference_mode=InferenceMode.STRUCTURED_OUTPUT,
        streaming=StreamingConfig(enabled=True, mode=StreamingMode.ALL),
        max_loops=2,
        behaviour_on_max_loops=Behavior.RETURN,
    )

    bad_text = "not valid json"
    mocker.patch(
        "dynamiq.nodes.llms.base.BaseLLM._completion",
        side_effect=lambda stream=False, *a, **kw: (
            _mock_llm_stream(bad_text) if stream else _mock_llm_response(bad_text)
        ),
    )

    spy = mocker.patch.object(agent, "_stream_agent_event", wraps=agent._stream_agent_event)

    agent.run(input_data={"input": "test"}, config=RunnableConfig())

    error_calls = [c for c in spy.call_args_list if c[0][1] == "tool_input_error"]
    assert len(error_calls) >= 1
    event_data = error_calls[0][0][0]
    assert isinstance(event_data, AgentToolInputErrorEventMessageData)
    assert event_data.name == "test-agent"


def test_structured_output_multiple_jsons_takes_first(mocker):
    """When LLM returns multiple JSON objects, only the first should be parsed."""
    import uuid

    from dynamiq import connections, prompts
    from dynamiq.nodes.agents import Agent
    from dynamiq.nodes.llms import OpenAI
    from dynamiq.nodes.types import InferenceMode

    conn = connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key")
    llm = OpenAI(
        name="TestLLM",
        model="gpt-4o-mini",
        connection=conn,
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )
    agent = Agent(name="test-agent", llm=llm, tools=[], inference_mode=InferenceMode.STRUCTURED_OUTPUT)

    output = (
        '{"thought": "run shell", "action": "SandboxShellTool", '
        '"action_input": "{\\"command\\": \\"ls\\"}", "output_files": ""}\n'
        '{"thought": "wait", "action": "finish", "action_input": "done", "output_files": ""}'
    )
    thought, action, action_input = agent._handle_structured_output_mode(output, loop_num=1)
    assert thought == "run shell"
    assert action == "SandboxShellTool"
    assert action_input == {"command": "ls"}


def test_structured_output_action_input_with_literal_newlines():
    """strict=False allows action_input containing literal newlines (common LLM mistake)."""
    import uuid

    from dynamiq import connections, prompts
    from dynamiq.nodes.agents import Agent
    from dynamiq.nodes.llms import OpenAI
    from dynamiq.nodes.types import InferenceMode

    conn = connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key")
    llm = OpenAI(
        name="TestLLM",
        model="gpt-4o-mini",
        connection=conn,
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )
    agent = Agent(name="test-agent", llm=llm, tools=[], inference_mode=InferenceMode.STRUCTURED_OUTPUT)

    # action_input is a JSON string with a literal newline inside (not escaped as \\n).
    # This is what LLMs produce for multi-line shell commands / code.
    output = (
        '{"thought": "run script", "action": "SandboxShellTool", '
        '"action_input": "{\\"command\\": \\"echo hello\\necho world\\"}", "output_files": ""}'
    )
    thought, action, action_input = agent._handle_structured_output_mode(output, loop_num=1)
    assert thought == "run script"
    assert action == "SandboxShellTool"
    assert action_input == {"command": "echo hello\necho world"}


def test_structured_output_fallback_decoder_with_literal_newlines():
    """Fallback JSONDecoder(strict=False) handles multiple concatenated JSONs with literal newlines."""
    import uuid

    from dynamiq import connections, prompts
    from dynamiq.nodes.agents import Agent
    from dynamiq.nodes.llms import OpenAI
    from dynamiq.nodes.types import InferenceMode

    conn = connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key")
    llm = OpenAI(
        name="TestLLM",
        model="gpt-4o-mini",
        connection=conn,
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )
    agent = Agent(name="test-agent", llm=llm, tools=[], inference_mode=InferenceMode.STRUCTURED_OUTPUT)

    # Two concatenated JSON objects where the first has a literal newline in action_input.
    # json.loads fails (extra data), fallback raw_decode must also use strict=False.
    output = (
        '{"thought": "write file", "action": "SandboxShellTool", '
        '"action_input": "{\\"command\\": \\"cat > f.py\\nprint(1)\\"}", "output_files": ""}'
        '\n{"thought": "done", "action": "finish", "action_input": "ok", "output_files": ""}'
    )
    thought, action, action_input = agent._handle_structured_output_mode(output, loop_num=1)
    assert thought == "write file"
    assert action == "SandboxShellTool"
    assert action_input == {"command": "cat > f.py\nprint(1)"}


def test_function_calling_action_input_with_literal_newlines(mocker):
    """FC mode: strict=False allows action_input with literal newlines."""
    import uuid

    from dynamiq import connections, prompts
    from dynamiq.nodes.agents import Agent
    from dynamiq.nodes.llms import OpenAI
    from dynamiq.nodes.types import InferenceMode

    conn = connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key")
    llm = OpenAI(
        name="TestLLM",
        model="gpt-4o-mini",
        connection=conn,
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )
    agent = Agent(name="test-agent", llm=llm, tools=[], inference_mode=InferenceMode.FUNCTION_CALLING)

    # Simulate an LLM result with a tool_call whose action_input has a literal newline
    mock_result = mocker.MagicMock()
    mock_result.output = {
        "tool_calls": [
            {
                "function": {
                    "name": "SandboxShellTool",
                    "arguments": {"thought": "run it", "action_input": '{"cmd": "ls\nls -la"}'},
                }
            }
        ]
    }

    thought, action, action_input = agent._handle_function_calling_mode(mock_result, loop_num=1)
    assert thought == "run it"
    assert action == "SandboxShellTool"
    assert action_input == {"cmd": "ls\nls -la"}


def _mock_llm_response(text: str):
    from litellm import ModelResponse

    model_r = ModelResponse()
    model_r["choices"][0]["message"]["content"] = text
    return model_r


def _mock_llm_stream(text: str):
    from litellm import ModelResponse
    from litellm.utils import Delta

    for char in text:
        model_r = ModelResponse(stream=True)
        model_r.choices[0].delta = Delta(role="assistant", content=char)
        yield model_r


def _make_agent(inference_mode=None, response_format=None):
    """Build a minimal Agent suitable for in-process unit tests."""
    import uuid

    from dynamiq import connections, prompts
    from dynamiq.nodes.agents import Agent
    from dynamiq.nodes.llms import OpenAI
    from dynamiq.nodes.types import InferenceMode

    conn = connections.OpenAI(id=str(uuid.uuid4()), api_key="fake-key")
    llm = OpenAI(
        name="TestLLM",
        model="gpt-4o-mini",
        connection=conn,
        prompt=prompts.Prompt(messages=[prompts.Message(role="user", content="{{input}}")]),
    )
    return Agent(
        name="test-agent",
        llm=llm,
        tools=[],
        inference_mode=inference_mode or InferenceMode.DEFAULT,
        response_format=response_format,
    )


class _DocumentModel:
    """Placeholder — real model is defined inside tests to avoid import-time side effects."""


def test_build_final_answer_function_schema_default_unchanged():
    """With no response_format the schema is the existing FINAL_ANSWER_FUNCTION_SCHEMA."""
    from dynamiq.nodes.agents.components import schema_generator

    assert schema_generator.build_final_answer_function_schema(None) is schema_generator.FINAL_ANSWER_FUNCTION_SCHEMA


def test_build_final_answer_function_schema_with_pydantic_model():
    """Pydantic classes are expanded and replace the answer property."""
    from pydantic import BaseModel

    from dynamiq.nodes.agents.components import schema_generator

    class Doc(BaseModel):
        title: str
        tags: list[str]

    schema = schema_generator.build_final_answer_function_schema(Doc)
    props = schema["function"]["parameters"]["properties"]
    assert props["answer"]["type"] == "object"
    assert set(props["answer"]["properties"].keys()) == {"title", "tags"}
    assert props["thought"]["type"] == "string"
    assert props["output_files"]["type"] == "string"


def test_build_final_answer_function_schema_unwraps_dict_wrapper():
    """When a response_format dict uses litellm's json_schema wrapper it is unwrapped."""
    from dynamiq.nodes.agents.components import schema_generator

    wrapped = {
        "type": "json_schema",
        "json_schema": {
            "name": "x",
            "schema": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
        },
    }
    schema = schema_generator.build_final_answer_function_schema(wrapped)
    answer = schema["function"]["parameters"]["properties"]["answer"]
    assert answer["type"] == "object"
    assert "title" in answer["properties"]


def test_agent_default_returns_string_without_response_format():
    """Regression: agents without response_format still return strings."""
    agent = _make_agent()
    assert agent.response_format is None
    # _coerce_to_response_format is a no-op
    assert agent._coerce_to_response_format("plain string") == "plain string"


def test_agent_coerce_to_dict_schema():
    """Dict response_format parses JSON string into a dict."""
    schema = {
        "type": "object",
        "properties": {"title": {"type": "string"}},
        "required": ["title"],
        "additionalProperties": False,
    }
    agent = _make_agent(response_format=schema)
    result = agent._coerce_to_response_format('{"title": "Harry Potter"}')
    assert result == {"title": "Harry Potter"}


def test_agent_coerce_to_pydantic_instance():
    """BaseModel input is normalized to its JSON schema dict; coerce returns a parsed dict
    that the caller can validate back into the original BaseModel."""
    from pydantic import BaseModel

    class Doc(BaseModel):
        title: str
        tags: list[str]

    agent = _make_agent(response_format=Doc)
    # BaseModel class input is converted to dict schema on field validation.
    assert isinstance(agent.response_format, dict)
    assert "title" in agent.response_format["properties"]

    result = agent._coerce_to_response_format('{"title": "HP", "tags": ["a", "b"]}')
    assert result == {"title": "HP", "tags": ["a", "b"]}

    # Callers who want a typed object keep their original BaseModel class and validate.
    doc = Doc.model_validate(result)
    assert isinstance(doc, Doc)
    assert doc.title == "HP"
    assert doc.tags == ["a", "b"]


def test_agent_coerce_handles_markdown_fenced_json():
    """The coercion strips Markdown code fences around JSON."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    agent = _make_agent(response_format=schema)
    result = agent._coerce_to_response_format('```json\n{"x": 7}\n```')
    assert result == {"x": 7}


def test_agent_coerce_accepts_already_parsed_dict():
    """FUNCTION_CALLING already parses arguments to dict; coerce must pass it through."""
    from pydantic import BaseModel

    class Doc(BaseModel):
        title: str

    agent = _make_agent(response_format=Doc)
    result = agent._coerce_to_response_format({"title": "HP"})
    assert result == {"title": "HP"}


def test_agent_coerce_raises_on_invalid_json():
    """Malformed final answer raises a ParsingError with a clear message."""
    from dynamiq.nodes.agents.exceptions import ParsingError

    schema = {"type": "object"}
    agent = _make_agent(response_format=schema)
    with pytest.raises(ParsingError):
        agent._coerce_to_response_format("definitely not json")


def test_agent_response_format_injects_prompt_instructions():
    """Non-FUNCTION_CALLING modes inline the response_format schema into the rendered prompt."""
    from dynamiq.nodes.types import InferenceMode

    schema = {"type": "object", "properties": {"title": {"type": "string"}}}
    agent = _make_agent(inference_mode=InferenceMode.DEFAULT, response_format=schema)
    rendered = agent.generate_prompt()
    assert "MUST be a valid JSON document" in rendered
    assert "title" in rendered


def test_agent_response_format_function_calling_uses_schema_in_tools():
    """FUNCTION_CALLING mode swaps the answer schema in the final-answer function."""
    from pydantic import BaseModel

    from dynamiq.nodes.types import InferenceMode

    class Doc(BaseModel):
        title: str

    agent = _make_agent(inference_mode=InferenceMode.FUNCTION_CALLING, response_format=Doc)
    final_answer_schema = next(s for s in agent._tools if s.get("function", {}).get("name") == "provide_final_answer")
    answer = final_answer_schema["function"]["parameters"]["properties"]["answer"]
    assert answer["type"] == "object"
    assert "title" in answer["properties"]


def test_agent_response_format_structured_output_schema_unchanged():
    """STRUCTURED_OUTPUT keeps its simple `action_input: string` schema; the
    user's response_format is enforced via prompt injection + coerce instead."""
    from pydantic import BaseModel

    from dynamiq.nodes.types import InferenceMode

    class Doc(BaseModel):
        title: str
        tags: list[str]

    agent = _make_agent(inference_mode=InferenceMode.STRUCTURED_OUTPUT, response_format=Doc)
    schema = agent._response_format["json_schema"]["schema"]
    assert schema["properties"]["action_input"] == {
        "type": "string",
        "description": "Input for chosen action.",
    }
    rendered = agent.generate_prompt()
    assert "MUST be a valid JSON document" in rendered
    assert "title" in rendered
    assert "tags" in rendered


def test_agent_structured_output_finish_with_json_string_action_input():
    """STRUCTURED_OUTPUT finish emits a JSON string; coerce parses it into a dict."""
    from pydantic import BaseModel

    from dynamiq.nodes.types import InferenceMode

    class Doc(BaseModel):
        title: str

    agent = _make_agent(inference_mode=InferenceMode.STRUCTURED_OUTPUT, response_format=Doc)

    output = '{"thought": "done", "action": "finish", ' '"action_input": "{\\"title\\": \\"HP\\"}", "output_files": ""}'
    thought, action, action_input = agent._handle_structured_output_mode(output, loop_num=1)
    assert action == "final_answer"
    coerced = agent._coerce_to_response_format(action_input)
    assert coerced == {"title": "HP"}
