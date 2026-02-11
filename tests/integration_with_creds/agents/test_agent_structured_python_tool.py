"""
Test ReAct agent with Python tool returning structured outputs (content + raw_response).
Demonstrates how agent sees only content while traces preserve all structured data.
"""

import pytest

from dynamiq import connections
from dynamiq.callbacks.tracing import TracingCallbackHandler
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms.openai import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.runnables import RunnableConfig


@pytest.fixture
def calculator_python_tool():
    """Python calculator tool that returns structured output with content + raw_response."""
    calculator_code = '''
def run(input_data):
    """
    Calculator tool that performs basic math operations.
    Returns structured output with content for agent and raw_response for traces.
    """
    expression = input_data.get('expression', '')

    try:
        # Simple calculator for basic operations (avoiding eval for security)
        if '+' in expression:
            parts = expression.split('+')
            result = float(parts[0].strip()) + float(parts[1].strip())
        elif '*' in expression:
            parts = expression.split('*')
            result = float(parts[0].strip()) * float(parts[1].strip())
        elif '-' in expression:
            parts = expression.split('-')
            result = float(parts[0].strip()) - float(parts[1].strip())
        elif '/' in expression:
            parts = expression.split('/')
            result = float(parts[0].strip()) / float(parts[1].strip())
        else:
            # Try to parse as a single number
            result = float(expression.strip())

        # Convert back to int if it's a whole number
        if result == int(result):
            result = int(result)

        # Return structured output
        return {
            'content': f'The result of {expression} is {result}',
            'raw_response': {
                'expression': expression,
                'result': result,
                'result_type': 'int' if isinstance(result, int) else 'float',
                'operation_status': 'success'
            }
        }
    except Exception as e:
        return {
            'content': f'Error calculating {expression}: {str(e)}',
            'raw_response': {
                'expression': expression,
                'error': str(e),
                'operation_status': 'error'
            }
        }
'''

    return Python(
        name="Calculator Tool",
        description="Performs mathematical calculations and returns detailed results",
        code=calculator_code,
    )


@pytest.fixture
def string_handler_python_tool():
    """Python string handler tool that returns structured output."""
    string_handler_code = '''
def run(input_data):
    """
    String manipulation tool that processes text.
    Returns structured output with content for agent and raw_response for traces.
    """
    text = input_data.get('text', '')
    operation = input_data.get('operation', 'analyze')

    if operation == 'analyze':
        word_count = len(text.split())
        char_count = len(text)

        return {
            'content': f'Text analysis: {word_count} words, {char_count} characters',
            'raw_response': {
                'original_text': text,
                'word_count': word_count,
                'char_count': char_count,
                'char_count_no_spaces': len(text.replace(' ', '')),
                'operation': operation,
                'status': 'completed'
            }
        }

    elif operation == 'uppercase':
        result = text.upper()
        return {
            'content': f'Converted to uppercase: {result}',
            'raw_response': {
                'original_text': text,
                'converted_text': result,
                'operation': operation,
                'status': 'completed'
            }
        }

    else:
        return {
            'content': f'Unknown operation: {operation}',
            'raw_response': {
                'original_text': text,
                'operation': operation,
                'error': f'Unsupported operation: {operation}',
                'status': 'error'
            }
        }
'''

    return Python(
        name="String Handler Tool", description="Processes and manipulates text strings", code=string_handler_code
    )


def test_react_agent_with_structured_python_tool(string_handler_python_tool):
    """Test ReAct agent using Python tool with structured output (content + raw_response)."""
    llm = OpenAI(model="gpt-5-mini", connection=connections.OpenAI(), temperature=0.1)

    agent = Agent(name="Text Assistant", llm=llm, tools=[string_handler_python_tool], max_loops=10)

    tracing_handler = TracingCallbackHandler()
    config = RunnableConfig(callbacks=[tracing_handler])

    result = agent.run(
        input_data={"input": "Use the string handler tool with operation='analyze' and text='Hello world from Python'"},
        config=config,
    )

    assert result.status.value == "success"
    assert "content" in result.output

    agent_content = result.output["content"]
    assert isinstance(agent_content, str)

    traces = tracing_handler.runs
    assert len(traces) > 0

    tool_traces = [
        trace
        for trace in traces.values()
        if trace.name == "String Handler Tool"
        and "raw_response" in trace.output
        and trace.output.get("raw_response", {}).get("status") == "completed"
    ]

    assert len(tool_traces) > 0
    tool_trace = tool_traces[0]

    tool_output = tool_trace.output
    assert "content" in tool_output
    assert "raw_response" in tool_output

    raw_response = tool_output["raw_response"]
    assert "original_text" in raw_response
    assert "word_count" in raw_response
    assert "char_count" in raw_response
    assert "operation" in raw_response
    assert "status" in raw_response
    assert raw_response["status"] == "completed"
    assert raw_response["operation"] == "analyze"


def test_python_tool_structured_output_vs_agent_processing(calculator_python_tool):
    """Test that Python tool returns structured output but agent processing extracts only content."""
    from dynamiq.nodes.agents.utils import process_tool_output_for_agent

    tool_result = calculator_python_tool.run({"expression": "10 + 5"})

    assert tool_result.status.value == "success"
    tool_output = tool_result.output

    assert "content" in tool_output
    assert "raw_response" in tool_output
    assert isinstance(tool_output["content"], str)
    assert isinstance(tool_output["raw_response"], dict)

    raw_response = tool_output["raw_response"]
    assert raw_response["expression"] == "10 + 5"
    assert raw_response["result"] == 15
    assert raw_response["operation_status"] == "success"

    processed_content = process_tool_output_for_agent(tool_output)

    assert isinstance(processed_content, str)
    assert "The result of 10 + 5 is 15" in processed_content

    assert tool_output["raw_response"]["result"] == 15


def test_agent_optimized_mode_preserves_structure(calculator_python_tool):
    """Test that agent-optimized mode preserves structure while ensuring content is string."""
    calculator_python_tool.is_optimized_for_agents = True

    result = calculator_python_tool.run({"expression": "20 * 3"})

    assert result.status.value == "success"
    output = result.output

    assert "content" in output
    assert "raw_response" in output

    assert isinstance(output["content"], str)
    assert "The result of 20 * 3 is 60" in output["content"]

    assert isinstance(output["raw_response"], dict)
    assert output["raw_response"]["result"] == 60
    assert output["raw_response"]["operation_status"] == "success"
