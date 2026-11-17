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


@pytest.mark.flaky(reruns=3)
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
