import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


@pytest.fixture(scope="module")
def llm_instance():
    """Setup LLM instance for testing."""
    logger.info("\n--- Setting up LLM (Fixture) ---")
    connection = OpenAIConnection()
    llm = OpenAI(
        connection=connection,
        model="gpt-5-mini",
        max_tokens=2000,
        temperature=0,
    )
    return llm


@pytest.fixture(scope="module")
def agent_role():
    """Basic agent role for testing."""
    return "You are a helpful assistant that provides accurate information."


@pytest.fixture(scope="module")
def run_config():
    """Standard runnable configuration."""
    return RunnableConfig(request_timeout=120)


@pytest.fixture(scope="module")
def calculator_tool():
    """Simple calculator tool for testing."""
    code = """
def run(input_data):
    expression = input_data.get('expression', '')
    try:
        result = eval(expression)
        return f"The result is: {result}"
    except Exception as e:
        return f"Error: {str(e)}"
"""
    return Python(
        name="Calculator",
        description="Evaluates mathematical expressions. Input parameter: 'expression' (string).",
        code=code,
    )


@pytest.fixture(scope="module")
def data_formatter_tool():
    """Tool that returns structured data for delegation."""
    code = """
def run(input_data):
    data_type = input_data.get('type', 'summary')
    if data_type == 'summary':
        return {
            'title': 'Data Summary',
            'items': ['item1', 'item2', 'item3'],
            'count': 3
        }
    else:
        return {'raw_data': [1, 2, 3, 4, 5]}
"""
    return Python(
        name="DataFormatter",
        description="Returns formatted data. Input parameter: 'type' (string: 'summary' or 'raw').",
        code=code,
    )


@pytest.mark.integration
def test_agent_direct_final_output(llm_instance, agent_role, run_config):
    """Test that agent can provide final answer directly without tool calls."""
    agent = Agent(
        name="DirectOutputAgent",
        id="direct_output_agent",
        llm=llm_instance,
        role=agent_role,
        inference_mode=InferenceMode.DEFAULT,
        tools=[],
        direct_output=True,
        verbose=True,
    )

    input_data = {"input": "What is the capital of France? Provide a direct answer."}
    result = agent.run(input_data=input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS, f"Agent failed with status: {result.status}"
    content = result.output["content"]
    assert isinstance(content, str), f"Output should be string, got {type(content)}"
    assert "paris" in content.lower(), f"Expected 'Paris' in output, got: {content}"
    logger.info(f"Agent direct output: {content}")


@pytest.mark.integration
def test_agent_delegates_to_agent_tool_then_returns(llm_instance, agent_role, calculator_tool, run_config):
    """Test agent delegates to another agent (used as a tool) and returns the result as final answer."""
    # Create a calculator agent that will be used as a tool
    calculator_agent = Agent(
        name="CalculatorAgentTool",
        id="calculator_agent_tool",
        llm=llm_instance,
        role="You are a mathematical calculator assistant. Use the calculator tool to perform calculations accurately.",
        inference_mode=InferenceMode.DEFAULT,
        tools=[calculator_tool],
        verbose=True,
    )

    # Create main agent that will delegate to the calculator agent
    main_agent = Agent(
        name="DelegatorAgent",
        id="delegator_agent",
        llm=llm_instance,
        role="You are a helpful assistant. When users ask mathematical questions, delegate to the CalculatorAgentTool.",
        inference_mode=InferenceMode.DEFAULT,
        tools=[calculator_agent],  # Use the calculator agent as a tool
        verbose=True,
    )

    input_data = {"input": "What is 25 * 4? Please calculate this for me."}
    result = main_agent.run(input_data=input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert isinstance(content, str)
    # Should contain the result 100
    assert "100" in content, f"Expected '100' in output, got: {content}"
    logger.info(f"Agent-to-agent delegation result: {content}")


@pytest.mark.integration
def test_agent_delegates_structured_output(llm_instance, agent_role, data_formatter_tool, run_config):
    """Test agent can delegate to tool that returns structured data and format final output."""
    agent = Agent(
        name="DataAgent",
        id="data_agent",
        llm=llm_instance,
        role="You are a data assistant that helps users get formatted data.",
        inference_mode=InferenceMode.DEFAULT,
        tools=[data_formatter_tool],
        verbose=True,
    )

    input_data = {"input": "Get me a summary type data using the DataFormatter tool."}
    result = agent.run(input_data=input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert isinstance(content, str)

    # Agent should present the tool's structured output in its final answer
    # The tool returns {'title': 'Data Summary', 'items': [...], 'count': 3}
    assert any(
        keyword in content.lower() for keyword in ["summary", "item", "data"]
    ), f"Expected summary-related content, got: {content}"
    logger.info(f"Agent structured output delegation: {content}")


@pytest.mark.integration
@pytest.mark.parametrize(
    "inference_mode",
    [
        InferenceMode.DEFAULT,
        InferenceMode.XML,
        InferenceMode.FUNCTION_CALLING,
    ],
    ids=["default", "xml", "function_calling"],
)
def test_agent_direct_output_across_inference_modes(llm_instance, agent_role, run_config, inference_mode):
    """Test direct output works across different inference modes."""
    agent = Agent(
        name=f"DirectOutput_{inference_mode.value}",
        id=f"direct_output_{inference_mode.value.lower()}",
        llm=llm_instance,
        role=agent_role,
        inference_mode=inference_mode,
        tools=[],
        verbose=True,
    )

    input_data = {"input": "What is 2 + 2? Answer directly."}
    result = agent.run(input_data=input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS, f"Agent failed for {inference_mode.value}: {result.status}"
    content = result.output["content"]
    assert isinstance(content, str)
    assert "4" in content, f"Expected '4' in output for {inference_mode.value}, got: {content}"
    logger.info(f"Direct output test passed for {inference_mode.value}")


@pytest.mark.integration
def test_agent_delegates_multiple_tools_then_synthesizes(
    llm_instance, agent_role, calculator_tool, data_formatter_tool, run_config
):
    """Test agent can use multiple tools and synthesize a final answer."""
    agent = Agent(
        name="MultiToolAgent",
        id="multi_tool_agent",
        llm=llm_instance,
        role="You are an assistant that can perform calculations and format data.",
        inference_mode=InferenceMode.DEFAULT,
        tools=[calculator_tool, data_formatter_tool],
        verbose=True,
    )

    input_data = {
        "input": (
            "First, calculate 10 * 5 using the Calculator. "
            "Then provide me with a final answer that includes the result."
        )
    }
    result = agent.run(input_data=input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert isinstance(content, str)
    assert "50" in content, f"Expected calculation result '50' in output, got: {content}"
    logger.info(f"Multi-tool synthesis output: {content}")


@pytest.mark.integration
def test_agent_chooses_direct_output_over_unnecessary_tools(llm_instance, calculator_tool, run_config):
    """Test agent provides direct answer when tools aren't needed."""
    agent = Agent(
        name="SmartAgent",
        id="smart_agent",
        llm=llm_instance,
        role="You are an intelligent assistant. Use tools only when necessary.",
        inference_mode=InferenceMode.DEFAULT,
        tools=[calculator_tool],
        verbose=True,
    )

    # Simple question that doesn't need a calculator
    input_data = {"input": "What is the capital of Italy?"}
    result = agent.run(input_data=input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert isinstance(content, str)
    assert "rome" in content.lower(), f"Expected 'Rome' in output, got: {content}"

    # Agent should answer directly without using calculator
    logger.info(f"Smart direct output (no tool needed): {content}")


@pytest.mark.integration
def test_agent_output_formatting_after_delegation(llm_instance, calculator_tool, run_config):
    """Test agent formats tool output nicely in final response."""
    agent = Agent(
        name="FormattingAgent",
        id="formatting_agent",
        llm=llm_instance,
        role="You are a helpful assistant. Always format your final answers clearly and professionally.",
        inference_mode=InferenceMode.DEFAULT,
        tools=[calculator_tool],
        verbose=True,
    )

    input_data = {"input": "Calculate 123 + 456 and present the result in a clear, formatted way."}
    result = agent.run(input_data=input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert isinstance(content, str)
    assert "579" in content, f"Expected '579' in output, got: {content}"

    # Check that output is well-formatted (not just raw tool output)
    assert len(content) > 10, "Output should be more than just the number"
    logger.info(f"Formatted output: {content}")


@pytest.mark.integration
def test_agent_handles_empty_tool_response(llm_instance, run_config):
    """Test agent handles tools that return empty/minimal responses."""
    empty_tool = Python(
        name="EmptyTool",
        description="A tool that returns minimal output.",
        code="""
def run(input_data):
    return ""
""",
    )

    agent = Agent(
        name="EmptyResponseAgent",
        id="empty_response_agent",
        llm=llm_instance,
        role="You are a helpful assistant. Provide meaningful responses even when tools return empty results.",
        inference_mode=InferenceMode.DEFAULT,
        tools=[empty_tool],
        verbose=True,
    )

    input_data = {"input": "Use the EmptyTool and tell me what happened."}
    result = agent.run(input_data=input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS
    content = result.output["content"]
    assert isinstance(content, str)
    assert len(content) > 0, "Agent should provide output even when tool returns empty result"
    logger.info(f"Handled empty tool response: {content}")
