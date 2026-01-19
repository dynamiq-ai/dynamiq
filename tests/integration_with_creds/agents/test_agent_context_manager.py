"""
Integration tests for Context Manager Tool - automatic invocation.

Tests the agent's ability to automatically invoke the Context Manager Tool
when token limits are exceeded.
"""

import pytest

from dynamiq.connections import OpenAI as OpenAIConnection
from dynamiq.nodes.agents import Agent
from dynamiq.nodes.agents.prompts.react.instructions import PROMPT_AUTO_CLEAN_CONTEXT
from dynamiq.nodes.agents.utils import SummarizationConfig
from dynamiq.nodes.llms import OpenAI
from dynamiq.nodes.tools.python import Python
from dynamiq.nodes.types import InferenceMode
from dynamiq.runnables import RunnableConfig, RunnableStatus
from dynamiq.utils.logger import logger


@pytest.fixture(scope="module")
def llm_instance():
    """Create a real LLM instance for testing."""
    logger.info("\n--- Setting up REAL LLM (Fixture) ---")
    connection = OpenAIConnection()
    llm = OpenAI(
        name="test-llm",
        connection=connection,
        model="gpt-5-mini",
        temperature=0.1,
    )
    logger.info(f"LLM fixture ready: {llm.name} (model: {llm.model})")
    return llm


@pytest.fixture(scope="module")
def python_tool():
    """Create a Python tool for generating random word output."""
    code = """
def run(input_data):
    # Generate random word output to test summarization
    return "apple"
"""
    return Python(
        name="word-generator",
        code=code,
    )


@pytest.fixture(scope="module")
def run_config():
    """Create a runnable config with extended timeout."""
    return RunnableConfig(request_timeout=180)


def test_automatic_context_manager_invocation(llm_instance, python_tool, run_config):
    """
    Test automatic Context Manager Tool invocation when token limit is exceeded.

    This test verifies that the agent automatically invokes the ContextManagerTool
    when the conversation history exceeds the configured token limit.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Testing automatic Context Manager Tool invocation on token limit")
    logger.info(f"{'='*80}\n")

    # Create agent with REPLACE mode and LOW token limit to trigger automatic summarization
    agent = Agent(
        name="test-auto-summarization",
        llm=llm_instance,
        role="You are a helpful assistant that generates random words.",
        tools=[python_tool],
        max_loops=3,
        inference_mode=InferenceMode.XML,
        summarization_config=SummarizationConfig(
            enabled=True,
            max_token_context_length=5000,  # Low limit to trigger automatic summarization
            context_usage_ratio=0.5,
        ),
    )

    # Run task that generates lots of content to exceed token limit
    input_data = {
        "input": (
            "Results is exactly what word-generator tool returns. Please do the following:\n"
            "1. Use the word-generator tool to generate some content.\n"
            "2. Clean the context with a tool."
            "3. Finish execution with result of what word was returned."
        )
    }

    # Store initial message count
    initial_message_count = len(agent._prompt.messages)
    logger.info(f"Initial message count: {initial_message_count}")

    result = agent.run(input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS, f"Agent failed: {result.error}"
    logger.info("Agent completed successfully")

    # Check final message count
    final_message_count = len(agent._prompt.messages)
    logger.info(f"Final message count: {final_message_count}")

    logger.info(result.output)
    assert "apple" in result.output["content"], "Result is not correct"
    assert final_message_count == 5, "Final message count is not correct. Maybe context manager tool was not invoked."
    assert (
        agent.sanitize_tool_name(agent.tools[1].name) in agent._prompt.messages[-3].content.lower()
    ), "Context message not found"


def test_automatic_context_manager_auto_clean(llm_instance, python_tool, run_config):
    """
    Test automatic Context Manager Tool auto clean when token limit is exceeded.

    This test verifies that the agent automatically cleans the context when the
     conversation history exceeds the configured token limit.
    """
    logger.info(f"\n{'='*80}")
    logger.info("Testing automatic Context Manager Tool auto clean on token limit")
    logger.info(f"{'='*80}\n")

    # Create agent with REPLACE mode and LOW token limit to trigger automatic summarization
    agent = Agent(
        name="test-auto-summarization",
        llm=llm_instance,
        role="You are a helpful assistant that generates random words.",
        tools=[python_tool],
        max_loops=3,
        inference_mode=InferenceMode.XML,
        summarization_config=SummarizationConfig(
            enabled=True,
            max_token_context_length=1000,  # Low limit to trigger automatic summarization
            context_usage_ratio=0.5,
        ),
    )

    # Run task that generates lots of content to exceed token limit
    input_data = {
        "input": (
            "Results is exactly what word-generator tool. Please do the following:\n"
            "1. Use the word-generator tool to generate one random word.\n"
            "2. Finish execution right after summarization with result of what word was returned."
        )
    }

    # Store initial message count
    initial_message_count = len(agent._prompt.messages)
    logger.info(f"Initial message count: {initial_message_count}")

    result = agent.run(input_data, config=run_config)

    assert result.status == RunnableStatus.SUCCESS, f"Agent failed: {result.error}"
    logger.info("Agent completed successfully")

    # Check final message count
    final_message_count = len(agent._prompt.messages)
    logger.info(f"Final message count: {final_message_count}")

    logger.info(result.output)
    assert "apple" in result.output["content"], "Result is not correct"
    assert final_message_count == 5, "Final message count is not correct. Maybe context manager tool was not invoked."
    assert PROMPT_AUTO_CLEAN_CONTEXT == agent._prompt.messages[-3].content, "Auto clean context message not found"
